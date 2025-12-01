"""
Main benchmark runner for multi-agent debate experiments
"""
import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, List
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from datasets import load_dataset
from langgraph.graph import StateGraph

from src.debate.graph import build_graph, DebateState
from src.debate.prompts import parse_json_or_fallback, normalize_probs

def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging configuration"""
    # Create log file path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"benchmark_run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_dataset_from_config(dataset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load dataset based on configuration"""
    try:
        if dataset_config['type'] == 'hf':
            dataset = load_dataset(
                dataset_config['name'],
                dataset_config.get('subset'),
                split=dataset_config['split']
            )
            
            # Convert to list of examples
            examples = []
            for item in dataset:
                example = {
                    'question': item.get('question', ''),
                    'choices': {},
                    'answer': item.get('answer', ''),
                    'id': item.get('id', '')
                }
                
                # Handle different dataset formats
                if 'choices' in item:
                    choices = item['choices']
                    if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
                        # Format: {"text": [...], "label": [...]} (CommonSenseQA, arithmetic)
                        text_list = choices['text']
                        label_list = choices['label']
                        choices_dict = {}
                        for i, label in enumerate(label_list):
                            if i < len(text_list):
                                choices_dict[label] = text_list[i]
                        example['choices'] = choices_dict
                    elif isinstance(choices, list):
                        # Format: ["choice1", "choice2", ...] (MMLU)
                        choices_dict = {}
                        for i, choice in enumerate(choices):
                            choices_dict[chr(ord('A') + i)] = choice
                        example['choices'] = choices_dict
                    elif isinstance(choices, dict):
                        # Direct dict format
                        example['choices'] = choices
                elif 'endings' in item:
                    # HellaSwag format
                    endings = item['endings']
                    choices_dict = {}
                    for i, ending in enumerate(endings):
                        choices_dict[chr(ord('A') + i)] = ending
                    example['choices'] = choices_dict
                    example['question'] = item.get('ctx', '') + ' ' + item.get('ctx_b', '')
                
                # Handle answer format
                if 'answer' in item:
                    answer = item['answer']
                    if isinstance(answer, str) and len(answer) == 1:
                        example['answer'] = answer.upper()
                    elif isinstance(answer, int):
                        example['answer'] = chr(ord('A') + answer)
                elif 'answerKey' in item:
                    example['answer'] = item['answerKey']
                elif 'label' in item:
                    # HellaSwag uses 'label' for answer
                    label = item['label']
                    if isinstance(label, str) and label.isdigit():
                        example['answer'] = chr(ord('A') + int(label))
                    else:
                        example['answer'] = label
                
                examples.append(example)
            
            # Apply max_examples limit if specified
            max_examples = dataset_config.get('max_examples')
            if max_examples and len(examples) > max_examples:
                import random
                random.seed(42)  # For reproducible sampling
                examples = random.sample(examples, max_examples)
            
            return examples
            
        elif dataset_config['type'] == 'local':
            # Load local JSONL file
            import json
            examples = []
            with open(dataset_config['path'], 'r') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        
                        # Handle different choice formats
                        choices = item.get('choices', {})
                        if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
                            # Convert from {"text": [...], "label": [...]} format to {"A": "...", "B": "...", ...}
                            text_list = choices['text']
                            label_list = choices['label']
                            choices_dict = {}
                            for i, label in enumerate(label_list):
                                if i < len(text_list):
                                    choices_dict[label] = text_list[i]
                            choices = choices_dict
                        
                        example = {
                            'question': item.get('question', ''),
                            'choices': choices,
                            'answer': item.get('answer', ''),
                            'id': item.get('id', '')
                        }
                        examples.append(example)
            
            # Apply max_examples limit if specified
            max_examples = dataset_config.get('max_examples')
            if max_examples and len(examples) > max_examples:
                examples = examples[:max_examples]  # Take first N for local files
            
            return examples
            
    except Exception as e:
        print(f"Error loading dataset {dataset_config.get('name', dataset_config.get('path', 'unknown'))}: {e}")
        return []

def run_single_debate(example: Dict[str, Any], pairing: str, cfg_models: Dict, 
                     cfg_prompts: Dict, cfg_run: Dict) -> Dict[str, Any]:
    """Run a single debate for one example"""
    try:
        print(f"    Starting debate for example {example.get('id', 'unknown')}")
        print(f"    Question: {example['question'][:100]}...")
        print(f"    Choices: {example['choices']}")
        print(f"    Answer: {example['answer']}")
        
        # Print prompt lengths for this debate
        print(f"    Prompt lengths:")
        for prompt_name, prompt_content in cfg_prompts.items():
            if isinstance(prompt_content, str):
                print(f"      {prompt_name}: {len(prompt_content)} characters")
        
        # Build the debate graph
        graph = build_graph(cfg_prompts, cfg_models, pairing, cfg_run.get('with_judge', True))
        
        # Prepare initial state
        state = {
            'question': example['question'],
            'choices': example['choices'],
            'answer': example['answer'],
            'id': example['id'],
            'sys_debater': cfg_prompts['system_debater'],
            'sys_judge': cfg_prompts['system_judge'],
            'judge_crit_instructions': cfg_prompts['judge_crit_instructions'],
            'u_r1_A': cfg_prompts['user_round1_A'],
            'u_r1_B': cfg_prompts['user_round1_B'],
            'u_judge_r1': cfg_prompts['user_judge_r1'],
            'u_r2_A': cfg_prompts['user_round2_A'],
            'u_r2_B': cfg_prompts['user_round2_B'],
            'u_judge_r2': cfg_prompts['user_judge_r2'],
            'u_r3_A': cfg_prompts['user_round3_A'],
            'u_r3_B': cfg_prompts['user_round3_B'],
            'u_judge_r3': cfg_prompts['user_judge_r3'],
            'u_r4_A': cfg_prompts['user_round4_A'],
            'u_r4_B': cfg_prompts['user_round4_B'],
            'u_judge_r4': cfg_prompts['user_judge_r4'],
            'u_r5_A': cfg_prompts['user_round5_A'],
            'u_r5_B': cfg_prompts['user_round5_B'],
            'u_judge_r5': cfg_prompts['user_judge_r5'],
            'u_r6_A': cfg_prompts['user_round6_A'],
            'u_r6_B': cfg_prompts['user_round6_B'],
            'u_judge_r6': cfg_prompts['user_judge_r6'],
        }
        
        print(f"    Running debate graph...")
        # Run the debate
        result = graph.compile().invoke(state)
        print(f"    Debate completed successfully!")
        
        return {
            'example_id': example['id'],
            'pairing': pairing,
            'question': example['question'],
            'choices': example['choices'],
            'answer': example['answer'],
            'debate_state': result,
            'timestamp': time.time()
        }
        
    except Exception as e:
        print(f"Error in debate for example {example.get('id', 'unknown')}: {e}")
        traceback.print_exc()
        return {
            'example_id': example.get('id', 'unknown'),
            'pairing': pairing,
            'error': str(e),
            'timestamp': time.time()
        }

def run_benchmark(benchmark_config: Dict, models_config: Dict, 
                 datasets_config: Dict, prompts_config: Dict):
    """Run the complete benchmark"""
    
    # Create output directories
    output_dir_runs = Path(benchmark_config['io']['output_dir_runs'])
    output_dir_metrics = Path(benchmark_config['io']['output_dir_metrics'])
    cache_dir = Path(benchmark_config['io']['cache_dir'])
    
    output_dir_runs.mkdir(parents=True, exist_ok=True)
    output_dir_metrics.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir_metrics)
    logger.info("Starting benchmark run")
    
    # Print prompt lengths
    logger.info("Prompt lengths:")
    total_length = 0
    prompt_lengths = {}
    for prompt_name, prompt_content in prompts_config.items():
        if isinstance(prompt_content, str):
            length = len(prompt_content)
            prompt_lengths[prompt_name] = length
            total_length += length
            logger.info(f"  {prompt_name}: {length} characters")
        else:
            logger.info(f"  {prompt_name}: {type(prompt_content)} (not a string)")
    
    # Print summary statistics
    if prompt_lengths:
        avg_length = total_length / len(prompt_lengths)
        max_length = max(prompt_lengths.values())
        min_length = min(prompt_lengths.values())
        logger.info(f"Prompt statistics:")
        logger.info(f"  Total characters: {total_length}")
        logger.info(f"  Number of prompts: {len(prompt_lengths)}")
        logger.info(f"  Average length: {avg_length:.1f} characters")
        logger.info(f"  Max length: {max_length} characters")
        logger.info(f"  Min length: {min_length} characters")
    
    # Create dataset subfolders in metrics directory
    for dataset_name in benchmark_config['datasets']:
        dataset_metrics_dir = output_dir_metrics / dataset_name
        dataset_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    random.seed(benchmark_config['run']['seed'])
    
    # Load datasets
    datasets = {}
    for dataset_name in benchmark_config['datasets']:
        if dataset_name in datasets_config:
            logger.info(f"Loading dataset: {dataset_name}")
            dataset_examples = load_dataset_from_config(datasets_config[dataset_name])
            
            # Apply max_examples limit
            max_examples = benchmark_config['run']['max_examples']
            if max_examples and len(dataset_examples) > max_examples:
                if benchmark_config['run']['shuffle']:
                    random.shuffle(dataset_examples)
                dataset_examples = dataset_examples[:max_examples]
            
            datasets[dataset_name] = dataset_examples
            logger.info(f"Loaded {len(dataset_examples)} examples for {dataset_name}")
    
    # Run experiments
    all_results = []
    
    for pairing in benchmark_config['pairings']:
        logger.info(f"Running pairing: {pairing}")
        
        for dataset_name, examples in datasets.items():
            logger.info(f"  Dataset: {dataset_name} ({len(examples)} examples)")
            
            # Run debates in parallel
            with ThreadPoolExecutor(max_workers=benchmark_config['run']['num_workers']) as executor:
                futures = []
                
                for example in examples:
                    future = executor.submit(
                        run_single_debate,
                        example,
                        pairing,
                        models_config,
                        prompts_config,
                        benchmark_config['run']
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        # Save individual result with new naming convention
                        # Format: pairname_datasetnumber_examplenumber.json
                        example_number = i + 1
                        result_file = output_dir_metrics / dataset_name / f"{pairing}_{dataset_name}_{example_number}.json"
                        with open(result_file, 'w') as f:
                            json.dump(result, f, indent=2)
                            
                        logger.info(f"    Saved: {result_file}")
                            
                    except Exception as e:
                        logger.error(f"Error collecting result: {e}")
    
    # Save all results
    results_file = output_dir_metrics / "all_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("Benchmark completed!")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Individual runs saved to: {output_dir_runs}")

def main():
    parser = argparse.ArgumentParser(description="Run multi-agent debate benchmark")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark config")
    parser.add_argument("--models", required=True, help="Path to models config")
    parser.add_argument("--datasets", required=True, help="Path to datasets config")
    parser.add_argument("--prompts", required=True, help="Path to prompts config")
    
    args = parser.parse_args()
    
    # Load configurations
    benchmark_config = load_config(args.benchmark)
    models_config = load_config(args.models)
    datasets_config = load_config(args.datasets)
    prompts_config = load_config(args.prompts)
    
    # Run benchmark
    run_benchmark(benchmark_config, models_config, datasets_config, prompts_config)

if __name__ == "__main__":
    main()
