#!/usr/bin/env python3
"""
Download all datasets used in the benchmark to the data/ subfolder
"""
import os
import json
from pathlib import Path
from datasets import load_dataset
import argparse

def download_mmlu_professional_medicine():
    """Download MMLU Professional Medicine dataset"""
    print("📥 Downloading MMLU Professional Medicine...")
    
    output_dir = Path("data/mmlu_professional_medicine")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('cais/mmlu', 'professional_medicine', split='test')
    
    # Convert to JSONL format
    with open(output_dir / "test.jsonl", "w") as f:
        for i, example in enumerate(dataset):
            # Convert to our format
            choices_dict = {}
            for j, choice in enumerate(example['choices']):
                choices_dict[chr(ord('A') + j)] = choice
            
            formatted_example = {
                "id": f"mmlu_pro_med_{i}",
                "question": example['question'],
                "choices": choices_dict,
                "answer": chr(ord('A') + example['answer'])
            }
            f.write(json.dumps(formatted_example) + "\n")
    
    print(f"✅ Downloaded {len(dataset)} examples to {output_dir}/test.jsonl")

def download_mmlu_formal_logic():
    """Download MMLU Formal Logic dataset"""
    print("📥 Downloading MMLU Formal Logic...")
    
    output_dir = Path("data/mmlu_formal_logic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('cais/mmlu', 'formal_logic', split='test')
    
    # Convert to JSONL format
    with open(output_dir / "test.jsonl", "w") as f:
        for i, example in enumerate(dataset):
            # Convert to our format
            choices_dict = {}
            for j, choice in enumerate(example['choices']):
                choices_dict[chr(ord('A') + j)] = choice
            
            formatted_example = {
                "id": f"mmlu_formal_logic_{i}",
                "question": example['question'],
                "choices": choices_dict,
                "answer": chr(ord('A') + example['answer'])
            }
            f.write(json.dumps(formatted_example) + "\n")
    
    print(f"✅ Downloaded {len(dataset)} examples to {output_dir}/test.jsonl")

def download_hellaswag():
    """Download HellaSwag dataset"""
    print("📥 Downloading HellaSwag...")
    
    output_dir = Path("data/hellaswag")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('Rowan/hellaswag', split='validation')
    
    # Convert to JSONL format
    with open(output_dir / "validation.jsonl", "w") as f:
        for i, example in enumerate(dataset):
            # Convert endings to choices format
            choices_dict = {}
            for j, ending in enumerate(example['endings']):
                choices_dict[chr(ord('A') + j)] = ending
            
            formatted_example = {
                "id": f"hellaswag_{i}",
                "question": example['ctx'],
                "choices": choices_dict,
                "answer": example['label']
            }
            f.write(json.dumps(formatted_example) + "\n")
    
    print(f"✅ Downloaded {len(dataset)} examples to {output_dir}/validation.jsonl")

def download_commonsenseqa():
    """Download CommonSenseQA dataset"""
    print("📥 Downloading CommonSenseQA...")
    
    output_dir = Path("data/commonsenseqa")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('commonsense_qa', split='validation')
    
    # Convert to JSONL format
    with open(output_dir / "validation.jsonl", "w") as f:
        for i, example in enumerate(dataset):
            # Convert choices to our format
            choices_dict = {}
            for j, choice in enumerate(example['choices']['text']):
                choices_dict[example['choices']['label'][j]] = choice
            
            formatted_example = {
                "id": f"commonsenseqa_{i}",
                "question": example['question'],
                "choices": choices_dict,
                "answer": example['answerKey']
            }
            f.write(json.dumps(formatted_example) + "\n")
    
    print(f"✅ Downloaded {len(dataset)} examples to {output_dir}/validation.jsonl")

def download_gsm8k():
    """Download GSM8K dataset"""
    print("📥 Downloading GSM8K...")
    
    output_dir = Path("data/gsm8k")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('gsm8k', 'main', split='test')
    
    # Convert to JSONL format
    with open(output_dir / "test.jsonl", "w") as f:
        for i, example in enumerate(dataset):
            # GSM8K is open-form, so we'll format it for multiple choice
            # We'll create dummy choices for now
            choices_dict = {
                "A": f"Answer: {example['answer']}",
                "B": "Answer: [Incorrect option]",
                "C": "Answer: [Incorrect option]", 
                "D": "Answer: [Incorrect option]"
            }
            
            formatted_example = {
                "id": f"gsm8k_{i}",
                "question": example['question'],
                "choices": choices_dict,
                "answer": "A"  # Since we put the correct answer in A
            }
            f.write(json.dumps(formatted_example) + "\n")
    
    print(f"✅ Downloaded {len(dataset)} examples to {output_dir}/test.jsonl")

def download_hh_rlhf():
    """Download HH-RLHF dataset"""
    print("📥 Downloading HH-RLHF...")
    
    output_dir = Path("data/hh_rlhf")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_dataset('Anthropic/hh-rlhf', split='test')
    
    # Convert to JSONL format
    with open(output_dir / "test.jsonl", "w") as f:
        for i, example in enumerate(dataset):
            # HH-RLHF is open-form, so we'll format it for multiple choice
            choices_dict = {
                "A": example['chosen'],
                "B": example['rejected'],
                "C": "[Other option 1]",
                "D": "[Other option 2]"
            }
            
            formatted_example = {
                "id": f"hh_rlhf_{i}",
                "question": "Which response is more helpful and harmless?",
                "choices": choices_dict,
                "answer": "A"  # Chosen is typically better
            }
            f.write(json.dumps(formatted_example) + "\n")
    
    print(f"✅ Downloaded {len(dataset)} examples to {output_dir}/test.jsonl")

def download_all_datasets():
    """Download all datasets"""
    print("🚀 Starting download of all datasets...")
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Download each dataset
    try:
        download_mmlu_professional_medicine()
        download_mmlu_formal_logic()
        download_hellaswag()
        download_commonsenseqa()
        download_gsm8k()
        download_hh_rlhf()
        
        print("\n🎉 All datasets downloaded successfully!")
        print("\n📁 Dataset locations:")
        print("  - MMLU Professional Medicine: data/mmlu_professional_medicine/test.jsonl")
        print("  - MMLU Formal Logic: data/mmlu_formal_logic/test.jsonl")
        print("  - HellaSwag: data/hellaswag/validation.jsonl")
        print("  - CommonSenseQA: data/commonsenseqa/validation.jsonl")
        print("  - GSM8K: data/gsm8k/test.jsonl")
        print("  - HH-RLHF: data/hh_rlhf/test.jsonl")
        print("  - Arithmetic: data/arithmetic/dev.jsonl (already exists)")
        
    except Exception as e:
        print(f"❌ Error downloading datasets: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Download datasets for the benchmark')
    parser.add_argument('--dataset', choices=['mmlu_pro_med', 'mmlu_formal_logic', 'hellaswag', 'commonsenseqa', 'gsm8k', 'hh_rlhf', 'all'], 
                       default='all', help='Which dataset to download')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        download_all_datasets()
    elif args.dataset == 'mmlu_pro_med':
        download_mmlu_professional_medicine()
    elif args.dataset == 'mmlu_formal_logic':
        download_mmlu_formal_logic()
    elif args.dataset == 'hellaswag':
        download_hellaswag()
    elif args.dataset == 'commonsenseqa':
        download_commonsenseqa()
    elif args.dataset == 'gsm8k':
        download_gsm8k()
    elif args.dataset == 'hh_rlhf':
        download_hh_rlhf()

if __name__ == "__main__":
    main()
