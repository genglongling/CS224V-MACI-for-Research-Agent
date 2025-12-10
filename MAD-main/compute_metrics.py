#!/usr/bin/env python3
"""
Script to compute Diversity, Coherence, Conflict Surface, and Factual Grounding metrics
for all experiment results using GPT-4o.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import openai
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EVALUATION_PROMPT = """You are an expert evaluator scoring research reports on a scale of 1-5.

Report:
{report}

Evaluate this report on the following four dimensions (each on a scale of 1-5):

1. **Diversity** (1-5): How well does the report cover distinct perspectives, viewpoints, and alternative approaches? Does it explore the full semantic space of the problem?
   - 1: Single perspective, no alternatives considered
   - 3: Multiple perspectives mentioned but not deeply explored
   - 5: Comprehensive coverage of diverse viewpoints with clear distinctions

2. **Coherence** (1-5): How well-structured and logically flowing is the report? Is it easy to follow the argumentation?
   - 1: Disjointed, confusing structure
   - 3: Generally organized but some gaps in flow
   - 5: Clear, logical structure with smooth transitions

3. **Conflict Surface** (1-5): How explicitly does the report identify and analyze trade-offs, disagreements, and conflicts between perspectives?
   - 1: No conflicts identified, just lists opinions
   - 3: Some conflicts mentioned but not deeply analyzed
   - 5: Explicit identification and analysis of why perspectives differ, with clear trade-offs

4. **Factual Grounding** (1-5): How well-supported are claims with specific evidence, logic, examples, or citations?
   - 1: Vague claims with no support
   - 3: Some evidence but often generic
   - 5: Well-grounded with specific evidence, examples, or logical reasoning

Output STRICT JSON only:
{{
  "diversity": <float 1-5>,
  "coherence": <float 1-5>,
  "conflict_surface": <float 1-5>,
  "factual_grounding": <float 1-5>
}}
"""

def evaluate_report(report_text: str) -> Dict[str, float]:
    """Evaluate a single report using GPT-4o."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Output only valid JSON."},
                {"role": "user", "content": EVALUATION_PROMPT.format(report=report_text)}
            ],
            temperature=0.2,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        # Extract JSON if wrapped in code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return {
            "diversity": float(result.get("diversity", 0)),
            "coherence": float(result.get("coherence", 0)),
            "conflict_surface": float(result.get("conflict_surface", 0)),
            "factual_grounding": float(result.get("factual_grounding", 0))
        }
    except Exception as e:
        print(f"Error evaluating report: {e}")
        return {
            "diversity": 0.0,
            "coherence": 0.0,
            "conflict_surface": 0.0,
            "factual_grounding": 0.0
        }

def process_results_directory(results_dir: Path, baseline_dir: Path, output_file: Path):
    """Process all JSON files in results directory and compute metrics."""
    all_metrics = {}
    
    # Process each category
    for category in ["finance", "ai_governance", "social_policy"]:
        category_dir = results_dir / category
        baseline_category_dir = baseline_dir / category if baseline_dir else None
        
        if not category_dir.exists():
            continue
            
        all_metrics[category] = {}
        
        # Get all result JSON files (excluding briefs and logs)
        result_files = [f for f in category_dir.glob("*.json") 
                       if not f.name.startswith("interactive_debate") 
                       and not f.name.startswith("experiment_log")]
        
        for result_file in sorted(result_files):
            print(f"Processing {result_file}...")
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                question_id = data.get("question_type_id", "unknown")
                
                # Evaluate baseline report (from results file)
                baseline_report = data.get("baseline", {}).get("report", "")
                if not baseline_report and baseline_category_dir:
                    # Try to load from baseline directory
                    baseline_file = baseline_category_dir / result_file.name
                    if baseline_file.exists():
                        with open(baseline_file, 'r') as bf:
                            baseline_data = json.load(bf)
                            baseline_report = baseline_data.get("baseline", {}).get("report", "")
                
                if baseline_report:
                    baseline_metrics = evaluate_report(baseline_report)
                    print(f"  Baseline metrics: {baseline_metrics}")
                else:
                    baseline_metrics = None
                
                # Evaluate CollectiveMind report
                cm_report = data.get("collectivemind", {}).get("final_report", "")
                if cm_report:
                    cm_metrics = evaluate_report(cm_report)
                    print(f"  CollectiveMind metrics: {cm_metrics}")
                else:
                    cm_metrics = None
                
                # Get win rate if available
                win_rate = None
                if "evaluation" in data:
                    eval_data = data["evaluation"]
                    wins = eval_data.get("wins", 0)
                    losses = eval_data.get("losses", 0)
                    if wins + losses > 0:
                        win_rate = wins / (wins + losses)
                
                all_metrics[category][question_id] = {
                    "baseline": baseline_metrics,
                    "collectivemind": cm_metrics,
                    "win_rate": win_rate
                }
                
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save aggregated metrics
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Compute averages
    print("\n" + "="*80)
    print("AGGREGATED METRICS")
    print("="*80)
    
    baseline_totals = {"diversity": [], "coherence": [], "conflict_surface": [], "factual_grounding": []}
    cm_totals = {"diversity": [], "coherence": [], "conflict_surface": [], "factual_grounding": []}
    win_rates = []
    
    for category in all_metrics:
        for qid, metrics in all_metrics[category].items():
            if metrics["baseline"]:
                for key in baseline_totals:
                    baseline_totals[key].append(metrics["baseline"][key])
            if metrics["collectivemind"]:
                for key in cm_totals:
                    cm_totals[key].append(metrics["collectivemind"][key])
            if metrics["win_rate"] is not None:
                win_rates.append(metrics["win_rate"])
    
    print("\nBaseline (averages):")
    for key in baseline_totals:
        if baseline_totals[key]:
            avg = sum(baseline_totals[key]) / len(baseline_totals[key])
            print(f"  {key}: {avg:.2f}")
    
    print("\nCollectiveMind (averages):")
    for key in cm_totals:
        if cm_totals[key]:
            avg = sum(cm_totals[key]) / len(cm_totals[key])
            print(f"  {key}: {avg:.2f}")
    
    if win_rates:
        avg_win_rate = sum(win_rates) / len(win_rates)
        print(f"\nWin Rate: {avg_win_rate:.3f}")
    
    return all_metrics

if __name__ == "__main__":
    results_dir = Path("results/experiment")
    baseline_dir = Path("results(baselines)/experiment")
    output_file = Path("results/metrics_aggregated.json")
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    if not baseline_dir.exists():
        print(f"Warning: Baseline directory not found: {baseline_dir}")
        baseline_dir = None
    
    metrics = process_results_directory(results_dir, baseline_dir, output_file)
    print(f"\nMetrics saved to {output_file}")

