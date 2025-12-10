#!/usr/bin/env python3
"""
Compute ablation study metrics comparing:
1. Baseline (all GPT-4o) vs. Improved (GPT-4o, Claude, Gemini mix)
2. With research agent vs. Without research agent
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Import evaluation functions from evaluate_reports_directly
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_reports_directly import evaluate_report

def count_key_points_with_evidence(key_points: List[Dict]) -> tuple:
    """Count total key points and those with evidence."""
    total = len(key_points)
    with_evidence = sum(1 for kp in key_points if kp.get("evidence") and kp.get("evidence") != "null" and kp.get("evidence").strip())
    return total, with_evidence

def compute_win_rate(wins: int, losses: int, ties: int) -> float:
    """Compute win rate: wins / (wins + losses)."""
    total = wins + losses
    if total == 0:
        return 0.0
    return wins / total

def process_results_for_ablation(results_dir: Path) -> Dict[str, Any]:
    """Process results and compute metrics for ablation study."""
    all_metrics = {
        "diversity": [],
        "coherence": [],
        "conflict_surface": [],
        "factual_grounding": [],
        "key_points": [],
        "evidence_at_key": [],
        "wins": [],
        "losses": [],
        "ties": []
    }
    
    categories = ["finance", "ai_governance", "social_policy"]
    
    for category in categories:
        category_dir = results_dir / category
        if not category_dir.exists():
            continue
        
        # Get all result JSON files
        result_files = [f for f in category_dir.glob("*.json") 
                       if not f.name.startswith("interactive_debate") 
                       and not f.name.startswith("experiment_log")]
        
        for result_file in sorted(result_files):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Evaluate CollectiveMind report
                cm_report = data.get("collectivemind", {}).get("final_report", "")
                if cm_report:
                    cm_metrics = evaluate_report(cm_report, is_baseline=False)
                    all_metrics["diversity"].append(cm_metrics["diversity"])
                    all_metrics["coherence"].append(cm_metrics["coherence"])
                    all_metrics["conflict_surface"].append(cm_metrics["conflict_surface"])
                    all_metrics["factual_grounding"].append(cm_metrics["factual_grounding"])
                
                # Extract key points
                cm_kp_list = data.get("collectivemind", {}).get("key_points", [])
                cm_total, cm_with_evidence = count_key_points_with_evidence(cm_kp_list)
                all_metrics["key_points"].append(cm_total)
                if cm_total > 0:
                    all_metrics["evidence_at_key"].append(cm_with_evidence / cm_total)
                
                # Extract win rate data
                evaluation = data.get("evaluation", {})
                wins = evaluation.get("wins", data.get("wins", 0))
                losses = evaluation.get("losses", data.get("losses", 0))
                ties = evaluation.get("ties", data.get("ties", 0))
                
                if wins + losses > 0:
                    all_metrics["wins"].append(wins)
                    all_metrics["losses"].append(losses)
                    all_metrics["ties"].append(ties)
                
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                continue
    
    # Compute averages
    avg_metrics = {}
    for key in ["diversity", "coherence", "conflict_surface", "factual_grounding", "key_points"]:
        if all_metrics[key]:
            avg_metrics[key] = sum(all_metrics[key]) / len(all_metrics[key])
        else:
            avg_metrics[key] = 0.0
    
    if all_metrics["evidence_at_key"]:
        avg_metrics["evidence_at_key"] = sum(all_metrics["evidence_at_key"]) / len(all_metrics["evidence_at_key"])
    else:
        avg_metrics["evidence_at_key"] = 0.0
    
    # Compute win rate
    total_wins = sum(all_metrics["wins"])
    total_losses = sum(all_metrics["losses"])
    total_ties = sum(all_metrics["ties"])
    avg_metrics["win_rate"] = compute_win_rate(total_wins, total_losses, total_ties)
    avg_metrics["total_wins"] = total_wins
    avg_metrics["total_losses"] = total_losses
    
    avg_metrics["num_topics"] = len(all_metrics["diversity"])
    
    return avg_metrics

def generate_ablation_table(baseline_metrics: Dict[str, Any], improved_metrics: Dict[str, Any]) -> str:
    """Generate LaTeX table for ablation study."""
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{All GPT-4o} & \\textbf{Mixed Models} \\\\",
        "\\midrule",
    ]
    
    # Holistic metrics
    lines.append(f"Diversity & {baseline_metrics['diversity']:.1f} & \\textbf{{{improved_metrics['diversity']:.1f}}} \\\\")
    lines.append(f"Coherence & {baseline_metrics['coherence']:.1f} & \\textbf{{{improved_metrics['coherence']:.1f}}} \\\\")
    lines.append(f"Conflict Surface & {baseline_metrics['conflict_surface']:.1f} & \\textbf{{{improved_metrics['conflict_surface']:.1f}}} \\\\")
    lines.append(f"Factual Grounding & {baseline_metrics['factual_grounding']:.1f} & \\textbf{{{improved_metrics['factual_grounding']:.1f}}} \\\\")
    lines.append("\\midrule")
    
    # Key-point metrics
    lines.append(f"\\#Key Pts & {baseline_metrics['key_points']:.1f} & {improved_metrics['key_points']:.1f} \\\\")
    lines.append(f"Evidence@Key & {baseline_metrics['evidence_at_key']:.2f} & \\textbf{{{improved_metrics['evidence_at_key']:.2f}}} \\\\")
    lines.append(f"Win Rate & {baseline_metrics['win_rate']:.2f} & \\textbf{{{improved_metrics['win_rate']:.2f}}} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Ablation study: Comparison between all GPT-4o agents (Baseline) and mixed models (GPT-4o, Claude Sonnet 4, Gemini 2) with research agent removed. Results averaged over {improved_metrics['num_topics']} topics.}}",
        "\\label{tab:ablation}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    baseline_dir = Path("results(baselines)/experiment")
    improved_dir = Path("results/experiment")
    
    if not baseline_dir.exists():
        print(f"Baseline directory not found: {baseline_dir}")
        exit(1)
    
    if not improved_dir.exists():
        print(f"Improved directory not found: {improved_dir}")
        exit(1)
    
    print("Computing ablation study metrics...")
    print("\nProcessing baseline (all GPT-4o)...")
    baseline_metrics = process_results_for_ablation(baseline_dir)
    
    print("\nProcessing improved (mixed models, no research agent)...")
    improved_metrics = process_results_for_ablation(improved_dir)
    
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"\nBaseline (All GPT-4o, n={baseline_metrics['num_topics']}):")
    print(f"  Diversity: {baseline_metrics['diversity']:.2f}")
    print(f"  Coherence: {baseline_metrics['coherence']:.2f}")
    print(f"  Conflict Surface: {baseline_metrics['conflict_surface']:.2f}")
    print(f"  Factual Grounding: {baseline_metrics['factual_grounding']:.2f}")
    print(f"  #Key Pts: {baseline_metrics['key_points']:.2f}")
    print(f"  Evidence@Key: {baseline_metrics['evidence_at_key']:.2f}")
    print(f"  Win Rate: {baseline_metrics['win_rate']:.2f}")
    
    print(f"\nImproved (Mixed Models, n={improved_metrics['num_topics']}):")
    print(f"  Diversity: {improved_metrics['diversity']:.2f}")
    print(f"  Coherence: {improved_metrics['coherence']:.2f}")
    print(f"  Conflict Surface: {improved_metrics['conflict_surface']:.2f}")
    print(f"  Factual Grounding: {improved_metrics['factual_grounding']:.2f}")
    print(f"  #Key Pts: {improved_metrics['key_points']:.2f}")
    print(f"  Evidence@Key: {improved_metrics['evidence_at_key']:.2f}")
    print(f"  Win Rate: {improved_metrics['win_rate']:.2f}")
    
    print("\n" + "="*80)
    print("IMPROVEMENTS:")
    print("="*80)
    print(f"  Diversity: {improved_metrics['diversity'] - baseline_metrics['diversity']:+.2f}")
    print(f"  Coherence: {improved_metrics['coherence'] - baseline_metrics['coherence']:+.2f}")
    print(f"  Conflict Surface: {improved_metrics['conflict_surface'] - baseline_metrics['conflict_surface']:+.2f}")
    print(f"  Factual Grounding: {improved_metrics['factual_grounding'] - baseline_metrics['factual_grounding']:+.2f}")
    print(f"  Evidence@Key: {improved_metrics['evidence_at_key'] - baseline_metrics['evidence_at_key']:+.2f}")
    print(f"  Win Rate: {improved_metrics['win_rate'] - baseline_metrics['win_rate']:+.2f}")
    
    # Generate LaTeX table
    latex_table = generate_ablation_table(baseline_metrics, improved_metrics)
    print("\n" + "="*80)
    print("LATEX TABLE:")
    print("="*80)
    print(latex_table)
    
    # Save to file
    with open("results/ablation_table.tex", "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to results/ablation_table.tex")

