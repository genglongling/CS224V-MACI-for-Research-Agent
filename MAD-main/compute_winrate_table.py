#!/usr/bin/env python3
"""
Compute win rate table metrics from all experiment results.
Extracts: #Key Pts, Evidence@Key, and Win Rate for both Baseline and CollectiveMind.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

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

def process_all_results(results_dir: Path, baseline_dir: Path = None, costorm_dir: Path = None) -> Dict[str, Any]:
    """Process all result files and compute aggregated metrics."""
    baseline_kps = []
    cm_kps = []
    costorm_kps = []
    baseline_evidence = []
    cm_evidence = []
    costorm_evidence = []
    wins_list = []
    losses_list = []
    ties_list = []
    
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
                
                # Extract baseline key points
                baseline_kp_list = data.get("baseline", {}).get("key_points", [])
                baseline_total, baseline_with_evidence = count_key_points_with_evidence(baseline_kp_list)
                baseline_kps.append(baseline_total)
                if baseline_total > 0:
                    baseline_evidence.append(baseline_with_evidence / baseline_total)
                
                # Extract CollectiveMind key points
                cm_kp_list = data.get("collectivemind", {}).get("key_points", [])
                cm_total, cm_with_evidence = count_key_points_with_evidence(cm_kp_list)
                cm_kps.append(cm_total)
                if cm_total > 0:
                    cm_evidence.append(cm_with_evidence / cm_total)
                
                # Extract win rate data (may be in evaluation key or top level)
                evaluation = data.get("evaluation", {})
                wins = evaluation.get("wins", data.get("wins", 0))
                losses = evaluation.get("losses", data.get("losses", 0))
                ties = evaluation.get("ties", data.get("ties", 0))
                
                if wins + losses > 0:  # Only count if there were actual comparisons
                    wins_list.append(wins)
                    losses_list.append(losses)
                    ties_list.append(ties)
                
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                continue
    
    # Process CO-STORM results if directory exists
    if costorm_dir and costorm_dir.exists():
        # Try to read from experiment_final.json first
        costorm_final = costorm_dir.parent / "experiment_final.json"
        if costorm_final.exists():
            try:
                with open(costorm_final, 'r') as f:
                    costorm_data = json.load(f)
                    stats = costorm_data.get("stats", {})
                    # Extract aggregated stats
                    costorm_baseline_kps = stats.get("baseline_kps", 0)
                    costorm_cm_kps = stats.get("collective_kps", 0)
                    costorm_baseline_evidence = stats.get("baseline_evidence_count", 0)
                    costorm_cm_evidence = stats.get("collective_evidence_count", 0)
                    
                    # Calculate averages (assuming 20 topics)
                    num_topics = stats.get("topics", 20)
                    if num_topics > 0:
                        costorm_kps.append(costorm_cm_kps / num_topics)
                        if costorm_cm_kps > 0:
                            costorm_evidence.append(costorm_cm_evidence / costorm_cm_kps)
            except Exception as e:
                print(f"Error processing CO-STORM final file: {e}")
    
    # Compute averages
    avg_baseline_kps = sum(baseline_kps) / len(baseline_kps) if baseline_kps else 0.0
    avg_cm_kps = sum(cm_kps) / len(cm_kps) if cm_kps else 0.0
    avg_costorm_kps = sum(costorm_kps) / len(costorm_kps) if costorm_kps else 0.0
    avg_baseline_evidence = sum(baseline_evidence) / len(baseline_evidence) if baseline_evidence else 0.0
    avg_cm_evidence = sum(cm_evidence) / len(cm_evidence) if cm_evidence else 0.0
    avg_costorm_evidence = sum(costorm_evidence) / len(costorm_evidence) if costorm_evidence else 0.0
    
    # Compute overall win rate
    total_wins = sum(wins_list)
    total_losses = sum(losses_list)
    total_ties = sum(ties_list)
    win_rate = compute_win_rate(total_wins, total_losses, total_ties)
    
    return {
        "baseline": {
            "key_points": avg_baseline_kps,
            "evidence_at_key": avg_baseline_evidence
        },
        "collectivemind": {
            "key_points": avg_cm_kps,
            "evidence_at_key": avg_cm_evidence
        },
        "costorm": {
            "key_points": avg_costorm_kps,
            "evidence_at_key": avg_costorm_evidence
        },
        "win_rate": win_rate,
        "stats": {
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_ties": total_ties,
            "num_topics": len(baseline_kps)
        }
    }

def generate_latex_table(metrics: Dict[str, Any]) -> str:
    """Generate LaTeX table from metrics."""
    baseline = metrics["baseline"]
    cm = metrics["collectivemind"]
    costorm = metrics.get("costorm", {})
    win_rate = metrics["win_rate"]
    
    # Determine best values
    kp_best = "cm" if cm['key_points'] >= max(baseline['key_points'], costorm.get('key_points', 0) if costorm else 0) else ("costorm" if costorm and costorm.get('key_points', 0) >= baseline['key_points'] else "baseline")
    evidence_best = "cm" if cm['evidence_at_key'] >= max(baseline['evidence_at_key'], costorm.get('evidence_at_key', 0) if costorm else 0) else ("costorm" if costorm and costorm.get('evidence_at_key', 0) >= baseline['evidence_at_key'] else "baseline")
    
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{System} & \\textbf{\\#Key Pts} & \\textbf{Evidence@Key} & \\textbf{Win Rate} \\\\",
        "\\midrule",
    ]
    
    # Baseline row
    baseline_line = f"Baseline & {baseline['key_points']:.1f} & {baseline['evidence_at_key']:.2f} & {1 - win_rate:.2f} \\\\"
    lines.append(baseline_line)
    
    # CO-STORM row
    if costorm:
        costorm_line = f"CO-STORM & {costorm['key_points']:.1f} & {costorm['evidence_at_key']:.2f} & -- \\\\"
        lines.append(costorm_line)
    
    # CollectiveMind row
    cm_kp_str = f"\\textbf{{{cm['key_points']:.1f}}}" if kp_best == "cm" else f"{cm['key_points']:.1f}"
    cm_ev_str = f"\\textbf{{{cm['evidence_at_key']:.2f}}}" if evidence_best == "cm" else f"{cm['evidence_at_key']:.2f}"
    cm_line = f"CollectiveMind & {cm_kp_str} & {cm_ev_str} & \\textbf{{{win_rate:.2f}}} \\\\"
    lines.append(cm_line)
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Key-point extraction analysis averaged over {metrics['stats']['num_topics']} topics. \\#Key Pts: average number of distinct key points per topic; Evidence@Key: proportion of key points grounded in explicit evidence spans. Win Rate is the pairwise win rate of CollectiveMind over the Baseline.}}",
        "\\label{tab:keypoints}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    results_dir = Path("results/experiment")
    baseline_dir = Path("results(baselines)/experiment") if Path("results(baselines)/experiment").exists() else None
    costorm_dir = Path("results(co-storm)/experiment") if Path("results(co-storm)/experiment").exists() else None
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        exit(1)
    
    print("Computing win rate table metrics...")
    metrics = process_all_results(results_dir, baseline_dir, costorm_dir)
    
    print("\n" + "="*80)
    print("AGGREGATED METRICS")
    print("="*80)
    print(f"\nBaseline:")
    print(f"  #Key Pts: {metrics['baseline']['key_points']:.2f}")
    print(f"  Evidence@Key: {metrics['baseline']['evidence_at_key']:.2f}")
    
    if metrics.get("costorm"):
        print(f"\nCO-STORM:")
        print(f"  #Key Pts: {metrics['costorm']['key_points']:.2f}")
        print(f"  Evidence@Key: {metrics['costorm']['evidence_at_key']:.2f}")
    
    print(f"\nCollectiveMind:")
    print(f"  #Key Pts: {metrics['collectivemind']['key_points']:.2f}")
    print(f"  Evidence@Key: {metrics['collectivemind']['evidence_at_key']:.2f}")
    
    print(f"\nWin Rate (CollectiveMind over Baseline): {metrics['win_rate']:.2f}")
    print(f"  Total wins: {metrics['stats']['total_wins']}")
    print(f"  Total losses: {metrics['stats']['total_losses']}")
    print(f"  Total ties: {metrics['stats']['total_ties']}")
    print(f"  Number of topics: {metrics['stats']['num_topics']}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(metrics)
    print("\n" + "="*80)
    print("LATEX TABLE:")
    print("="*80)
    print(latex_table)
    
    # Save to file
    with open("results/winrate_table.tex", "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to results/winrate_table.tex")

