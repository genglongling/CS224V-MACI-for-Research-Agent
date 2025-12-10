import json
import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

# We import these but might mock them
from src.runners.interactive_debate import generate_viewpoints, generate_agent_brief, run_interactive_debate
from src.debate.models import LLMFactory

# --- Prompts ---

BASELINE_SYNTHESIS_PROMPT = """
You are a professor summarizing a set of research dossiers on a complex topic.
You have been given {num_agents} deep research briefs from different perspectives.

Topic: {topic}

Your goal is to write a comprehensive "Final Report" synthesizing these viewpoints.
Do NOT conduct new research. Only use the information provided in the briefs.

Structure the report as follows:
1. Research Question & Context
2. Summary of Viewpoints (Briefly describe each perspective)
3. Key Conflicts & Comparative Analysis (Where do they disagree and why?)
4. Tentative Recommendations
5. Limitations

Write in a formal, academic tone. Length: 800-1200 words.
"""

KEY_POINT_EXTRACTION_PROMPT = """
You are an expert evaluator. Read the following research report and extract a set of "Key Points".
A Key Point is an atomic argument or claim (e.g., "Regulated stablecoins improve cross-border settlement for BRI trade routes").

Report:
{report}

For each key point, provide:
1. "point": The statement of the key point.
2. "label": A short label (3-5 words).
3. "evidence": A verbatim quote (span) from the report that supports this point. If no specific evidence is cited, leave null.

Output strictly in JSON format:
{{
  "key_points": [
    {{
      "point": "...",
      "label": "...",
      "evidence": "..." or null
    }},
    ...
  ]
}}
"""

PAIRWISE_JUDGE_PROMPT = """
You are a judge comparing how well two research systems supported a specific Key Point.

Key Point: {point}

System A Evidence:
{evidence_a}

System B Evidence:
{evidence_b}

Task:
Which system provides better, more specific, and more logical support for this key point?
If one system provides a specific citation/example and the other is generic, prefer the specific one.
If both are similar, declare a tie.

Output strictly JSON:
{{
  "winner": "A" or "B" or "Tie",
  "reason": "..."
}}
"""

# --- Mock Model ---
class MockModel:
    def __init__(self, provider="mock", model="mock", temperature=0, max_tokens=0):
        self.provider = provider
        self.model = model
    
    def invoke(self, messages):
        content = ""
        # Detect prompt type based on content
        last_msg = messages[-1]["content"] if messages else ""
        
        if "Extract key points" in last_msg:
            content = json.dumps({
                "key_points": [
                    {"point": "AI regulation improves safety", "label": "Safety Benefit", "evidence": "According to report..."},
                    {"point": "AI regulation stifles innovation", "label": "Innovation Risk", "evidence": "As noted in section 2..."}
                ]
            })
        elif "Which system provides better" in last_msg:
            content = json.dumps({"winner": "B", "reason": "System B provided concrete evidence."})
        elif "Final Report" in last_msg or "Deep Preparation" in last_msg:
            content = f"Final Report on Topic.\n\n1. Context...\n2. Viewpoints...\n3. Conflicts...\n4. Recs...\n5. Limits..."
        elif "viewpoints" in last_msg or "perspective" in last_msg:
             content = json.dumps([
                 {"name": "Reformer", "position": "Pro Reform", "summary": "Supports change."},
                 {"name": "Traditionalist", "position": "Anti Reform", "summary": "Opposes change."},
                 {"name": "Moderate", "position": "Neutral", "summary": "Seeks balance."}
             ])
        elif "brief" in last_msg:
             content = json.dumps({
                 "name": "Agent", "position": "Pos", "summary_for_prompt": "Summary of position.",
                 "supporting_arguments": [], "anticipated_opponent_arguments": [], "self_weaknesses": [], "questions_to_ask": [], "debate_strategy": {}
             })
        elif "Does the following report support" in last_msg:
             content = "Yes, it says: 'Evidence found here.'"
        else:
             content = "Mock response content."
             
        return type('Response', (), {'content': content})

# --- Helpers ---

def load_models_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_baseline(topic: str, models_cfg: Dict, output_dir: Path, question_type_id: str = "1", mock=False):
    """
    Runs the Baseline: Generate Viewpoints -> Generate Briefs -> Synthesize Report (No Debate)
    """
    print(f"  [Baseline] Starting for: {topic}")
    
    # 1. Setup Models
    if mock:
        researcher = MockModel()
        judge = MockModel()
    else:
        pairing_cfg = models_cfg["pairings"]["experiment"]
        # Use model A for researcher (baseline) instead of separate researcher
        researcher = LLMFactory.make(**pairing_cfg["A"])
        judge = LLMFactory.make(**pairing_cfg["judge"])

    # 2. Viewpoint Discovery
    viewpoints = generate_viewpoints(researcher, topic, max_agents=3)
    
    # 3. Deep Preparation (Briefs)
    briefs = []
    briefs_dir = output_dir / "baseline_briefs" / question_type_id
    briefs_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, vp in enumerate(viewpoints, start=1):
        brief = generate_agent_brief(researcher, topic, vp, idx, briefs_dir)
        briefs.append(brief)

    # 4. Synthesis
    # Construct input context from briefs
    briefs_text = ""
    for b in briefs:
        briefs_text += f"--- Viewpoint: {b['name']} ({b['position']}) ---\n"
        briefs_text += json.dumps(b, indent=2) + "\n\n"
        
    prompt = BASELINE_SYNTHESIS_PROMPT.format(num_agents=len(briefs), topic=topic)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Here are the research briefs:\n{briefs_text}"}
    ]
    
    response = judge.invoke(messages)
    report = response.content
    
    return report

def extract_key_points_from_report(report: str, model) -> List[Dict]:
    messages = [
        {"role": "system", "content": KEY_POINT_EXTRACTION_PROMPT.format(report=report)},
        {"role": "user", "content": "Extract key points now."}
    ]
    resp = model.invoke(messages)
    content = resp.content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    elif content.startswith("```"):
        content = content.replace("```", "")
        
    try:
        data = json.loads(content)
        return data.get("key_points", [])
    except Exception as e:
        print(f"Error parsing key points: {e}")
        return []

def run_pairwise_comparison(kp_text: str, ev_a: str, ev_b: str, model) -> str:
    if not ev_a and not ev_b:
        return "Tie"
    
    prompt = PAIRWISE_JUDGE_PROMPT.format(
        point=kp_text,
        evidence_a=ev_a if ev_a else "(No explicit evidence found)",
        evidence_b=ev_b if ev_b else "(No explicit evidence found)"
    )
    
    messages = [{"role": "user", "content": prompt}]
    resp = model.invoke(messages)
    content = resp.content.strip()
    
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    elif content.startswith("```"):
        content = content.replace("```", "")
        
    try:
        data = json.loads(content)
        return data.get("winner", "Tie")
    except:
        return "Tie"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock models for testing")
    parser.add_argument("--topics", type=int, default=1, help="Number of topics to run")
    parser.add_argument("--category", type=str, default="finance", help="Category to run: finance, ai_governance, or social_policy")
    parser.add_argument("--start-from", type=int, default=1, help="Start from topic number (1-indexed)")
    args = parser.parse_args()

    # Load Topics
    with open("topics.json", "r") as f:
        topics_data = json.load(f)
    
    # Handle new structure with categories
    if isinstance(topics_data, dict):
        if args.category not in topics_data:
            print(f"Error: Category '{args.category}' not found. Available: {list(topics_data.keys())}")
            return
        category_topics = topics_data[args.category]
        # Convert to list of tuples: (question_type_id, topic)
        topics = [(qid, topic) for qid, topic in category_topics.items()]
    else:
        # Old structure: list of topics
        topics = [(str(i+1), topic) for i, topic in enumerate(topics_data)]
    
    # Filter by start_from and limit number of topics
    start_idx = args.start_from - 1  # Convert to 0-indexed
    if start_idx < 0:
        start_idx = 0
    if start_idx >= len(topics):
        print(f"Error: start-from ({args.start_from}) is greater than number of topics ({len(topics)})")
        return
    topics = topics[start_idx:]
    topics = topics[:args.topics]
    
    models_cfg_path = "configs/experiment_models.yaml"
    models_cfg = load_models_config(models_cfg_path)
    
    if args.mock:
        print(">>> RUNNING IN MOCK MODE <<<")
        # Monkey patch LLMFactory to return MockModel
        LLMFactory.make = lambda **kwargs: MockModel(**kwargs)
        evaluator_model = MockModel()
    else:
        pairing_cfg = models_cfg["pairings"]["experiment"]
        evaluator_model = LLMFactory.make(**pairing_cfg["judge"])

    results_dir = Path("results/experiment") / args.category
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to both file and console
    log_file = results_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting experiment for category: {args.category}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Processing {len(topics)} topics")
    
    # Also redirect print statements
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_file_handle = open(log_file, 'a', encoding='utf-8')
    sys.stdout = TeeOutput(sys.stdout, log_file_handle)
    sys.stderr = TeeOutput(sys.stderr, log_file_handle)
    
    aggregated_stats = {
        "baseline_kps": 0,
        "collective_kps": 0,
        "baseline_evidence_count": 0,
        "collective_evidence_count": 0,
        "cm_wins": 0,
        "cm_losses": 0, 
        "ties": 0,
        "topics": 0
    }
    
    topic_results = []

    for question_type_id, topic in topics:
        print(f"\nProcessing {args.category} - Question Type {question_type_id}: {topic[:80]}...")
        
        # --- Run Baseline ---
        baseline_report = run_baseline(topic, models_cfg, results_dir, question_type_id=question_type_id, mock=args.mock)
        
        # --- Run CollectiveMind ---
        print(f"  [CollectiveMind] Starting...")
        cm_result = run_interactive_debate(
            topic=topic,
            max_agents=3,
            num_rounds=3,
            models_cfg_path=models_cfg_path,
            pairing="experiment",
            output_dir=results_dir,
            question_type_id=question_type_id
        )
        cm_report = cm_result["final_report"]
        
        # --- Evaluation ---
        print(f"  [Eval] Extracting Key Points...")
        
        kps_baseline = extract_key_points_from_report(baseline_report, evaluator_model)
        kps_cm = extract_key_points_from_report(cm_report, evaluator_model)
        
        # Update Stats
        aggregated_stats["baseline_kps"] += len(kps_baseline)
        aggregated_stats["collective_kps"] += len(kps_cm)
        aggregated_stats["baseline_evidence_count"] += sum(1 for k in kps_baseline if k.get("evidence"))
        aggregated_stats["collective_evidence_count"] += sum(1 for k in kps_cm if k.get("evidence"))
        aggregated_stats["topics"] += 1
        
        all_kps = []
        for k in kps_baseline:
            k['origin'] = 'baseline'
            all_kps.append(k)
        for k in kps_cm:
            k['origin'] = 'cm'
            all_kps.append(k)
            
        print(f"  [Eval] Pairwise Comparison ({len(all_kps)} points)...")
        
        wins = 0
        losses = 0
        ties = 0
        
        pairwise_comparisons = []
        for kp in all_kps:
            point = kp['point']
            ev_baseline = kp['evidence'] if kp['origin'] == 'baseline' else None
            ev_cm = kp['evidence'] if kp['origin'] == 'cm' else None
            
            # Helper to search evidence
            def find_evidence(text, claim, model):
                msg = f"Does the following report support the claim: '{claim}'? If yes, extract a short verbatim quote. If no, return 'None'.\n\nReport: {text[:4000]}..." 
                resp = model.invoke([{"role": "user", "content": msg}])
                if "None" in resp.content or "does not support" in resp.content:
                    return None
                return resp.content
            
            if not ev_baseline:
                ev_baseline = find_evidence(baseline_report, point, evaluator_model)
            if not ev_cm:
                ev_cm = find_evidence(cm_report, point, evaluator_model)
                
            winner = run_pairwise_comparison(point, ev_baseline, ev_cm, evaluator_model)
            
            comparison_result = {
                "key_point": point,
                "baseline_evidence": ev_baseline,
                "cm_evidence": ev_cm,
                "winner": winner,
                "reason": winner  # Could be enhanced to include reason from judge
            }
            pairwise_comparisons.append(comparison_result)
            
            if winner == 'B': # B is CM
                wins += 1
            elif winner == 'A': # A is Baseline
                losses += 1
            else:
                ties += 1
        
        aggregated_stats["cm_wins"] += wins
        aggregated_stats["cm_losses"] += losses
        aggregated_stats["ties"] += ties
        
        # Save individual example JSON file
        import re
        topic_slug = re.sub(r'[^a-z0-9]+', '_', topic.lower())[:50]
        output_file = results_dir / f"{question_type_id}_{topic_slug}.json"
        
        example_output = {
            "category": args.category,
            "question_type_id": int(question_type_id),
            "topic": topic,
            "baseline": {
                "report": baseline_report,
                "key_points": kps_baseline
            },
            "collectivemind": {
                "viewpoints": cm_result.get("viewpoints", []),
                "agent_briefs": cm_result.get("agent_briefs", []),
                "agents": cm_result.get("agents", []),
                "conversation_log": cm_result.get("conversation_log", []),
                "judge_summary": cm_result.get("judge_summary", ""),
                "final_report": cm_report,
                "key_points": kps_cm
            },
            "evaluation": {
                "baseline_kps": len(kps_baseline),
                "cm_kps": len(kps_cm),
                "baseline_evidence_count": sum(1 for k in kps_baseline if k.get("evidence")),
                "cm_evidence_count": sum(1 for k in kps_cm if k.get("evidence")),
                "pairwise_comparisons": pairwise_comparisons,
                "wins": wins,
                "losses": losses,
                "ties": ties
            }
        }
        
        with open(output_file, "w") as f:
            json.dump(example_output, f, indent=2)
        print(f"  -> Saved to: {output_file}")
        
        topic_res = {
            "category": args.category,
            "question_type_id": int(question_type_id),
            "topic": topic,
            "baseline_kps": len(kps_baseline),
            "cm_kps": len(kps_cm),
            "wins": wins,
            "losses": losses,
            "ties": ties
        }
        topic_results.append(topic_res)
        print(f"  -> Result: {topic_res}")

    # Final Stats
    total_decisions = aggregated_stats["cm_wins"] + aggregated_stats["cm_losses"]
    win_rate = aggregated_stats["cm_wins"] / total_decisions if total_decisions > 0 else 0
    
    avg_kps_base = aggregated_stats["baseline_kps"] / aggregated_stats["topics"] if aggregated_stats["topics"] > 0 else 0
    avg_kps_cm = aggregated_stats["collective_kps"] / aggregated_stats["topics"] if aggregated_stats["topics"] > 0 else 0
    
    ev_rate_base = aggregated_stats["baseline_evidence_count"] / aggregated_stats["baseline_kps"] if aggregated_stats["baseline_kps"] > 0 else 0
    ev_rate_cm = aggregated_stats["collective_evidence_count"] / aggregated_stats["collective_kps"] if aggregated_stats["collective_kps"] > 0 else 0

    print("\n\n=== FINAL RESULTS (Table 2) ===")
    print(f"Topics: {aggregated_stats['topics']}")
    print(f"Avg Key Points (Baseline): {avg_kps_base:.1f}")
    print(f"Avg Key Points (CollectiveMind): {avg_kps_cm:.1f}")
    print(f"Evidence@Key (Baseline): {ev_rate_base:.2f}")
    print(f"Evidence@Key (CollectiveMind): {ev_rate_cm:.2f}")
    print(f"Win Rate (CollectiveMind): {win_rate:.2f}")
    
    with open("results/experiment_final.json", "w") as f:
        json.dump({
            "stats": aggregated_stats,
            "topics": topic_results
        }, f, indent=2)
    
    print(f"\nFull log saved to: {log_file}")
    
    # Restore stdout/stderr and close log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file_handle.close()

if __name__ == "__main__":
    main()
