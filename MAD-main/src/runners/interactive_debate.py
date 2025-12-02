"""
Interactive multi-agent debate runner for free-form user topics.

Workflow:
  1. User provides a topic and max number of agents (viewpoints).
  2. A "research" step asks the LLM to generate up to N distinct viewpoints.
  3. Each viewpoint becomes an agent with its own persona.
  4. Agents take turns debating for a small number of rounds.
  5. A judge persona summarizes the debate at the end.

This runner does NOT use datasets; it is fully interactive.
It reuses the LLMFactory / models.yaml infrastructure from MAD-main.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.debate.models import LLMFactory


logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Basic logging to file + console for interactive runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"interactive_debate_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to: %s", log_file)
    return logger


def load_models_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_base_model(models_cfg: Dict[str, Any], pairing: str) -> Any:
    """
    Use the 'A' model from a pairing as the base conversational model
    for research, agents, and judge personas.
    """
    if "pairings" in models_cfg:
        pairing_cfg = models_cfg["pairings"].get(pairing)
    else:
        pairing_cfg = models_cfg.get(pairing)

    if not pairing_cfg or "A" not in pairing_cfg:
        raise ValueError(f"Cannot find pairing '{pairing}' with an 'A' model in models config.")

    logger.info("Using pairing '%s' A-model as base interactive model: %s", pairing, pairing_cfg["A"])
    return LLMFactory.make(**pairing_cfg["A"])


def invoke_raw(model: Any, system: str, user: str) -> str:
    """Thin wrapper to call the underlying chat model and return raw text."""
    try:
        resp = model.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        content = getattr(resp, "content", str(resp))
        return str(content or "").strip()
    except Exception as e:
        logger.error("Interactive model call failed: %s", e)
        return ""


def generate_viewpoints(model: Any, topic: str, max_agents: int) -> List[Dict[str, str]]:
    """
    Ask the model to propose up to max_agents distinct viewpoints on a topic.
    Returns a list of dicts: [{"name": ..., "position": ..., "summary": ...}, ...]
    """
    system_prompt = (
        "You are a research assistant that enumerates distinct viewpoints on a topic.\n"
        "You MUST output STRICT JSON: a list of objects with fields "
        "\"name\", \"position\", and \"summary\".\n"
        "Do not add any extra keys or prose outside JSON."
    )

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"Generate between 2 and {max_agents} DISTINCT viewpoints that could appear in a debate.\n"
        'Each item should look like: {"name": "...", "position": "...", "summary": "..."}.\n'
        "The 'name' should be a short label (e.g., 'Economic Optimist', 'Cautious Ethicist').\n"
        "The 'position' should be a concise stance sentence.\n"
        "The 'summary' should briefly explain the reasoning in 1–3 sentences.\n"
    )

    raw = invoke_raw(model, system_prompt, user_prompt)
    logger.info("Raw viewpoints JSON: %s", raw)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            result = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                pos = str(item.get("position", "")).strip()
                summ = str(item.get("summary", "")).strip()
                if name and pos:
                    result.append({"name": name, "position": pos, "summary": summ})
            if len(result) >= 2:
                return result[:max_agents]
    except Exception as e:
        logger.warning("Failed to parse viewpoints JSON: %s", e)

    # Fallback: fabricate two generic viewpoints
    logger.warning("Falling back to generic 2-agent viewpoints.")
    return [
        {
            "name": "Pro Position",
            "position": f"Strongly supports the statement: {topic}",
            "summary": "Argues in favor, emphasizing benefits and positive outcomes.",
        },
        {
            "name": "Con Position",
            "position": f"Strongly opposes the statement: {topic}",
            "summary": "Argues against, emphasizing risks and drawbacks.",
        },
    ]


def run_interactive_debate(
    topic: str,
    max_agents: int,
    num_rounds: int,
    models_cfg_path: str,
    pairing: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Main interactive pipeline:
      - Build base model
      - Generate viewpoints -> agents
      - Multi-round debate
      - Judge summary
      - Save to JSON
    """
    # Ensure output directory exists when called programmatically (e.g., from FastAPI)
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_cfg = load_models_config(models_cfg_path)
    base_model = make_base_model(models_cfg, pairing)

    logger.info("Generating viewpoints for topic: %s", topic)
    viewpoints = generate_viewpoints(base_model, topic, max_agents)
    logger.info("Got %d viewpoints", len(viewpoints))

    # Build agent personas
    agents: List[Dict[str, Any]] = []
    for vp in viewpoints:
        name = vp["name"]
        system_prompt = (
            f"You are an agent named '{name}'.\n"
            f"Your core position on the topic is: {vp['position']}.\n"
            "In the debate, you MUST consistently argue from this perspective.\n"
            "Your primary goal is to REBUT other agents, not to agree with them.\n"
            "Actively look for conflicts and contradictions in their statements and exploit them.\n"
            "When possible, quote 1–2 short phrases from others and respond with patterns like "
            "\"you said X, but you ignore Y\" or \"you claim X, however Y shows the opposite\".\n"
            "Be concise, sharp, and argumentative, while remaining professional.\n"
        )
        agents.append(
            {
                "name": name,
                "position": vp["position"],
                "summary": vp["summary"],
                "system_prompt": system_prompt,
                "messages": [],
            }
        )

    conversation_log: List[Dict[str, Any]] = []

    # Round 0: record research / viewpoint generation as setup in the transcript
    setup_lines = []
    for idx, vp in enumerate(viewpoints, start=1):
        setup_lines.append(f"Agent {idx} ({vp['name']}): {vp['position']}")
    setup_text = "Initial viewpoints (Round 0 - research/setup):\n" + "\n".join(setup_lines)
    conversation_log.append(
        {
            "speaker": "Research / Setup",
            "content": setup_text,
            "round": 0,
        }
    )

    def format_history(limit_chars: int = 3000) -> str:
        """Serialize recent conversation history to a single string."""
        text = ""
        for msg in conversation_log:
            text += f"{msg['speaker']}: {msg['content']}\n"
        if len(text) > limit_chars:
            return text[-limit_chars:]
        return text

    # Debate rounds
    for r in range(1, num_rounds + 1):
        logger.info("=== Debate Round %d ===", r)
        for agent in agents:
            name = agent["name"]
            sys = agent["system_prompt"]
            history_text = format_history()
            user_prompt = (
                f"Topic: {topic}\n\n"
                f"Round: {r}\n"
                "You are taking a turn in a multi-agent debate.\n"
                "Previous conversation (if any):\n"
                f"{history_text}\n\n"
                "Instructions for this turn:\n"
                "- Your main job is to ATTACK and REBUT other agents' arguments.\n"
                "- Pick 1–2 specific sentences or claims from recent messages above and quote them explicitly.\n"
                "- Use phrasing like \"you said X, but you ignore Y\" or \"you claim X, however Y shows the opposite\".\n"
                "- Make the conflict clear and focused, not vague; address concrete points.\n"
                "- Also briefly restate and reinforce your own position.\n"
                "Now write your next contribution (1–3 short paragraphs).\n"
            )

            logger.info("Agent '%s' taking a turn for round %d.", name, r)
            reply = invoke_raw(base_model, sys, user_prompt)
            if not reply:
                reply = "[No response due to API error; treating as empty turn.]"
            conversation_log.append({"speaker": name, "content": reply, "round": r})
            agent["messages"].append({"round": r, "content": reply})

    # Judge summary
    judge_system = (
        "You are a neutral moderator and judge.\n"
        "You will receive a debate transcript and should:\n"
        "1) Summarize the main viewpoints.\n"
        "2) Highlight key agreements and disagreements.\n"
        "3) Provide a balanced assessment and (optionally) a tentative conclusion.\n"
    )
    history_text = ""
    for msg in conversation_log:
        history_text += f"{msg['speaker']}: {msg['content']}\n"

    judge_user = (
        f"Topic: {topic}\n\n"
        "Here is the full multi-agent debate transcript:\n"
        f"{history_text}\n\n"
        "Now provide your summary and assessment in 2–5 paragraphs."
    )

    logger.info("Calling judge to summarize debate.")
    judge_summary = invoke_raw(base_model, judge_system, judge_user)

    # Final academic-style report for instructor / professor
    report_system = (
        "You are an expert research writer preparing a high-quality report for a professor.\n"
        "Your task is to synthesize a multi-agent debate on a given topic into a clear, well-structured report.\n"
        "Write in a formal, objective, and academically oriented style (but not like a paper submission).\n"
    )

    viewpoints_text_lines = []
    for idx, vp in enumerate(viewpoints, start=1):
        viewpoints_text_lines.append(
            f"Agent {idx} ({vp['name']}): position={vp['position']}; summary={vp['summary']}"
        )
    viewpoints_text = "\n".join(viewpoints_text_lines)

    full_history = ""
    for msg in conversation_log:
        round_tag = msg.get("round", None)
        if round_tag is not None:
            full_history += f"[Round {round_tag}] {msg['speaker']}: {msg['content']}\n"
        else:
            full_history += f"{msg['speaker']}: {msg['content']}\n"

    report_user = (
        f"Topic: {topic}\n\n"
        "Below are the main viewpoints (agents) and the debate transcript.\n\n"
        "Viewpoints:\n"
        f"{viewpoints_text}\n\n"
        "Debate transcript:\n"
        f"{full_history}\n\n"
        "Judge summary of the debate:\n"
        f"{judge_summary}\n\n"
        "Now write a final report with the following sections:\n"
        "1. Research Question & Context (1 short paragraph)\n"
        "2. Summary of Viewpoints (bullet points or short paragraphs)\n"
        "3. Comparative Analysis & Key Conflicts (focus on where agents explicitly disagree, with examples)\n"
        "4. Tentative Conclusion & Recommendation (what an informed decision-maker should tentatively believe or do)\n"
        "5. Limitations & Suggestions for Further Investigation (1–2 short paragraphs).\n"
        "The report should be self-contained and should not mention being an AI model or referencing 'the debate' mechanics.\n"
    )

    logger.info("Calling model to generate final report.")
    final_report = invoke_raw(base_model, report_system, report_user)

    # Fallback: if the long-context report fails (empty string), try a shorter prompt
    if not final_report:
        logger.warning("Final report generation returned empty text; retrying with shortened prompt.")
        short_history = full_history[-4000:]  # keep only the tail of the transcript
        short_report_user = (
            f"Topic: {topic}\n\n"
            "Here is a shortened version of the debate transcript and the main viewpoints.\n\n"
            "Viewpoints:\n"
            f"{viewpoints_text}\n\n"
            "Shortened debate transcript (most recent turns first):\n"
            f"{short_history}\n\n"
            "Now write a concise but high-quality report with the structure:\n"
            "1. Context\n"
            "2. Main viewpoints\n"
            "3. Key conflicts\n"
            "4. Tentative conclusion & recommendation\n"
            "5. Limitations / open questions.\n"
        )
        final_report = invoke_raw(base_model, report_system, short_report_user)

    # Build result payload
    result = {
        "topic": topic,
        "viewpoints": viewpoints,
        "agents": [
            {
                "name": a["name"],
                "position": a["position"],
                "summary": a["summary"],
                "messages": a["messages"],
            }
            for a in agents
        ],
        "conversation_log": conversation_log,
        "judge_summary": judge_summary,
        "final_report": final_report,
        "timestamp": time.time(),
    }

    # Save to disk
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outfile = output_dir / f"interactive_debate_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Interactive debate saved to: %s", outfile)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run an interactive multi-agent debate on a user topic.")
    parser.add_argument("--topic", type=str, required=True, help="Debate topic provided by the user.")
    parser.add_argument("--max_agents", type=int, default=5, help="Maximum number of agents/viewpoints.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds.")
    parser.add_argument(
        "--models",
        type=str,
        default="configs/models.yaml",
        help="Path to models config YAML (reuses MAD-main models.yaml).",
    )
    parser.add_argument(
        "--pairing",
        type=str,
        default="qwen_qwen",
        help="Which pairing key from models.yaml to use as base model (uses its A-model).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/interactive",
        help="Directory to store interactive debate logs and JSON.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)

    if args.max_agents < 2:
        logger.warning("max_agents < 2 is not meaningful; forcing to 2.")
        args.max_agents = 2

    logger.info("Starting interactive debate.")
    logger.info("Topic: %s", args.topic)
    logger.info("Max agents: %d, Rounds: %d", args.max_agents, args.rounds)

    run_interactive_debate(
        topic=args.topic,
        max_agents=args.max_agents,
        num_rounds=args.rounds,
        models_cfg_path=args.models,
        pairing=args.pairing,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()


