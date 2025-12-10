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


def _extract_json_object(text: str) -> str:
    """
    Try to extract a JSON object string from an LLM response.
    Handles cases like ```json ...``` or leading/trailing explanations.
    """
    if not text:
        return ""
    s = text.strip()
    # Strip Markdown fences
    if s.startswith("```"):
        s = s.lstrip("`")
        # remove possible 'json' or language token
        if s.lower().startswith("json"):
            s = s[4:]
        # remove trailing fences
        if "```" in s:
            s = s.split("```", 1)[0]
    # Find first '{' and last '}'
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


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
    
    print(f"\n{'='*80}")
    print(f"VIEWPOINTS RAW OUTPUT:")
    print(f"{'='*80}")
    print(raw)
    print(f"{'='*80}\n")
    
    logger.info("Raw viewpoints output (full): %s", raw)

    # Try to parse, but don't fail if it doesn't work
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
                logger.info("Successfully parsed viewpoints JSON")
                return result[:max_agents]
    except Exception as e:
        logger.info("JSON parsing skipped for viewpoints (using fallback): %s", str(e))

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


def generate_agent_brief(
    model: Any,
    topic: str,
    viewpoint: Dict[str, str],
    agent_index: int,
    briefs_dir: Path,
) -> Dict[str, Any]:
    """
    Generate a deep research 'preparation file' for a single agent.
    The brief captures supporting arguments, anticipated attacks, self-weaknesses,
    questions to ask, and an internal strategy summary.
    """
    name = viewpoint.get("name", f"Agent {agent_index}")
    position = viewpoint.get("position", "")
    vp_summary = viewpoint.get("summary", "")

    system_prompt = (
        "You are a senior research strategist preparing an in-depth pre-debate brief for an agent.\n"
        "Think as if you had several hours to do deep research (using your world knowledge, reports, history, economics, etc.).\n"
        "The brief should be comprehensive and long (roughly 1,500–2,500 words), not a shallow outline.\n"
        "You MUST output STRICT JSON with the following top-level keys:\n"
        "  - agent_id (int)\n"
        "  - name (str)\n"
        "  - topic (str)\n"
        "  - position (str)\n"
        "  - role_summary (str)\n"
        "  - supporting_arguments (list of objects)\n"
        "  - anticipated_opponent_arguments (list of objects)\n"
        "  - self_weaknesses (list of objects)\n"
        "  - questions_to_ask (list of strings)\n"
        "  - debate_strategy (object)\n"
        "  - summary_for_prompt (str)\n"
        "Do NOT include any explanation outside the JSON object."
    )

    user_prompt = (
        f"Topic: {topic}\n\n"
        f"This agent is called: {name}\n"
        f"Core position: {position}\n"
        f"Short summary of this stance: {vp_summary}\n\n"
        "Write a high-quality, research-style preparation brief that:\n"
        "- Gathers AT LEAST 5 concrete supporting arguments. For each argument include:\n"
        "    * 'claim': a clear statement of the point being defended.\n"
        "    * 'logic': 3–5 sentences explaining why this claim holds (causal chain, mechanisms, trade-offs).\n"
        "    * 'evidence': 2–4 sentences citing empirical facts, historical cases, expert reports, or plausible numeric examples.\n"
        "    * 'risks_or_limits': 2–3 sentences about where this argument might break or be limited.\n"
        "    * 'use_when': when in the debate this argument is most powerful.\n"
        "- Anticipates AT LEAST 5 strong counter-arguments from opposing viewpoints. For each include:\n"
        "    * 'from_side': which kind of opponent would raise it.\n"
        "    * 'attack': 2–3 sentences summarizing the objection as fairly and strongly as possible.\n"
        "    * 'why_plausible': 2–3 sentences explaining why a smart critic might believe this.\n"
        "    * 'counter_strategy': 3–5 sentences giving the logical way to respond.\n"
        "    * 'prewritten_counter': 3–6 sentences that could be almost directly used in the debate.\n"
        "- Identifies 2–4 genuine weaknesses or edge cases for this position and how the agent should acknowledge and reframe them.\n"
        "- Proposes 4–8 probing questions the agent can ask others to expose contradictions or missing details.\n"
        "- Describes an overall debate_strategy object with: 'tone', 'priority_order' (list of claims to push first), and 'red_lines'.\n"
        "- Ends with a compact summary_for_prompt (roughly 300–600 tokens) that distills the MOST important content for use at runtime.\n"
    )

    # Get raw response - no JSON parsing, just use raw output
    raw = invoke_raw(model, system_prompt, user_prompt)
    
    print(f"\n{'='*80}")
    print(f"AGENT BRIEF RAW OUTPUT for {name}:")
    print(f"{'='*80}")
    print(raw)
    print(f"{'='*80}\n")
    
    logger.info("Raw brief output for agent %s (full): %s", name, raw)
    
    # Create brief structure with raw output stored
    brief = {
        "agent_id": agent_index,
        "name": name,
        "topic": topic,
        "position": position,
        "role_summary": vp_summary or position,
        "supporting_arguments": [],
        "anticipated_opponent_arguments": [],
        "self_weaknesses": [],
        "questions_to_ask": [],
        "debate_strategy": {},
        "summary_for_prompt": f"{position}\n\n{vp_summary}",
        "raw_json_response": raw,  # Store the raw response
    }
    
    # Try to parse JSON if possible, but don't fail if it doesn't work
    if raw:
        try:
            cleaned = _extract_json_object(raw)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                # Merge parsed fields into brief, but keep raw response
                for key, value in parsed.items():
                    if key not in ["raw_json_response"]:
                        brief[key] = value
                logger.info("Successfully parsed JSON for %s", name)
        except Exception as e:
            logger.info("JSON parsing skipped for %s (using raw output): %s", name, str(e))
            # Continue with raw output - no error

    # Ensure required fields exist and have reasonable defaults
    brief.setdefault("agent_id", agent_index)
    brief.setdefault("name", name)
    brief.setdefault("topic", topic)
    brief.setdefault("position", position)
    brief.setdefault("role_summary", vp_summary or position)
    brief.setdefault("supporting_arguments", [])
    brief.setdefault("anticipated_opponent_arguments", [])
    brief.setdefault("self_weaknesses", [])
    brief.setdefault("questions_to_ask", [])
    brief.setdefault("debate_strategy", {})
    brief.setdefault("summary_for_prompt", f"{position}\n\n{vp_summary}")
    if not isinstance(brief.get("summary_for_prompt", ""), str) or not brief.get("summary_for_prompt", "").strip():
        brief["summary_for_prompt"] = f"{position}\n\n{vp_summary}"

    # Always generate a long-form research dossier text for humans to inspect.
    # This does not need to be JSON; it is stored under 'raw_brief'.
    long_system = (
        "You are a senior research analyst preparing an in-depth dossier for an agent before a debate.\n"
        "Write a long, well-structured document (roughly 1,500–2,500 words) in English.\n"
        "Use clear section headings and bullet points where helpful. Do NOT output JSON.\n"
        "Focus on:\n"
        "1) The agent's position and its theoretical foundation.\n"
        "2) Deep supporting arguments with concrete evidence, examples, or historical analogies.\n"
        "3) Anticipated counter-arguments from smart opponents and how to rebut them.\n"
        "4) Genuine weaknesses or edge cases and how to acknowledge/reframe them.\n"
        "5) Probing questions to pressure opponents.\n"
        "6) A final recommended debate strategy.\n"
    )
    long_user = (
        f"Topic: {topic}\n\n"
        f"Agent name: {name}\n"
        f"Core position: {position}\n"
        f"Short description: {vp_summary}\n\n"
        "Write the dossier now."
    )
    raw_brief = invoke_raw(model, long_system, long_user)
    brief["raw_brief"] = raw_brief or brief.get("summary_for_prompt", "")

    # Save per-agent brief to disk
    briefs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    brief_path = briefs_dir / f"brief_agent{agent_index}_{safe_name}.json"
    try:
        with open(brief_path, "w") as f:
            json.dump(brief, f, indent=2, ensure_ascii=False)
        logger.info("Saved agent brief for %s to %s", name, brief_path)
    except Exception as e:
        logger.warning("Failed to save agent brief for %s: %s", name, e)

    return brief


def run_interactive_debate(
    topic: str,
    max_agents: int,
    num_rounds: int,
    models_cfg_path: str,
    pairing: str,
    output_dir: Path,
    question_type_id: str = "1",
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
    pairing_cfg = models_cfg["pairings"][pairing]
    
    # Create models for different agents
    agent_models = [
        LLMFactory.make(**pairing_cfg["A"]),  # Agent 1 uses model A
        LLMFactory.make(**pairing_cfg["B"]),  # Agent 2 uses model B
    ]
    # For third agent, use model C if available, otherwise use A
    if "C" in pairing_cfg:
        agent_models.append(LLMFactory.make(**pairing_cfg["C"]))
    else:
        agent_models.append(LLMFactory.make(**pairing_cfg["A"]))  # Fallback to A
    
    model_a_name = f"{pairing_cfg['A'].get('provider', 'unknown')}/{pairing_cfg['A'].get('model', 'unknown')}"
    model_b_name = f"{pairing_cfg['B'].get('provider', 'unknown')}/{pairing_cfg['B'].get('model', 'unknown')}"
    model_c_name = f"{pairing_cfg.get('C', {}).get('provider', 'unknown')}/{pairing_cfg.get('C', {}).get('model', 'A (fallback)')}"
    
    print(f"\n{'='*80}")
    print(f"AGENT MODEL ASSIGNMENT:")
    print(f"{'='*80}")
    print(f"Agent 1 (Model A): {model_a_name}")
    print(f"Agent 2 (Model B): {model_b_name}")
    print(f"Agent 3 (Model C): {model_c_name}")
    print(f"{'='*80}\n")
    
    logger.info("Created %d agent models: A=%s, B=%s, C=%s", 
                len(agent_models), model_a_name, model_b_name, model_c_name)
    
    # Use model A for viewpoint generation and other non-agent tasks
    base_model = agent_models[0]

    logger.info("Generating viewpoints for topic: %s", topic)
    viewpoints = generate_viewpoints(base_model, topic, max_agents)
    logger.info("Got %d viewpoints", len(viewpoints))

    # Prepare per-agent research briefs (pre-debate files)
    briefs_dir = output_dir / "briefs" / question_type_id
    agent_briefs: List[Dict[str, Any]] = []

    # Build agent personas
    agents: List[Dict[str, Any]] = []
    for idx, vp in enumerate(viewpoints, start=1):
        name = vp["name"]
        
        # Assign model to agent (cycle through available models)
        agent_model = agent_models[(idx - 1) % len(agent_models)]
        model_key = ["A", "B", "C"][(idx - 1) % len(agent_models)]
        model_info = f"{pairing_cfg[model_key].get('provider', 'unknown')}/{pairing_cfg[model_key].get('model', 'unknown')}"
        
        print(f"Agent {idx} ({name}) -> Model {model_key}: {model_info}")
        logger.info("Agent %d (%s) assigned model: %s (from %s)", idx, name, model_info, model_key)

        brief = generate_agent_brief(agent_model, topic, vp, idx, briefs_dir)
        agent_briefs.append(brief)

        system_prompt = (
            f"You are an agent named '{name}'.\n"
            f"Your core position on the topic is: {vp['position']}.\n"
            "In the debate, you MUST consistently argue from this perspective.\n"
            "Your primary goal is to REBUT other agents, not to agree with them.\n"
            "Actively look for conflicts and contradictions in their statements and exploit them.\n"
            "When possible, quote 1–2 short phrases from others and respond with patterns like "
            "\"you said X, but you ignore Y\" or \"you claim X, however Y shows the opposite\".\n"
            "Be concise, sharp, and argumentative, while remaining professional.\n"
            "\n"
            "You also have the following preparation notes (do NOT reveal them verbatim; "
            "use them as an internal memory for your reasoning):\n"
            f"{brief.get('summary_for_prompt', '')}\n"
        )
        agents.append(
            {
                "name": name,
                "position": vp["position"],
                "summary": vp["summary"],
                "system_prompt": system_prompt,
                "messages": [],
                "brief": brief,
                "model": agent_model,  # Assign specific model to this agent
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
            # Use agent-specific model instead of base_model
            agent_model = agent.get("model", base_model)
            model_info = f"{agent_model.provider}/{agent_model.model}" if hasattr(agent_model, 'provider') else "unknown"
            logger.info("Using model for agent '%s': %s", name, model_info)
            reply = invoke_raw(agent_model, sys, user_prompt)
            if not reply:
                reply = "[No response due to API error; treating as empty turn.]"
            
            print(f"\n{'='*80}")
            print(f"ROUND {r} - {name}:")
            print(f"{'='*80}")
            print(reply)
            print(f"{'='*80}\n")
            
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
    
    print(f"\n{'='*80}")
    print(f"JUDGE SUMMARY:")
    print(f"{'='*80}")
    print(judge_summary)
    print(f"{'='*80}\n")

    # Final academic-style report for instructor / professor
    report_system = (
        "You are an expert research writer preparing a high-quality report for a professor.\n"
        "Your task is to synthesize a multi-agent debate on a given topic into a clear, well-structured report.\n"
        "Write in a formal, objective, and academically oriented style (but not like a paper submission).\n"
        "You have access to a web search tool; you may consult external information when helpful, "
        "and you MUST attribute concrete facts or external claims to specific sources in a References section.\n"
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
        "6. References: a bullet list (max 8 items) of the most important sources you used, with URLs and 1-line annotations.\n"
        "The report should be self-contained and should not mention being an AI model or referencing 'the debate' mechanics.\n"
    )

    logger.info("Calling model to generate final report.")
    final_report = invoke_raw(base_model, report_system, report_user)
    
    print(f"\n{'='*80}")
    print(f"FINAL REPORT:")
    print(f"{'='*80}")
    print(final_report)
    print(f"{'='*80}\n")

    # Fallback 1: if the long-context report fails (empty string), try a shorter prompt
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

    # Fallback 2: if LLM still returns nothing, build a deterministic template-based report
    if not final_report:
        logger.error("Final report still empty after LLM retries; using deterministic template-based report.")
        lines = []
        # 1. Context
        lines.append("1. Research Question & Context")
        lines.append(f"   - Topic: {topic}")
        lines.append(
            "   - This report summarizes a multi-agent debate with several distinct viewpoints "
            "generated by an LLM and then refined through multi-round argumentation."
        )
        # 2. Summary of Viewpoints
        lines.append("")
        lines.append("2. Summary of Viewpoints")
        for idx, vp in enumerate(viewpoints, start=1):
            lines.append(f"   - Agent {idx} ({vp['name']}): {vp['position']}")
            if vp.get("summary"):
                lines.append(f"       • Rationale: {vp['summary']}")
        # 3. Key Conflicts
        lines.append("")
        lines.append("3. Comparative Analysis & Key Conflicts")
        lines.append(
            "   - Agents disagree on both the desirability and the acceptable risk level of the proposal."
        )
        lines.append(
            "   - Pro-style agents emphasize potential benefits and opportunities, while more cautious "
            "agents focus on sovereignty, systemic risk, or ethical and distributional concerns."
        )
        lines.append(
            "   - Conditional or moderate agents typically argue for tightly scoped pilots or safeguards, "
            "attempting to reconcile innovation with control."
        )
        # 4. Tentative Conclusion
        lines.append("")
        lines.append("4. Tentative Conclusion & Recommendation")
        lines.append(
            "   - Given the diversity of arguments, a reasonable tentative recommendation is a phased, "
            "evidence-driven approach: start with limited pilots and strong monitoring, while explicitly "
            "defining success criteria and red lines informed by the more conservative viewpoints."
        )
        # 5. Limitations
        lines.append("")
        lines.append("5. Limitations & Suggestions for Further Investigation")
        lines.append(
            "   - This report is based solely on LLM-generated arguments and may miss empirical constraints, "
            "domain-specific regulations, or political feasibility considerations."
        )
        lines.append(
            "   - Future work should incorporate real data, expert interviews, and stress-testing of the "
            "policy options under adverse scenarios (e.g., crisis conditions, adversarial misuse)."
        )
        final_report = "\n".join(lines)

    # Build result payload
    result = {
        "topic": topic,
        "viewpoints": viewpoints,
        "agent_briefs": agent_briefs,
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


