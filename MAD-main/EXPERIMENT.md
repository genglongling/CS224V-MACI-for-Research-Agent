# CollectiveMind Experiment Guide

This document explains how to reproduce the "Key Point Extraction & Pairwise Comparison" experiment described in Section 3.6 of our paper.

## 1. Setup

Ensure you have the dependencies installed and your OpenAI API key set:

```bash
cd MAD-main
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## 2. Running the Experiment

To run the full evaluation on 20 topics (or a subset), use the `run_experiment.py` script:

```bash
# Run on 1 topic (test mode)
python3 run_experiment.py --topics 1

# Run on all 20 topics (reproduce paper results)
python3 run_experiment.py --topics 20
```

The script performs the following steps for each topic:
1.  **Baseline**: Generates single-agent Deep Research Briefs and synthesizes a report *without* debate.
2.  **CollectiveMind**: Runs the full multi-agent debate pipeline and synthesizes a final report.
3.  **Evaluation**:
    *   Extracts **Key Points** and evidence spans from both reports.
    *   Performs **Pairwise Comparison** using a Judge LLM to determine which system provided better support for each key point.

## 3. Results

Results are saved to `results/experiment_final.json`.

### Latest Run (1 Topic Sample)
*   **Topic**: "Should central banks issue retail digital currencies (CBDCs)?"
*   **Win Rate**: **0.71** (CollectiveMind won 15 out of 21 comparisons)
*   **Key Points**: Baseline (11), CollectiveMind (10)
*   **Evidence Grounding**: 100% for both systems (using GPT-4o).

### Interpreting the JSON
The output JSON contains detailed stats:
```json
{
  "stats": {
    "cm_wins": 15,
    "cm_losses": 6,
    "win_rate": 0.71,
    ...
  },
  "topics": [
    {
      "topic": "...",
      "wins": 15,
      "losses": 6
    }
  ]
}
```

