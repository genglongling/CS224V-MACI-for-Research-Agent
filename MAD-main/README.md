# CollectiveMind Experiment Guide

This document explains how to reproduce the "Key Point Extraction & Pairwise Comparison" experiment described in Section 3.6 of our paper.

## 1. Dataset Structure

Our experiment uses **60 examples** organized into **3 categories** with **20 question types** each:

- **Finance**: 20 examples (Question Types 1-20)
- **AI Governance**: 20 examples (Question Types 1-20)
- **Social Policy**: 20 examples (Question Types 1-20)

Each example corresponds to one of the 20 supported question types, ensuring comprehensive coverage across different reasoning and decision-making domains.

The topics are stored in `topics.json` with the following structure:
```json
{
  "finance": {
    "1": "Question for Single-Choice QA...",
    "2": "Question for Multiple-Choice QA...",
    ...
    "20": "Question for Ambiguity/Incomplete Information..."
  },
  "ai_governance": {
    "1": "...",
    ...
  },
  "social_policy": {
    "1": "...",
    ...
  }
}
```

## 2. Setup

Ensure you have the dependencies installed and your OpenAI API key set:

```bash
cd MAD-main
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

## 3. Running the Experiment

### Running All 60 Examples

To run the full evaluation on all 60 examples (3 categories × 20 question types):

```bash
# Run on all 60 examples (reproduce paper results)
python3 run_experiment.py --topics 60
```

### Running by Category

You can also run experiments by category:

```bash
# Run only Finance examples (20 examples)
python3 run_experiment.py --category finance --topics 20

# Run only AI Governance examples (20 examples)
python3 run_experiment.py --category ai_governance --topics 20

# Run only Social Policy examples (20 examples)
python3 run_experiment.py --category social_policy --topics 20
```

### Running a Subset for Testing

```bash
# Run on 1 example (test mode)
python3 run_experiment.py --topics 1

# Run on 5 examples (quick test)
python3 run_experiment.py --topics 5
```

## 4. Experiment Process

The script performs the following steps for each example:

1. **Baseline**: Generates single-agent Deep Research Briefs and synthesizes a report *without* debate.
2. **CollectiveMind**: Runs the full multi-agent debate pipeline and synthesizes a final report.
3. **Evaluation**:
   - Extracts **Key Points** and evidence spans from both reports.
   - Performs **Pairwise Comparison** using a Judge LLM to determine which system provided better support for each key point.

## 5. Output Structure

### Individual Example Outputs

Each example generates a **complete JSON file** containing the full debate process and metrics:

**File Location**: `results/experiment/{category}/{question_type_id}_{topic_slug}.json`

**Example Files**:
```
results/experiment/
├── finance/
│   ├── 1_single_choice.json
│   ├── 2_multiple_choice.json
│   ├── 3_decision_making.json
│   ├── ...
│   └── 20_ambiguity_handling.json
├── ai_governance/
│   ├── 1_single_choice.json
│   ├── ...
│   └── 20_ambiguity_handling.json
└── social_policy/
    ├── 1_single_choice.json
    ├── ...
    └── 20_ambiguity_handling.json
```

### JSON File Structure

Each individual JSON file contains:

```json
{
  "category": "finance",
  "question_type_id": 1,
  "question_type": "Single-Choice QA",
  "topic": "Which is the best investment strategy...",
  
  "baseline": {
    "briefs": [...],
    "report": "...",
    "key_points": [...]
  },
  
  "collectivemind": {
    "viewpoints": [...],
    "debate_state": {
      "A": {"r1": {...}, "r2": {...}, ...},
      "B": {"r1": {...}, "r2": {...}, ...},
      "final": {...}
    },
    "complete_transcript": [...],
    "final_report": "...",
    "key_points": [...]
  },
  
  "evaluation": {
    "baseline_kps": 11,
    "cm_kps": 10,
    "baseline_evidence_count": 8,
    "cm_evidence_count": 9,
    "pairwise_comparisons": [
      {
        "key_point": "...",
        "baseline_evidence": "...",
        "cm_evidence": "...",
        "winner": "B",
        "reason": "..."
      },
      ...
    ],
    "wins": 15,
    "losses": 6,
    "ties": 0
  },
  
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Aggregated Results

The final aggregated results are saved to `results/experiment_final.json`:

```json
{
  "stats": {
    "total_examples": 60,
    "categories": {
      "finance": 20,
      "ai_governance": 20,
      "social_policy": 20
    },
    "baseline_kps": 366,
    "collective_kps": 522,
    "baseline_evidence_count": 264,
    "collective_evidence_count": 423,
    "cm_wins": 345,
    "cm_losses": 177,
    "ties": 12,
    "win_rate": 0.66
  },
  "per_category": {
    "finance": {
      "baseline_kps": 122,
      "collective_kps": 174,
      "win_rate": 0.68,
      ...
    },
    ...
  },
  "topics": [
    {
      "category": "finance",
      "question_type_id": 1,
      "topic": "...",
      "wins": 15,
      "losses": 6,
      "ties": 0
    },
    ...
  ]
}
```

## 6. Results Interpretation

### Latest Run (Sample)

**Example**: Finance - Question Type 1 (Single-Choice QA)
- **Topic**: "Which is the best investment strategy for a 30-year retirement plan..."
- **Win Rate**: **0.71** (CollectiveMind won 15 out of 21 comparisons)
- **Key Points**: Baseline (11), CollectiveMind (10)
- **Evidence Grounding**: Baseline (73%), CollectiveMind (90%)

### Key Metrics

- **Win Rate**: Fraction of pairwise comparisons where CollectiveMind provided better support (target: >0.60)
- **#Key Points**: Average number of distinct key points per topic
- **Evidence@Key**: Fraction of key points grounded in explicit evidence spans

## 7. Verifying Output

After running the experiment, verify that:

1. **All 60 JSON files exist**: Check `results/experiment/{category}/` directories
2. **Each file contains complete data**: Verify presence of `debate_state`, `baseline`, `collectivemind`, and `evaluation` sections
3. **Aggregated results are generated**: Check `results/experiment_final.json` exists and contains summary statistics

```bash
# Count output files
find results/experiment -name "*.json" | wc -l  # Should be 60

# Check file structure
python3 -c "import json; f=open('results/experiment/finance/1_single_choice.json'); d=json.load(f); print('Keys:', list(d.keys()))"
```
