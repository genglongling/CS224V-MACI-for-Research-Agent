# CollectiveMind: Multi-Agent Deliberation for High-Stakes Decision Reasoning 

This repository runs **multi-agent debates** over multiple-choice QA datasets, using **LangGraph** pipelines.  
Each debate consists of **6 rounds** (contentiousness 0.9 → 0.1), with a **judge invoked every round**.  
We log **per-round information-theoretic metrics** and **LLM-based CRIT scores**.
The systems support LangGraph + 6-Round Protocol + LLM-as-a-Judge (CRIT).

---

## 🔹 Features
- **Debate Protocol**: 2 agents (A, B) exchange arguments in 6 rounds, probabilities + rationales per choice.
- **Per-Round Judge**: Independent judge model evaluates outputs after each round, computes CRIT_A / CRIT_B.
- **CRIT Scoring**: LLM-based algorithm using judge prompts to evaluate argument quality and reliability.
- **Metrics**: KL Divergence, JSD, Wasserstein Distance, Mutual Information, Entropy, Information Gain, AvgCRIT.
- **Datasets**: 7 benchmarks covering arithmetic, medical knowledge, logic, commonsense, and ethical reasoning.
- **Pairings**:  
  1. Qwen2.5-7B-Instruct vs Qwen2.5-7B-Instruct (self-debate), Qwen2.5-7B-Instruct (Judge)
  2. Llama3.1-8B-Instruct vs Llama3.1-8B-Instruct (self-debate), Llama3.1-8B-Instruct (Judge)
  3. Qwen2.5-7B-Instruct vs Llama3.1-8B-Instruct
---

## 🔹 Supported Question Types

The system can handle **17 types of questions** across different reasoning domains:

| Category | ID | Question Type | Description |
|----------|----|--------------|-------------|
| Basic Selection | 1 | Single-Choice QA | Select one correct answer from multiple options |
| Basic Selection | 2 | Multiple-Choice QA | Select one or more correct answers from multiple options |
| Basic Selection | 3 | Decision-Making (Yes/No) | Binary decisions requiring clear yes/no answers with justification |
| Analytical | 4 | Factor Brainstorming | Identify 1 to N factors, each with different levels of possibilities/probabilities |
| Analytical | 5 | Mathematical Questions | Calculation, prediction, and regression problems requiring numerical reasoning |
| Analytical | 6 | Ranking/Ordering | Rank options from most to least important/effective/feasible with justification |
| Analytical | 7 | Comparative Analysis | Compare and contrast multiple options across several dimensions |
| Strategic | 8 | Trade-off Analysis | Analyze trade-offs between competing factors or options |
| Strategic | 9 | Scenario-Based/Conditional | Answer questions based on hypothetical scenarios ("If X happens, then...") |
| Strategic | 10 | Counterfactual | Explore alternative outcomes ("What would happen if X had not occurred?") |
| Strategic | 11 | Preference Elicitation | Determine preferred combinations of factors or multi-attribute preferences |
| Assessment | 12 | Risk Assessment | Quantify risks, probabilities, and uncertainties associated with decisions |
| Assessment | 13 | Causal Reasoning | Identify causes and effects, analyze causal chains and mechanisms |
| Assessment | 14 | Policy/Strategy | Recommend concrete policies or strategies to achieve specific goals |
| Complex Reasoning | 15 | Open-Ended Synthesis | Generate solutions or approaches without fixed answer choices |
| Complex Reasoning | 16 | Temporal/Sequential | Determine ordering of actions or events over time |
| Complex Reasoning | 17 | Resource Allocation | Optimize resource distribution under constraints and budgets |

---

## 🔹 Code Structure

```
MAD/
├── configs/                    # Configuration files
│   ├── benchmark.yaml         # Main benchmark settings
│   ├── datasets.yaml          # Dataset configurations
│   ├── models.yaml            # Model pairings and settings
│   └── prompts.yaml           # Debate and judge prompts
├── data/                      # Local datasets
│   └── arithmetic/
│       └── dev.jsonl          # Custom arithmetic dataset (100 questions)
├── results/                   # Output directory
│   ├── runs/                  # Individual debate results
│   ├── metrics/               # Aggregated metrics
│   └── tables/                # LaTeX tables
├── scripts/                   # Utility scripts
│   ├── setup_local_models.py  # Download local models
│   └── download_datasets.sh   # Download HuggingFace datasets
└── src/                       # Source code
    ├── debate/                # Core debate system
    │   ├── graph.py           # LangGraph debate pipeline
    │   ├── models.py          # Model wrappers (OpenAI, Local, etc.)
    │   ├── prompts.py         # Response parsing and validation
    │   └── metrics.py         # Information-theoretic metrics
    ├── datasets/              # Dataset loaders
    └── runners/               # Execution scripts
        ├── run_benchmark.py   # Main benchmark runner
        ├── export_table.py    # Results export
        └── interactive_debate.py  # Interactive user-topic debate (no datasets)
```

---

## 🔹 Installation
```bash
git clone <this-repo>
cd MAD
pip install -r requirements.txt
```

> If you only want to use the OpenAI-based interactive debate (no local models), it is enough to install `requirements.txt` and set `OPENAI_API_KEY`.

### Setting up Local Models
To use the local models (Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct), you need to download them first:

```bash
# Setup both models (~30GB total)
python scripts/setup_local_models.py --all

# Or setup individually
python scripts/setup_local_models.py --qwen   # Qwen2.5-7B-Instruct (~14GB)
python scripts/setup_local_models.py --llama  # Llama3.1-8B-Instruct (~16GB)
```

**Requirements for local models:**
- Sufficient RAM (16GB+ recommended)
- GPU with sufficient VRAM (8GB+ recommended for 7B/8B models)
- Stable internet connection for initial download
- HuggingFace account with access to Llama models (for Llama3.1-8B-Instruct)

**Note for Llama models:** You may need to request access to Llama models on HuggingFace and login with `huggingface-cli login`.

---

## 🔹 Interactive Debate Playground (User-defined topics)

This repo also includes an **interactive front-end** where a user can type any free-form topic, automatically generate multiple viewpoints (agents), run a multi-round debate, and receive a **professor-level final report**.

### Backend setup (once per machine)

1. Make sure dependencies are installed:
   ```bash
   cd MAD-main
   pip install -r requirements.txt
   # make sure FastAPI + Uvicorn are available
   pip install fastapi "uvicorn[standard]"
   ```
2. Set your OpenAI API key (the same key will be used for research agent, debaters, judge, and final report):
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```
3. Ensure `configs/models.yaml` has at least one OpenAI pairing (we use `qwen_qwen` as the base), for example:
   ```yaml
   pairings:
     qwen_qwen:
       A:
         provider: openai
         model: gpt-5.1        # or any model id your account can use
         temperature: 0.7
         max_tokens: 512
       B:
         provider: openai
         model: gpt-5.1
         temperature: 0.8
         max_tokens: 512
       judge:
         provider: openai
         model: gpt-5.1
         temperature: 0.2
         max_tokens: 1024
       researcher:
         provider: openai
         model: gpt-5.1
         temperature: 0.3
         max_tokens: 512
   ```

### Run the interactive server

From `MAD-main` directory:

```bash
export OPENAI_API_KEY="your_key_here"   # if not already set

uvicorn app_interactive:app --reload --port 8001
```

- The server will be available at `http://127.0.0.1:8001`.
- The main UI lives at: `http://127.0.0.1:8001/interactive`.

### Using the front-end

1. Open `http://127.0.0.1:8001/interactive` in a browser.
2. Input a **Debate topic** (any English or Chinese sentence).
3. Choose:
   - **Max agents (viewpoints)** – how many distinct viewpoints to generate (2–5).
   - **Rounds** – how many debate rounds (1–4).
4. Click **Run debate**.

The system will:
- Round 0: run a *research/setup* step to generate multiple viewpoints and turn them into agents.
- Rounds 1..N: let all agents take turns attacking each other’s arguments with explicit quotes (“you said X, but ignore Y…”).
- Judge: synthesize a **final report** with academic-style structure:
  1. Research question & context  
  2. Summary of viewpoints  
  3. Comparative analysis & key conflicts  
  4. Tentative conclusion & recommendation  
  5. Limitations & suggestions for further investigation  

The front-end shows:
- **Viewpoints / Agents**: automatically generated personas and positions.
- **Final Report**: long-form synthesis suitable for including in project reports or appendices.
- **Debate Transcript**: all agent turns, with explicit `Round 0 (research)` and `Round k` labels.

### Programmatic use (without front-end)

You can also call the interactive debate runner directly from the CLI:

```bash
python -m src.runners.interactive_debate \
  --topic "Should autonomous driving vehicles operate without human labor?" \
  --max_agents 5 \
  --rounds 3 \
  --models configs/models.yaml \
  --pairing qwen_qwen \
  --output_dir results/interactive
```

This will save a JSON file under `results/interactive/interactive_debate_*.json` containing:
- topic, viewpoints, agent messages per round  
- conversation_log (with round numbers)  
- judge_summary and final_report  

---

## 🔹 Datasets

### Supported Datasets (7 total, 1,698 questions)

1. **Arithmetic** (100 questions): Custom arithmetic reasoning dataset
2. **GSM8K** (300 questions): Mathematical word problems
3. **MMLU Professional Medicine** (272 questions): Medical knowledge assessment
4. **MMLU Formal Logic** (126 questions): Logical reasoning problems
5. **HellaSwag** (300 questions): Commonsense reasoning
6. **CommonSenseQA** (300 questions): Commonsense question answering
7. **HH-RLHF** (300 questions): Helpful and Harmless RLHF dataset

### Download Datasets
```bash
# Download HuggingFace datasets
bash scripts/download_datasets.sh

# Custom arithmetic dataset is already included in data/arithmetic/dev.jsonl
```

---

## 🔹 Running Debates

### Quick Start (Single Example)
```bash
# Run with just 1 example per dataset for testing
python -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

### Full Benchmark Run
```bash
# Run all pairings × datasets (1,698 questions total)
python -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

### Configuration Options

**Edit `configs/benchmark.yaml` to customize:**
```yaml
# Which pairings to run
pairings:
  - qwen_qwen      # Qwen self-debate
  # - qwen_llama   # Qwen vs Llama
  # - llama_llama  # Llama self-debate

# Which datasets to run
datasets:
  - arithmetic     # Custom arithmetic dataset
  - gsm8k          # Mathematical reasoning
  - mmlu_pro_med   # Medical knowledge
  - mmlu_formal_logic  # Logical reasoning
  - hellaswag      # Commonsense reasoning
  - commonsenseqa  # Commonsense QA
  - hh_rlhf        # Ethical reasoning
```

**Edit `configs/datasets.yaml` to adjust dataset sizes:**
```yaml
# Example: Change GSM8K to use 100 questions instead of 300
gsm8k: {type: hf, name: gsm8k, subset: main, split: test, max_examples: 100}
```

---

## 🔹 Outputs

### Individual Results
```
results/runs/
├── qwen_qwen_arithmetic_arithmetic_1.json
├── qwen_qwen_gsm8k_gsm8k_1.json
└── ...
```

Each file contains:
- Complete debate transcript (6 rounds)
- Agent responses with probabilities and rationales
- Judge evaluations with CRIT scores
- Information-theoretic metrics

### Aggregated Metrics
```
results/metrics/
└── all_results.json  # Summary of all experiments
```

### LaTeX Tables
```bash
# Export accuracy table
python -m src.runners.export_table

# Export per-round metrics
python -m src.runners.export_table --metrics round
```

---

## 🔹 Debate Protocol

### Round Structure
1. **Round 1**: Initial analysis (contentiousness: 0.9)
2. **Round 2**: Confrontational debate (contentiousness: 0.9)
3. **Round 3**: Balanced discussion (contentiousness: 0.7)
4. **Round 4**: Middle ground exploration (contentiousness: 0.5)
5. **Round 5**: Supportive discussion (contentiousness: 0.3)
6. **Round 6**: Final synthesis (contentiousness: 0.1)

### Response Format
Each agent generates:
```json
{
  "output": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
  "reason": {
    "A": "Rationale for choice A",
    "B": "Rationale for choice B",
    "C": "Rationale for choice C",
    "D": "Rationale for choice D"
  }
}
```

### Judge Evaluation
After each round, the judge provides:
```json
{
  "outputA": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
  "outputB": {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
  "CRIT_A": 0.75,
  "CRIT_B": 0.85,
  "NOTE_A": "Evaluation of Agent A's arguments",
  "NOTE_B": "Evaluation of Agent B's arguments"
}
```

---

## 🔹 Metrics

### Information-Theoretic Metrics
- **KL Divergence**: Measures disagreement between agents
- **Jensen-Shannon Distance**: Symmetric measure of distribution difference
- **Wasserstein Distance**: Earth mover's distance between distributions
- **Mutual Information**: Information shared between agents
- **Entropy**: Uncertainty in agent responses
- **Information Gain**: Reduction in uncertainty over rounds

### CRIT Scores
- **CRIT_A/CRIT_B**: Judge's evaluation of argument quality (0-1)
- **AvgCRIT**: Average CRIT score across agents

---

## 🔹 Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
# Reinstall transformers with correct architecture
pip install --upgrade --force-reinstall transformers torch
```

**Memory Issues:**
```bash
# Reduce batch size in configs/benchmark.yaml
num_workers: 1
batch_size: 1
```

**Dataset Loading Errors:**
```bash
# Check HuggingFace access for Llama models
huggingface-cli login
```

### Performance Tips
- Use GPU for local models (8GB+ VRAM recommended)
- Reduce `max_examples` in datasets.yaml for faster testing
- Use `num_workers: 1` for stability with local models

---

## 🔹 Citation

If you use this code, please cite:
```bibtex
@article{collectivemind2024,
  title={CollectiveMind: Multi-Agent Deliberation for High-Stakes Decision Reasoning},
  author={...},
  journal={...},
  year={2024}
}
```