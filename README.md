# 🚀 CS224V: CollectiveMind: Multi-Agent Deliberation for High-Stakes Decision Reasoning 

```
A conversational agent based on multi-agent system for research tasks

--- CollectiveMind Authors
```


# 🏗️ **6 Multi-Agent tools**  
   - Our current version MACI support LangGraph multi-agent tool.
   - Our previous MACI-framework could support 6 multi-agent tools: SagaLLM, Magentic, AutoGen, Crewai, LangChain, LangGraph, and to be extended on LlamaIndex, Haystack.

# 🤝 **20+ LLM Agents**  
   - **[OpenAI LLMs:](https://openai.com/)** including gpt-5.1, gpt-5-search-api, gpt-4, gpt-4o, gpt-4o-mini, etc.
   - **[Ollama:](https://ollama.com/)** including Llama 3.3, DeepSeek-R1, Phi-4, Mistral, Gemma 2, etc.
   - **[Anthropic:](https://www.anthropic.com/)** including Claud 3.7.
   - **[Mistral:](https://mistral.ai/)** 
   - **[LiteLLM:](https://docs.litellm.ai/)** 
   - **or any other OpenAI schema-compatible model** 
   - **FinAgent(ours, previous work):** **Access to financial database such as [Alphavantage](https://www.alphavantage.co/documentation/), and yahoo api.** adapts to unexpected changes in real-time financial information.
   - **Your Self-developed LLMs(ours)**

# 🤖 **Interactive Back-End App**  
## ⚠️ 1. LLM Limitations in Complex Planning  

Large Language Models (LLMs) excel at pattern recognition but struggle with complex planning tasks that require:  

- 🧠 **Deliberate reasoning**  
- ⏳ **Temporal awareness**  
- 📏 **Constraint management**  

### 🔍 1.1 Key Limitations of Current LLM Models:  

1. ❌ **Lack of Self-Verification**  
   - LLMs cannot validate their own outputs, leading to errors.  

2. 🎯 **Attention Bias & Constraint Drift**  
   - Contextual focus shifts, ignoring earlier constraints.  

3. 🏗️ **Lack of Common Sense Integration**  
   - Omits real-world constraints (e.g., logistics delays).  

---

## 🤖 2. MACI: Multi-Agent Collaborative Intelligence  

MACI is designed to overcome these LLM limitations using a three-layer approach:  

1. 🏗️ **Meta-Planner (MP)**  
   - Constructs task-specific workflows, identifying roles and constraints.  

2. 🤝 **Common & Task-Specific Agents**  
   - **Common Agents:** Validate constraints & reasoning quality.  
   - **Task-Specific Agents:** Optimize domain-specific tasks.  

3. 👥 **Multi-agent Debate via Information Theory**

4. 📡 **Run-Time Monitor**  
   - Adapts to unexpected changes in real-time.  

---

## 🔄 3. Multi-Agent Debate System: Pairs & Features

The system supports **both local model pairs and OpenAI API pairs**, providing flexible deployment options for different use cases and resource constraints.

### 🤝 Supported Model Pairs

- **Local Model Pairs**: Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct (self-debate and cross-model debates)
- **OpenAI API Pairs**: GPT-4o, GPT-5-search-api, GPT-3.5-turbo, and other OpenAI models
- **Hybrid Configurations**: Mix local and API models for cost-effective experimentation

### 🎯 Key Feature Pairs

The system offers four key feature dimensions that can be configured based on your needs:

| Feature Pair | Option 1 | Option 2 | Use Case |
|--------------|----------|----------|----------|
| **Agent Architecture** | Multi-Agent System | Single Agent | Multi-agent enables diverse perspectives and debate; Single agent for focused analysis |
| **Interaction Mode** | Debate Mode | Collaboration Mode | Debate for adversarial reasoning; Collaboration for consensus-building |
| **Memory Management** | File Logging | Context Window | File logging for persistent history; Context for in-memory efficiency |
| **Information Source** | Real-Time Search | Closed Database | Real-time search for current information; Closed database for controlled experiments |

### 🔧 Configuration Examples

**Debate Mode with Real-Time Search:**
- Best for: Research tasks requiring current information
- Configuration: Multi-agent + Debate + Real-Time Search + File Logging

**Collaboration Mode with Closed Database:**
- Best for: Reproducible experiments on fixed datasets
- Configuration: Multi-agent + Collaboration + Closed Database + Context Window

---

## 📅 4. Project Plan  

### 🤖 4.1 Functionalities:  
Different LLMs agent tailored for **Decision-making task**:

- 🔍 **(1) Stock Prediction(buy/sell)✅**: Real-time data integration (Alphaventage API)
- 🏗 **(2) Company/News Sentiment Analysis(good/bad)✅**:  Real-time data integration (Alphaventage API)
- 📈 **(2) Personal Insights(good/bad)✅**: Real-time data integration (Alphaventage API)
- 👥 **(4) Job Market Trends(up/down)✅: Investment Expert Analysis & Job Analysis**: Real-time data integration (Alphaventage API)
  
---

## 🔹 5. Multi-Agent Debate System Architecture

### Architecture Overview
- **Debate Protocol**: 2 agents (A, B) exchange arguments in 6 rounds, probabilities + rationales per choice.
- **Per-Round Judge**: Independent judge model evaluates outputs after each round, computes CRIT_A / CRIT_B.
- **CRIT Scoring**: LLM-based algorithm using judge prompts to evaluate argument quality and reliability.
- **Metrics**: KL Divergence, JSD, Wasserstein Distance, Mutual Information, Entropy, Information Gain, AvgCRIT.
- **Datasets**: 7 benchmarks covering arithmetic, medical knowledge, logic, commonsense, and ethical reasoning.

### Supported Question Types

The system can handle **17 types of questions** across different reasoning and decision-making domains:

| Category | ID | Question Type | Description | Example |
|----------|----|--------------|-------------|---------|
| Basic Selection | 1 | Single-Choice QA | Select one correct answer from multiple options | Which is the best investment strategy: A) Diversification, B) All-in tech stocks, C) Real estate only, D) Cryptocurrency? |
| Basic Selection | 2 | Multiple-Choice QA | Select one or more correct answers from multiple options | Which factors affect stock prices? (Select all) A) Interest rates, B) Company earnings, C) Market sentiment, D) Weather |
| Basic Selection | 3 | Decision-Making (Yes/No) | Binary decisions requiring clear yes/no answers with justification | Should I buy Apple stock on 2026/1/1? |
| Analytical | 4 | Factor Brainstorming | Identify 1 to N factors, each with different levels of possibilities/probabilities | Should autonomous driving allow human intervention during operation? What factors should be considered? |
| Analytical | 5 | Mathematical Questions | Calculation, prediction, and regression problems requiring numerical reasoning | How much will Apple stock increase/decrease from 2025/12/2 to 2026/1/1? |
| Analytical | 6 | Ranking/Ordering | Rank options from most to least important/effective/feasible with justification | Rank these investment options from most to least risky: Stocks, Bonds, Real Estate, Cryptocurrency |
| Analytical | 7 | Comparative Analysis | Compare and contrast multiple options across several dimensions | Compare and contrast the benefits and risks of electric vehicles vs. traditional gasoline vehicles |
| Strategic | 8 | Trade-off Analysis | Analyze trade-offs between competing factors or options | What are the trade-offs between investing in growth stocks vs. value stocks? |
| Strategic | 9 | Scenario-Based/Conditional | Answer questions based on hypothetical scenarios ("If X happens, then...") | If the Federal Reserve raises interest rates by 2%, how should I adjust my investment portfolio? |
| Strategic | 10 | Counterfactual | Explore alternative outcomes ("What would happen if X had not occurred?") | What would have happened to the stock market if the 2008 financial crisis had been prevented? |
| Strategic | 11 | Preference Elicitation | Determine preferred combinations of factors or multi-attribute preferences | What combination of risk tolerance, time horizon, and return expectations would you prefer for your investment portfolio? |
| Assessment | 12 | Risk Assessment | Quantify risks, probabilities, and uncertainties associated with decisions | What are the risks and probabilities of investing in emerging market stocks? |
| Assessment | 13 | Causal Reasoning | Identify causes and effects, analyze causal chains and mechanisms | What caused the recent volatility in tech stocks? |
| Assessment | 14 | Policy/Strategy | Recommend concrete policies or strategies to achieve specific goals | What investment strategy should be implemented to achieve 10% annual returns with moderate risk? |
| Complex Reasoning | 15 | Open-Ended Synthesis | Generate solutions or approaches without fixed answer choices | How should I approach building a diversified investment portfolio? |
| Complex Reasoning | 16 | Temporal/Sequential | Determine ordering of actions or events over time | What should be the sequence of steps for launching a new product: market research, product development, testing, or marketing? |
| Complex Reasoning | 17 | Resource Allocation | Optimize resource distribution under constraints and budgets | How should I allocate $100,000 across stocks (60%), bonds (30%), and cash (10%) to maximize returns while minimizing risk? |

---

## 🚀 6. How to Run the Multi-Agent Debate System

### Installation

```bash
cd MAD-main
pip install -r requirements.txt
```

> If you only want to use the OpenAI-based interactive debate (no local models), it is enough to install `requirements.txt` and set `OPENAI_API_KEY`.

### Setting up Local Models (Optional)

To use local models (Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct):

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

### Interactive Debate Playground

This repo includes an **interactive front-end** where a user can type any free-form topic, automatically generate multiple viewpoints (agents), run a multi-round debate, and receive a **professor-level final report**.

#### Backend Setup

1. Make sure dependencies are installed:
   ```bash
   cd MAD-main
   pip install -r requirements.txt
   pip install fastapi "uvicorn[standard]"
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

3. Ensure `configs/models.yaml` has at least one OpenAI pairing configured.

#### Run the Interactive Server

```bash
cd MAD-main
export OPENAI_API_KEY="your_key_here"
uvicorn app_interactive:app --reload --port 8001
```

- The server will be available at `http://127.0.0.1:8001`
- The main UI lives at: `http://127.0.0.1:8001/interactive`

#### Using the Front-End

1. Open `http://127.0.0.1:8001/interactive` in a browser
2. Input a **Debate topic** (any English or Chinese sentence)
3. Choose:
   - **Max agents (viewpoints)** – how many distinct viewpoints to generate (2–8)
   - **Rounds** – how many debate rounds (1–6)
4. Click **Run debate**

The system will:
- Round 0: run a *research/setup* step to generate multiple viewpoints and turn them into agents
- Rounds 1..N: let all agents take turns attacking each other's arguments
- Judge: synthesize a **final report** with academic-style structure

### Running Benchmark Debates

#### Quick Start (Single Example)
```bash
cd MAD-main
python -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

#### Full Benchmark Run
```bash
# Run all pairings × datasets
python -m src.runners.run_benchmark \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

---

## ⚙️ 7. Experiment and Evaluation

### 7.1 Set-up:
7.1.1 Debate Protocol

#### Round Structure
1. **Round 1**: Initial analysis (contentiousness: 0.9)
2. **Round 2**: Confrontational debate (contentiousness: 0.9)
3. **Round 3**: Balanced discussion (contentiousness: 0.7)
4. **Round 4**: Middle ground exploration (contentiousness: 0.5)
5. **Round 5**: Supportive discussion (contentiousness: 0.3)
6. **Round 6**: Final synthesis (contentiousness: 0.1)


### 📜 7.2 Datasets:

7.2.1 QA datasets:
1. **Arithmetic** (100 questions): Custom arithmetic reasoning dataset
2. **GSM8K** (300 questions): Mathematical word problems
3. **MMLU Professional Medicine** (272 questions): Medical knowledge assessment
4. **MMLU Formal Logic** (126 questions): Logical reasoning problems
5. **HellaSwag** (300 questions): Commonsense reasoning
6. **CommonSenseQA** (300 questions): Commonsense question answering
7. **HH-RLHF** (300 questions): Helpful and Harmless RLHF dataset

7.2.2 Human evaluation datasets


### ⚖️ 7.3 Baselines:  

7.3.1. 📊 **OpenAI Deepresearch**  

7.3.2. 🆚 **STORM/CO-STORM**  

### ⚖️ 7.4 Metrics: 

#### 7.4.1: Information-Theoretic Metrics
- **KL Divergence**: Measures disagreement between agents
- **Jensen-Shannon Distance**: Symmetric measure of distribution difference
- **Wasserstein Distance**: Earth mover's distance between distributions
- **Mutual Information**: Information shared between agents
- **Entropy**: Uncertainty in agent responses
- **Information Gain**: Reduction in uncertainty over rounds

#### 7.4.2: CRIT Scores
- **CRIT_A/CRIT_B**: Judge's evaluation of argument quality (0-1)
- **AvgCRIT**: Average CRIT score across agents

---

## 🎓 8. Previous Publications  

1. 📄 **Paper: Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations** -*Yucheng Jiang, Yijia Shao, Dekun Ma, Sina J. Semnani, Monica S. Lam*
2. 📄 **Paper: Multi-Agent Collaborative Intelligence for Robust Temporal Planning** – *Edward Y. Chang*  
3. 📄 **Paper: REALM-Bench: A Real-World Planning Benchmark for LLMs and Multi-Agent Systems** – *Longling Gloria Geng, Edward Y. Chang*  
4. 💻 **GitHub Setup, App Development, and Experiments** – *Gloria Longling Geng and Henry Zengxiao He*  


---

This **README** provides an overview of the **CS224V-MACI-for-Research-Agent** project, highlighting its **motivations, project plan, methodologies, demo, and future directions.** 🚀  

---

## 🔹 Citation

If you use this code, please cite:
```bibtex
@article{collectivemind2024,
  title={CollectiveMind: Multi-Agent Deliberation for High-Stakes Decision Reasoning},
  author={...},
  journal={...},
  year={2024}
<<<<<<< HEAD
}
=======
}
>>>>>>> 3b6b5d98dd7e5c6ad811bfb109337cf0dc54e8b4
