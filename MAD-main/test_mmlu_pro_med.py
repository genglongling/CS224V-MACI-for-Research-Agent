#!/usr/bin/env python3
"""
Test script for MMLU Professional Medicine dataset
"""
import json
import requests
from datasets import load_dataset
from src.debate.models import LLMFactory


def test_mmlu_pro_med():
    """Test MMLU Professional Medicine dataset"""
    
    # Set up output redirection to file
    import sys
    from datetime import datetime
    
    # Create filename based on model configuration and example number
    filename = f"qwen_qwen_qwen_mmlu_pro_med_1.txt"
    
    # Redirect stdout to both console and file
    original_stdout = sys.stdout
    
    class TeeOutput:
        def __init__(self, filename):
            self.terminal = original_stdout
            self.log = open(filename, 'w', encoding='utf-8')
            self.log.write(f"=== MMLU Professional Medicine Test Output ===\n")
            self.log.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log.write(f"Filename: {filename}\n")
            self.log.write("="*80 + "\n\n")
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Start output redirection
    tee_output = TeeOutput(filename)
    sys.stdout = tee_output
    
    print("=== MMLU Professional Medicine Test ===")
    
    # Check if litgpt models are accessible
    import requests
    import time
    
    print("🔍 Checking if litgpt models are accessible...")
    ports = [8000, 8001, 8003]
    model_names = ["Agent A", "Agent B", "Judge"]
    
    for port, name in zip(ports, model_names):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {name} (port {port}) is accessible")
            else:
                print(f"⚠️ {name} (port {port}) returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {name} (port {port}) is not accessible: {e}")
            print(f"   Make sure to run: python launch_models.py")
            return False
    
    print("✅ All models are accessible!")
    
    # Load dataset
    dataset = load_dataset('cais/mmlu', 'professional_medicine', split='test')
    example = dataset[0]
    
    print(f"Question: {example['question'][:200]}...")
    print(f"Choices: {example['choices']}")
    print(f"Answer: {example['answer']}")
    print(f"Question length: {len(example['question'])} characters")
    
    # Convert to our format
    choices_dict = {}
    for i, choice in enumerate(example['choices']):
        choices_dict[chr(ord('A') + i)] = choice
    
    choices_csv = ", ".join([f"{k}: {v}" for k, v in choices_dict.items()])
    
    # Initialize two debaters
    print("\n--- Initializing Debaters ---")
    print("Using litgpt-served models on different ports...")
    
    # local models
    # Initialize models using litgpt API endpoints
    agent_a = LLMFactory.make("litgpt", "http://localhost:8000", temperature=0.7, max_tokens=1024)
    agent_b = LLMFactory.make("litgpt", "http://localhost:8001", temperature=0.8, max_tokens=1024)
    judge = LLMFactory.make("litgpt", "http://localhost:8003", temperature=0.2, max_tokens=2048)
    
    print("✅ Debaters initialized successfully (litgpt API mode)")
    print("   Agent A: http://localhost:8000")
    print("   Agent B: http://localhost:8001")
    print("   Judge:   http://localhost:8003")
    
    # Store responses for each round
    responses = {'A': {}, 'B': {}}
    
    # Round 1: Agent A -> Agent B -> Judge
    print("\n=== ROUND 1 ===")
    
    # Agent A Round 1
    print("--- Agent A Round 1 ---")
    debater_prompt_a1 = f"""Round 1: Initial Analysis
Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_a1}
        ]
        
        response_a1 = agent_a.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent A Round 1:")
        print(f"   {response_a1.content}")
        print(f"   {'='*80}")
        
        responses['A']['r1'] = response_a1.content
        
    except Exception as e:
        print(f"❌ Agent A Round 1 failed: {e}")
        return False
    
    # Agent B Round 1
    print("--- Agent B Round 1 ---")
    debater_prompt_b1 = f"""Round 1: Initial Analysis
Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_b1}
        ]
        
        response_b1 = agent_b.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent B Round 1:")
        print(f"   {response_b1.content}")
        print(f"   {'='*80}")
        
        responses['B']['r1'] = response_b1.content
        
    except Exception as e:
        print(f"❌ Agent B Round 1 failed: {e}")
        return False
    
    # Judge Round - COMMENTED OUT 1 - COMMENTED OUT
    # print("--- Judge Round 1 ---")
    # judge_prompt_r1 = f"""Round: 1
    # Question: {example['question']}
    # Choices: {choices_csv}

    # # Agent A (Round 1)
    # {responses['A']['r1']}

    # # Agent B (Round 1)
    # {responses['B']['r1']}

    # Evaluate both agents using the CRIT algorithm:
    # 1. Identify the main claims and supporting reasons for each agent
    # 2. Assess the quality of reasoning and evidence
    # 3. Provide CRIT scores (0.0 to 1.0) for each agent
    # 4. Give brief notes on strengths/weaknesses

    # Output format:
    # CRIT_A: [score] - [brief note]
    # CRIT_B: [score] - [brief note]"""

    # try:
    #     messages = [
    #         {"role": "system", "content": "You are a rigorous judge evaluating a debate between two agents on a medical question. Evaluate the debate fairly and provide clear CRIT scores. IMPORTANT: For Probs, always list probabilities in A/B/C/D order as [prob_A, prob_B, prob_C, prob_D]."},
    #         {"role": "user", "content": judge_prompt_r1}
    #     ]
        
    #     judge_response_r1 = judge.invoke(messages)
    #     print(f"🔍 RAW RESPONSE - Judge Round 1:")
    #     print(f"   {judge_response_r1.content}")
    #     print(f"   {'='*80}")
        
    #     print(f"✅ Judge Round 1 response received")
    #     print(f"   Response length: {len(judge_response_r1.content)} characters")
            
    # except Exception as e:
    #     print(f"❌ Judge Round 1 failed: {e}")
    
    # Round 2: Agent A -> Agent B -> Judge
    print("\n=== ROUND 2 ===")
    
    # Agent A Round 2
    print("--- Agent A Round 2 ---")
    debater_prompt_a2 = f"""Round 2: κ = 0.9 (highly contentious)
Refute your opponent's answer and justifications. Press on weak assumptions. You may use careful counterfactuals to stress-test their claims. Then provide your probabilities and justification.

Agent B's previous analysis: {responses['B']['r1']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_a2}
        ]
        
        response_a2 = agent_a.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent A Round 2:")
        print(f"   {response_a2.content}")
        print(f"   {'='*80}")
        
        responses['A']['r2'] = response_a2.content
        
        # print(f"✅ Agent A Round 2 response received")
        # print(f"   Response length: {len(response_a2.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent A Round 2 failed: {e}")
        return False
    
    # Agent B Round 2
    print("--- Agent B Round 2 ---")
    debater_prompt_b2 = f"""Round 2: κ = 0.9 (highly contentious)
Refute your opponent's answer and justifications. Press on weak assumptions. You may use careful counterfactuals to stress-test their claims. Then provide your probabilities and justification.

Agent A's previous analysis: {responses['A']['r1']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_b2}
        ]
        
        response_b2 = agent_b.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent B Round 2:")
        print(f"   {response_b2.content}")
        print(f"   {'='*80}")
        
        responses['B']['r2'] = response_b2.content
        
        # print(f"✅ Agent B Round 2 response received")
        # print(f"   Response length: {len(response_b2.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent B Round 2 failed: {e}")
        return False
    
    # Judge Round - COMMENTED OUT 2
    # print("--- Judge Round 2 ---")
    # judge_prompt_r2 = f"""Judge these medical answers:

    # Question: {example['question'][:200]}...
    # Choices: {choices_csv}

    # Agent A: {responses['A']['r2']}
    # Agent B: {responses['B']['r2']}

    # Rate each (0.0-1.0) with brief note:
    # CRIT_A: [score] - [note]
    # CRIT_B: [score] - [note]"""

    # try:
    #     messages = [
    #         {"role": "system", "content": "You are a rigorous judge evaluating a debate between two agents on a medical question. Evaluate the debate fairly and provide clear CRIT scores. IMPORTANT: For Probs, always list probabilities in A/B/C/D order as [prob_A, prob_B, prob_C, prob_D]."},
    #         {"role": "user", "content": judge_prompt_r2}
    #     ]
        
    #     judge_response_r2 = judge.invoke(messages)
    #     print(f"🔍 RAW RESPONSE - Judge Round 2:")
    #     print(f"   {judge_response_r2.content}")
    #     print(f"   {'='*80}")
        
    #     print(f"✅ Judge Round 2 response received")
    #     print(f"   Response length: {len(judge_response_r2.content)} characters")
            
    # except Exception as e:
    #     print(f"❌ Judge Round 2 failed: {e}")
    
    # Continue with remaining rounds (3-6) following the same pattern
    print("\n=== ROUNDS 3-6 (Full Implementation) ===")
    
    # Round 3: Agent A -> Agent B -> Judge
    print("\n=== ROUND 3 ===")
    
    # Agent A Round 3
    print("--- Agent A Round 3 ---")
    debater_prompt_a3 = f"""Round 3: κ = 0.7 (moderately contentious)
Engage with your opponent's arguments more constructively. Acknowledge valid points while still defending your position. Look for areas of agreement and disagreement.

Agent B's previous analysis: {responses['B']['r2']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_a3}
        ]
        
        response_a3 = agent_a.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent A Round 3:")
        print(f"   {response_a3.content}")
        print(f"   {'='*80}")
        
        responses['A']['r3'] = response_a3.content
        
        # print(f"✅ Agent A Round 3 response received")
        # print(f"   Response length: {len(response_a3.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent A Round 3 failed: {e}")
        return False
    
    # Agent B Round 3
    print("--- Agent B Round 3 ---")
    debater_prompt_b3 = f"""Round 3: κ = 0.7 (moderately contentious)
Engage with your opponent's arguments more constructively. Acknowledge valid points while still defending your position. Look for areas of agreement and disagreement.

Agent A's previous analysis: {responses['A']['r2']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_b3}
        ]
        
        response_b3 = agent_b.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent B Round 3:")
        print(f"   {response_b3.content}")
        print(f"   {'='*80}")
        
        responses['B']['r3'] = response_b3.content
        
        # print(f"✅ Agent B Round 3 response received")
        # print(f"   Response length: {len(response_b3.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent B Round 3 failed: {e}")
        return False
    
    # Judge Round - COMMENTED OUT 3
    # print("--- Judge Round 3 ---")
    # judge_prompt_r3 = f"""You are a rigorous judge evaluating a debate between two agents on a medical question.

# Round: 3
# Question: {example['question']}
# Choices: {choices_csv}

# # Agent A (Round 3)
# {responses['A']['r3']}

# # Agent B (Round 3)
# {responses['B']['r3']}

# Evaluate both agents using the CRIT algorithm:
# 1. Identify the main claims and supporting reasons for each agent
# 2. Assess the quality of reasoning and evidence
# 3. Provide CRIT scores (0.0 to 1.0) for each agent
# 4. Give brief notes on strengths/weaknesses

# Output format:
# CRIT_A: [score] - [brief note]
# CRIT_B: [score] - [brief note]"""

    # try:
    #     messages = [
    #         {"role": "system", "content": "You are a rigorous judge evaluating a debate between two agents on a medical question. Evaluate the debate fairly and provide clear CRIT scores. IMPORTANT: For Probs, always list probabilities in A/B/C/D order as [prob_A, prob_B, prob_C, prob_D]."},
    #         {"role": "user", "content": judge_prompt_r3}
    #     ]
        
    #     # judge_response_r3 = judge.invoke(messages)
    #     print(f"🔍 RAW RESPONSE - Judge Round 3:")
    #     # print(f"   {judge_response_r3.content}")
    #     print(f"   {'='*80}")
        
    #     print(f"✅ Judge Round 3 response received")
    #     # print(f"   Response length: {len(judge_response_r3.content)} characters")
            
    # except Exception as e:
    #     print(f"❌ Judge Round 3 failed: {e}")
    
    # Round 4: Agent A -> Agent B -> Judge
    print("\n=== ROUND 4 ===")
    
    # Agent A Round 4
    print("--- Agent A Round 4 ---")
    debater_prompt_a4 = f"""Round 4: κ = 0.5 (balanced)
Take a more balanced approach. Consider both your position and your opponent's arguments. Look for synthesis and common ground while maintaining your core reasoning.

Agent B's previous analysis: {responses['B']['r3']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_a4}
        ]
        
        response_a4 = agent_a.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent A Round 4:")
        print(f"   {response_a4.content}")
        print(f"   {'='*80}")
        
        responses['A']['r4'] = response_a4.content
        
        # print(f"✅ Agent A Round 4 response received")
        # print(f"   Response length: {len(response_a4.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent A Round 4 failed: {e}")
        return False
    
    # Agent B Round 4
    print("--- Agent B Round 4 ---")
    debater_prompt_b4 = f"""Round 4: κ = 0.5 (balanced)
Take a more balanced approach. Consider both your position and your opponent's arguments. Look for synthesis and common ground while maintaining your core reasoning.

Agent A's previous analysis: {responses['A']['r3']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_b4}
        ]
        
        response_b4 = agent_b.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent B Round 4:")
        print(f"   {response_b4.content}")
        print(f"   {'='*80}")
        
        responses['B']['r4'] = response_b4.content
        
        # print(f"✅ Agent B Round 4 response received")
        # print(f"   Response length: {len(response_b4.content)} characters")
            
    except Exception as e:
        print(f"❌ Agent B Round 4 failed: {e}")
        return False
    
    # Judge Round - COMMENTED OUT 4
    # print("--- Judge Round 4 ---")
    # judge_prompt_r4 = f"""You are a rigorous judge evaluating a debate between two agents on a medical question.

# Round: 4
# Question: {example['question']}
# Choices: {choices_csv}

# # Agent A (Round 4)
# {responses['A']['r4']}

# # Agent B (Round 4)
# {responses['B']['r4']}

# Evaluate both agents using the CRIT algorithm:
# 1. Identify the main claims and supporting reasons for each agent
# 2. Assess the quality of reasoning and evidence
# 3. Provide CRIT scores (0.0 to 1.0) for each agent
# 4. Give brief notes on strengths/weaknesses

# Output format:
# CRIT_A: [score] - [brief note]
# CRIT_B: [score] - [brief note]"""

    # try:
    #     messages = [
    #         {"role": "system", "content": "You are a rigorous judge evaluating a debate between two agents on a medical question. Evaluate the debate fairly and provide clear CRIT scores. IMPORTANT: For Probs, always list probabilities in A/B/C/D order as [prob_A, prob_B, prob_C, prob_D]."},
    #         {"role": "user", "content": judge_prompt_r4}
    #     ]
        
    #     # judge_response_r4 = judge.invoke(messages)
    #     print(f"🔍 RAW RESPONSE - Judge Round 4:")
    #     # print(f"   {judge_response_r4.content}")
    #     print(f"   {'='*80}")
        
    #     print(f"✅ Judge Round 4 response received")
    #     # print(f"   Response length: {len(judge_response_r4.content)} characters")
            
    # except Exception as e:
    #     print(f"❌ Judge Round 4 failed: {e}")
    
    # Round 5: Agent A -> Agent B -> Judge
    print("\n=== ROUND 5 ===")
    
    # Agent A Round 5
    print("--- Agent A Round 5 ---")
    debater_prompt_a5 = f"""Round 5: κ = 0.3 (low contentiousness)
Focus on finding common ground and synthesis. Acknowledge the strongest points from both sides. Work towards a collaborative solution while maintaining your core position.

Agent B's previous analysis: {responses['B']['r4']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_a5}
        ]
        
        response_a5 = agent_a.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent A Round 5:")
        print(f"   {response_a5.content}")
        print(f"   {'='*80}")
        
        responses['A']['r5'] = response_a5.content
        
        # print(f"✅ Agent A Round 5 response received")
        # print(f"   Response length: {len(response_a5.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent A Round 5 failed: {e}")
        return False
    
    # Agent B Round 5
    print("--- Agent B Round 5 ---")
    debater_prompt_b5 = f"""Round 5: κ = 0.3 (low contentiousness)
Focus on finding common ground and synthesis. Acknowledge the strongest points from both sides. Work towards a collaborative solution while maintaining your core position.

Agent A's previous analysis: {responses['A']['r4']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_b5}
        ]
        
        response_b5 = agent_b.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent B Round 5:")
        print(f"   {response_b5.content}")
        print(f"   {'='*80}")
        
        responses['B']['r5'] = response_b5.content
        
        # print(f"✅ Agent B Round 5 response received")
        # print(f"   Response length: {len(response_b5.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent B Round 5 failed: {e}")
        return False
    
    # Judge Round - COMMENTED OUT 5
    # print("--- Judge Round 5 ---")
    # judge_prompt_r5 = f"""You are a rigorous judge evaluating a debate between two agents on a medical question.

# Round: 5
# Question: {example['question']}
# Choices: {choices_csv}

# # Agent A (Round 5)
# {responses['A']['r5']}

# # Agent B (Round 5)
# {responses['B']['r5']}

# Evaluate both agents using the CRIT algorithm:
# 1. Identify the main claims and supporting reasons for each agent
# 2. Assess the quality of reasoning and evidence
# 3. Provide CRIT scores (0.0 to 1.0) for each agent
# 4. Give brief notes on strengths/weaknesses

# Output format:
# CRIT_A: [score] - [brief note]
# CRIT_B: [score] - [brief note]"""

    # try:
    #     messages = [
    #         {"role": "system", "content": "You are a rigorous judge evaluating a debate between two agents on a medical question. Evaluate the debate fairly and provide clear CRIT scores. IMPORTANT: For Probs, always list probabilities in A/B/C/D order as [prob_A, prob_B, prob_C, prob_D]."},
    #         {"role": "user", "content": judge_prompt_r5}
    #     ]
        
    #     # judge_response_r5 = judge.invoke(messages)
    #     print(f"🔍 RAW RESPONSE - Judge Round 5:")
    #     # print(f"   {judge_response_r5.content}")
    #     print(f"   {'='*80}")
        
    #     print(f"✅ Judge Round 5 response received")
    #     # print(f"   Response length: {len(judge_response_r5.content)} characters")
            
    # except Exception as e:
    #     print(f"❌ Judge Round 5 failed: {e}")
    
    # Round 6: Agent A -> Agent B -> Judge (Final Round)
    print("\n=== ROUND 6 (FINAL) ===")
    
    # Agent A Round 6
    print("--- Agent A Round 6 (Final) ---")
    debater_prompt_a6 = f"""Round 6: κ = 0.1 (very low contentiousness - final synthesis)
This is the final round. Synthesize the best arguments from both sides. Provide your final, most confident answer based on the entire debate. Acknowledge the strongest points from your opponent and explain why your position is ultimately correct.

Agent B's previous analysis: {responses['B']['r5']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_a6}
        ]
        
        response_a6 = agent_a.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent A Round 6 (Final):")
        print(f"   {response_a6.content}")
        print(f"   {'='*80}")
        
        responses['A']['r6'] = response_a6.content
        
        # print(f"✅ Agent A Round 6 (Final) response received")
        # print(f"   Response length: {len(response_a6.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent A Round 6 (Final) failed: {e}")
        return False
    
    # Agent B Round 6
    print("--- Agent B Round 6 (Final) ---")
    debater_prompt_b6 = f"""Round 6: κ = 0.1 (very low contentiousness - final synthesis)
This is the final round. Synthesize the best arguments from both sides. Provide your final, most confident answer based on the entire debate. Acknowledge the strongest points from your opponent and explain why your position is ultimately correct.

Agent A's previous analysis: {responses['A']['r5']}

Output format (always):
1. Your Final Answer: A/B/C/D
2. Probs: {{"A": pA, "B": pB, "C": pC, "D": pD}} - probabilities that sum to 1.00. Each probability represents the likelihood that this choice is final answer.
3. Justification: up to 5 sentences citing the key reasons.

Question: {example['question']}
Choices: {choices_csv}"""

    try:
        messages = [
            {"role": "system", "content": "You are debating another agent on a 4-choice question (A/B/C/D). \"Contentiousness\" κ controls how strongly you challenge the opponent: κ=0.9 is highly adversarial; κ=0.1 is mostly consolidating. Follow the exact output format specified. IMPORTANT: Always provide Final Answer first, then Probs in JSON format {{\"A\": pA, \"B\": pB, \"C\": pC, \"D\": pD}} where probabilities sum to 1.00."},
            {"role": "user", "content": debater_prompt_b6}
        ]
        
        response_b6 = agent_b.invoke(messages)
        print(f"🔍 RAW RESPONSE - Agent B Round 6 (Final):")
        print(f"   {response_b6.content}")
        print(f"   {'='*80}")
        
        responses['B']['r6'] = response_b6.content
        
        # print(f"✅ Agent B Round 6 (Final) response received")
        # print(f"   Response length: {len(response_b6.content)} characters")
        
    except Exception as e:
        print(f"❌ Agent B Round 6 (Final) failed: {e}")
        return False
    
    # Final Judge Round 6
    # print("--- Final Judge Round 6 ---")
    # judge_prompt_r6 = f"""You are a rigorous judge evaluating the final round of a debate between two agents on a medical question.

# Final Round: 6
# Question: {example['question']}
# Choices: {choices_csv}

# # Agent A (Final Round 6)
# {responses['A']['r6']}

# # Agent B (Final Round 6)
# {responses['B']['r6']}

# This is the final evaluation. Consider the entire debate progression and provide:
# 1. Final CRIT scores (0.0 to 1.0) for each agent
# 2. Overall debate winner
# 3. Summary of key arguments and reasoning quality
# 4. Final recommendation

# Output format:
# CRIT_A: [final_score] - [comprehensive note]
# CRIT_B: [final_score] - [comprehensive note]
# WINNER: [Agent A/Agent B/Tie] - [brief explanation]
# SUMMARY: [2-3 sentences on overall debate quality]"""

    # try:
    #     messages = [
    #         # {"role": "system", "content": "You are a rigorous judge. Evaluate the final debate round comprehensively and declare a winner."},
    #         # {"role": "user", "content": judge_prompt_r6}
    #     ]
        
        # judge_response_r6 = judge.invoke(messages)
        # print(f"🔍 RAW RESPONSE - Final Judge Round 6:")
        # print(f"   {judge_response_r6.content}")
        # print(f"   {'='*80}")
        
        # print(f"✅ Final Judge Round 6 response received")
        # print(f"   Response length: {len(judge_response_r6.content)} characters")
            
    # except Exception as e:
    #     print(f"❌ Final Judge Round 6 failed: {e}")
    
    print(f"\n=== MMLU Professional Medicine Test Results ===")
    print(f"✅ Test completed successfully!")
    print(f"   Agent A responses: {list(responses['A'].keys())}")
    print(f"   Agent B responses: {list(responses['B'].keys())}")
    print(f"   Total rounds completed: 6")
    print(f"   Contentiousness progression: 0.9 → 0.9 → 0.7 → 0.5 → 0.3 → 0.1")
    
    # Clean up output redirection
    sys.stdout = original_stdout
    tee_output.log.close()
    print(f"📁 Output saved to: {filename}")
    
    return True

if __name__ == "__main__":
    success = test_mmlu_pro_med()
    if success:
        print("🎉 MMLU Professional Medicine test passed!")
    else:
        print("⚠️ MMLU Professional Medicine test failed!")
