#!/usr/bin/env python3
"""
Test script for CommonSenseQA dataset
"""
import json
from datasets import load_dataset
from src.debate.models import LLMFactory
from src.debate.prompts import parse_json_or_fallback, parse_judge_json

def test_commonsenseqa():
    """Test CommonSenseQA dataset"""
    
    print("=== CommonSenseQA Test ===")
    
    # Load dataset
    dataset = load_dataset('commonsense_qa', split='validation')
    example = dataset[0]
    
    print(f"Question: {example['question']}")
    print(f"Choices: {example['choices']}")
    print(f"Answer: {example['answerKey']}")
    print(f"Question length: {len(example['question'])} characters")
    
    # Convert to our format
    choices_dict = {}
    text_list = example['choices']['text']
    label_list = example['choices']['label']
    for i, label in enumerate(label_list):
        if i < len(text_list):
            choices_dict[label] = text_list[i]
    
    choices_csv = ", ".join([f"{k}: {v}" for k, v in choices_dict.items()])
    
    # Test debater
    print("\n--- Testing Debater ---")
    debater = LLMFactory.make("local", "Qwen/Qwen2.5-7B-Instruct", temperature=0.7, max_tokens=1024)
    
    debater_prompt = f"""Round 1. You are presented with the following multiple-choice question, and provide your own analysis of output and reasoning:

Question: {example['question']}
Choices: {choices_csv}

Output (strict JSON):
{{"output": {{"A": pA, "B": pB, "C": pC, "D": pD, "E": pE}}, "reason": {{"A": rA, "B": rB, "C": rC, "D": rD, "E": rE}}}}"""

    try:
        messages = [
            {"role": "system", "content": "You are a careful multiple-choice reasoner. Always answer in STRICT JSON and nothing else."},
            {"role": "user", "content": debater_prompt}
        ]
        
        response = debater.invoke(messages)
        parsed = parse_json_or_fallback(response.content, list(choices_dict.keys()))
        
        print(f"✅ Debater response parsed successfully")
        print(f"   Probabilities: {parsed['probs']}")
        print(f"   Top choice: {max(parsed['probs'], key=parsed['probs'].get)}")
        
        debater_success = True
    except Exception as e:
        print(f"❌ Debater failed: {e}")
        debater_success = False
    
    # Test judge
    print("\n--- Testing Judge ---")
    judge = LLMFactory.make("local", "Qwen/Qwen2.5-7B-Instruct", temperature=0.2, max_tokens=2048)
    
    judge_prompt = f"""Round: 1
Question: {example['question']}
Choices: {choices_csv}

# Agent A (Round 1)
outputA: {{"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}}
reasonA: {{"A": "Equal probability", "B": "Equal probability", "C": "Equal probability", "D": "Equal probability", "E": "Equal probability"}}

# Agent B (Round 1)
outputB: {{"A": 0.1, "B": 0.2, "C": 0.4, "D": 0.2, "E": 0.1}}
reasonB: {{"A": "Less likely", "B": "Somewhat likely", "C": "Most likely", "D": "Somewhat likely", "E": "Less likely"}}

CRIT: "Function Γ = CRIT(d)
Input: document d   Output: validation score Γ
Vars: Ω claim; R and R′ sets of reasons and rival reasons
Subs: CLAIM(), FINDDOC(), VALIDATE()
Begin
#1–#2 Identify in d the claim Ω. Find a set of supporting reasons R for Ω.
#3 For each r ∈ R evaluate r ⇒ Ω.
   If CLAIM(r) then (γ_r, θ_r) = CRIT(FINDDOC(r)).
   Else (γ_r, θ_r) = VALIDATE(r ⇒ Ω).
#4–#6–#7–#8 Find a set of rival reasons R′ against Ω.
   #5 For each r′ ∈ R′ compute (γ_r′, θ_r′) = VALIDATE(r′ ⇒ Ω).
   Compute a weighted sum Γ from the validation scores.
   Analyze arguments to justify the final Γ score.
   Reflect on transfer of CRIT to other contexts.
End"

Output STRICT JSON only:
{{
  "outputA": {{"A": pA, "B": pB, "C": pC, "D": pD, "E": pE}},
  "outputB": {{"A": pA, "B": pB, "C": pC, "D": pD, "E": pE}},
  "CRIT_A": float,
  "CRIT_B": float,
  "NOTE_A": "string",
  "NOTE_B": "string"
}}"""

    try:
        messages = [
            {"role": "system", "content": "You are a rigorous, deterministic judge. Apply the CRIT algorithm directly and output STRICT JSON only."},
            {"role": "user", "content": judge_prompt}
        ]
        
        response = judge.invoke(messages)
        parsed = parse_judge_json(response.content, list(choices_dict.keys()))
        
        if parsed["CRIT_A"] is not None and parsed["CRIT_B"] is not None:
            print(f"✅ Judge response parsed successfully")
            print(f"   CRIT_A: {parsed['CRIT_A']}")
            print(f"   CRIT_B: {parsed['CRIT_B']}")
            judge_success = True
        else:
            print(f"❌ Judge returned null CRIT scores")
            judge_success = False
            
    except Exception as e:
        print(f"❌ Judge failed: {e}")
        judge_success = False
    
    print(f"\n=== CommonSenseQA Test Results ===")
    print(f"Debater: {'✅ PASS' if debater_success else '❌ FAIL'}")
    print(f"Judge: {'✅ PASS' if judge_success else '❌ FAIL'}")
    
    return debater_success and judge_success

if __name__ == "__main__":
    success = test_commonsenseqa()
    if success:
        print("🎉 CommonSenseQA test passed!")
    else:
        print("⚠️ CommonSenseQA test failed!")
