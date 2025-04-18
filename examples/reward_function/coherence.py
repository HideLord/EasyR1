import re
from typing import Dict, List

def extract_score(text):
    pattern = r'(?<![\d\.])((?:[0-9]|10)(?:\.\d+)?)\s*/\s*10' # e.g. Score: 9.1/10
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def get_extra_reward_for_format(text):
    pattern = r'Score: (10|[0-9])/10$'
    return bool(re.search(pattern, text.strip(), re.DOTALL))

def convert_to_reward(expected_score, predicted_score):
    error = abs(expected_score - predicted_score)
    
    if error <= 2:
        penalty_factor = 0.5 * error
    else:
        penalty_factor = 1 + (error-2) # 1 for error up to 2 and linear error afterwards.
        
    return 9-penalty_factor

def compute_score(
    prompts: List[str],
    responses: List[str], 
    ground_truths: List[str]
) -> List[Dict[str, float]]:
    results = []
    
    for predicted, expected in zip(responses, ground_truths):
        expected_score = float(expected) if not isinstance(expected, str) else extract_score(expected)
        predicted_score = extract_score(predicted)
    
        result = {}
            
        if expected_score is not None and predicted_score is not None:
            result["overall"] = convert_to_reward(expected_score, predicted_score) + get_extra_reward_for_format(predicted)
            result["format"] = get_extra_reward_for_format(predicted)
            results.append(result)
        else:
            result["overall"] = -1.0
            result["format"] = 0
            results.append(result)
        
    return results