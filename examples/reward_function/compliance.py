import re
from typing import Dict, List

def extract_score(text):
    pattern = r'(?<![\d\.])((?:[1-9]|10)(?:\.\d+)?)\s*/\s*10' # e.g. Score: 9.1/10
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def convert_to_reward(expected_score, predicted_score):
    max_error = (10-1)**2
    return max_error - (expected_score - predicted_score)**2

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
            result["overall"] = convert_to_reward(expected_score, predicted_score) 
            result["format"] = 1
            results.append(result)
        else:
            result["overall"] = -1.0
            result["format"] = 0
            results.append(result)
        
    return results