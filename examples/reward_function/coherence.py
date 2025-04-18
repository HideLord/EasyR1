import re
import math
from typing import Dict, List

def _extract_score(text):
    pattern = r'(?<![\d\.])((?:[0-9]|10)(?:\.\d+)?)\s*/\s*10' # e.g. Score: 9.1/10
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def _get_reward_for_format(text):
    pattern = r'Score: (10|[0-9])/10$'
    matches = re.findall(pattern, text.strip(), re.DOTALL)
    return 0.1 if len(matches) == 1 else 0.0

def _convert_to_reward(expected_score, predicted_score, sigma=1.7):
    error = abs(expected_score - predicted_score)
    return math.exp(-(error ** 2) / (2 * sigma ** 2))

def compute_score(
    prompts: List[str],
    responses: List[str], 
    ground_truths: List[str]
) -> List[Dict[str, float]]:
    results = []
    
    for predicted, expected in zip(responses, ground_truths):
        expected_score = float(expected) if not isinstance(expected, str) else _extract_score(expected)
        predicted_score = _extract_score(predicted)
    
        result = {i:0 for i in range(11)}
            
        if expected_score is not None and predicted_score is not None:
            dense_score_reward = _convert_to_reward(expected_score, predicted_score)
            format_reward = _get_reward_for_format(predicted)
            
            result["overall"] = dense_score_reward + format_reward
            result["dense_score"] = dense_score_reward
            result["format"] = format_reward
            result[int(predicted_score)]=1
            results.append(result)
        else:
            result["overall"] = -1.0
            result["format"] = 0
            result["dense_score"] = 0
            results.append(result)
        
    return results