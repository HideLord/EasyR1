import re
import math
from typing import Dict, List

_step_counter = 0

def _extract_score(text):
    pattern = r'(?<![\d\.])((?:[0-9]|10)(?:\.\d+)?)\s*/\s*10' # e.g. Score: 9.1/10
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def _get_reward_for_format(text):
    pattern = r'Score: (10|[0-9])/10$'
    matches = re.findall(pattern, text.strip(), re.DOTALL)
    return 0.1 if len(matches) == 1 else 0.0

def _convert_to_reward(expected_score, predicted_score):
    distance = abs(expected_score - predicted_score)
    
    if math.isclose(distance, 0):
        return 2
    elif distance <= 1:
        return 2-distance
    else:
        if _step_counter > 50:
            return 0
        
        max_distance = max(10-predicted_score, predicted_score-0)
        max_reward = max_distance-2
        
        reward = (max_distance-distance)
        # max normalized reward would be max_reward/(max_reward*5) = 1/5 = 0.2
        return reward/(max_reward*5)

def compute_score(
    prompts: List[str],
    responses: List[str], 
    ground_truths: List[str]
) -> List[Dict[str, float]]:
    global _step_counter
    results = []
    
    for predicted, expected in zip(responses, ground_truths):
        expected_score = float(expected) if not isinstance(expected, str) else _extract_score(expected)
        predicted_score = _extract_score(predicted)
    
        result = {}
            
        if expected_score is not None and predicted_score is not None:
            score_reward = _convert_to_reward(expected_score, predicted_score)
            format_reward = _get_reward_for_format(predicted)
            
            result["overall"] = score_reward + format_reward
            result["score"] = score_reward
            result["format"] = format_reward
            results.append(result)
        else:
            result["overall"] = -1.0
            result["format"] = 0
            result["score"] = 0
            results.append(result)
            
    _step_counter += 1
        
    return results