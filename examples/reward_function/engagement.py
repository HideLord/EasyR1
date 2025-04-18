import re
from typing import Dict, List

def compute_score(
    prompts: List[str],
    responses: List[str], 
    ground_truths: List[str]
) -> List[Dict[str, float]]:
    METRICS = [
        'Show, Don\'t Tell',
        'Hook & Story Drive',
        'Relatable Characters',
        'Plot',
        'Emotional Resonance',
    ]
    def parse_deductions(text):
        lines = text.split('\n')
        start_idx = -1
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if line.strip() == '[Scoring]':
                start_idx = i
            elif start_idx != -1 and line.startswith('[') and line.endswith(']'):
                end_idx = i
                break
        
        if start_idx == -1:
            return None
        
        scoring_lines = lines[start_idx+1:end_idx]
        
        categories = {m:None for m in METRICS}
        
        for line in scoring_lines:
            line = line.strip()
            if not line or not line.startswith('-'):
                continue
                
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
                
            category = parts[0].strip('- ')
            deduction_part = parts[1].strip()
            
            if deduction_part == '0':
                deduction = 0
            elif deduction_part.startswith('-'):
                try:
                    deduction = int(deduction_part)
                except ValueError:
                    continue
            else:
                continue
                
            if category in categories:
                categories[category] = deduction
        
        total = 0
        for val in categories.values():
            if val is None:
                return None
            total += val
        
        categories['Total'] = total
        return categories
    
    def get_rewards(expected, predicted):
        rewards = {}
        overall_reward = 0
        
        for metric in METRICS:
            expected_deduction = expected[metric]
            predicted_deduction = predicted[metric]
            
            error = (expected_deduction - predicted_deduction) ** 2
            reward = 100-error
            rewards[metric] = reward
            overall_reward += reward
            
        rewards['overall'] = overall_reward
        return rewards
    
    def get_empty_rewards():
        rewards = {}
        for metric in METRICS:
            rewards[metric] = -1.0
            
        rewards['overall'] = -5.0
        
        return rewards
    
    results = []
    for predicted, expected in zip(responses, ground_truths):
        deductions = parse_deductions(predicted)
        
        if deductions is not None:
            results.append(get_rewards(expected, deductions))
        else:
            results.append(get_empty_rewards())
        
    return results