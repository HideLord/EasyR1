import json
import requests
from typing import Dict, List

def compute_score(
    prompts: List[str],
    responses: List[str], 
    ground_truths: List[str], 
    format_weight: float = 0.1
) -> List[Dict[str, float]]:
    headers = {"Content-Type": "application/json", "X-API-Key": 'dummy_api_key'}
    messages = []
    for prompt, response in zip(prompts, responses):
        messages.append({
            'prompt':prompt,
            'response':response
        })
    payload = {"model": "model", "messages": messages}
    response = requests.post("https://5566-46-10-148-86.ngrok-free.app/get_rewards", json=payload, headers=headers)
    response = json.loads(response.text)
    
    combined_scores = response["scores"]
    coherence_scores = response["coherence_scores"]
    reward_scores = response["reward_scores"]
    
    results = []
    for combined,coherence,reward in zip(combined_scores, coherence_scores, reward_scores):
        results.append({
            "overall": combined,
            "coherence": coherence,
            "reward": reward
        })
        
    return results