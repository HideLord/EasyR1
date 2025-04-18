import time
import json
import random
import requests
from typing import List, Dict

def compute_score(
    prompts: List[str],
    responses: List[str], 
    ground_truths: List[str], 
    format_weight: float = 0.1,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> List[Dict[str, float]]:
    headers = {"Content-Type": "application/json", "X-API-Key": 'dummy_api_key'}
    messages = []
    for prompt, response in zip(prompts, responses):
        messages.append({
            'prompt': prompt,
            'response': response
        })
    payload = {"model": "model", "messages": messages}
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                "https://390e-46-10-148-86.ngrok-free.app/get_rewards", 
                json=payload, 
                headers=headers
            )
            
            response.raise_for_status()
            
            parsed_response = json.loads(response.text)
            
            for entry in parsed_response:
                entry['overall'] = entry['combined_score']
                entry.pop('combined_score', None)
                
            return parsed_response
            
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries:
                print(f"All {max_retries + 1} attempts failed. Returning default scores.")
                result = []
                for r in responses:
                    result.append({
                        'overall': 0,
                        'coherence_score': 0,
                        'compliance_score': 0,
                        'engagement_score': 0,
                        'technical_score': 0,
                    })
                return result
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 0.1 * delay)
            total_delay = delay + jitter
            
            print(f"Retrying in {total_delay:.2f} seconds...")
            time.sleep(total_delay)
    
    result = []
    for r in responses:
        result.append({
            'overall': 0,
            'coherence_score': 0,
            'compliance_score': 0,
            'engagement_score': 0,
            'technical_score': 0,
        })
    return result