import json
import requests
from typing import Dict, List

def fix_ppo_prompt(prompt: str):
    prompt = prompt.strip()
    prompt = prompt.removeprefix("""system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
user""")
    prompt = prompt.removesuffix("""First, create a detailed plan on what you'd write. The plan MUST be enclosed within <think> </think> tags. After that, write the story.
assistant""")
    prompt = prompt.removesuffix("""assistant""")
    
    return prompt.strip()

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
            'prompt':fix_ppo_prompt(prompt),
            'response':response
        })
    payload = {"model": "model", "messages": messages}
    response = requests.post("https://7d9a-46-10-148-86.ngrok-free.app/get_rewards", json=payload, headers=headers)
    response = json.loads(response.text)
    
    for entry in response:
        entry['overall'] = entry['combined_score']
        entry.pop('combined_score', None)
        
    return response