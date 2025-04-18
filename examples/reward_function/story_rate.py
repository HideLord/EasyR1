import re
from typing import Dict, List

def compute_score(
    prompts: List[str],
    responses: List[str], 
    ground_truths: List[str]
) -> List[Dict[str, float]]:
    METRICS = [
        "Adherence to Instructions",
        "Believable Character Actions",
        "Nuanced Characters",
        "Consistent Voice/Tone of Writing",
        "Imagery and Descriptive Quality",
        "Elegant Prose",
        "Emotionally Engaging",
        "Emotionally Complex",
        "Coherent",
        "Meandering",
        "Weak Dialogue",
        "Tell-Don't-Show",
        "Unsurprising or Uncreative",
        "Amateurish",
        "Purple Prose",
        "Overwrought",
        "Incongruent Ending Positivity",
        "Unearned Transformations",
        "Well-earned Lightness or Darkness",
        "Sentences Flow Naturally",
        "Overall Reader Engagement",
        "Overall Impression"
    ]
    
    def get_rewards(predicted, ground):
        all_rewards_and_errors = {}
        
        total_error = 0
        count = 0
        
        for metric in METRICS:
            if metric in predicted and metric in ground:
                error = abs(predicted[metric] - ground[metric]) / 20.0
                total_error += error
                count += 1
                
                all_rewards_and_errors[metric] = error
            elif metric in ground:
                total_error += 1.0
                count += 1
                all_rewards_and_errors[metric] = 1.0
            else:
                all_rewards_and_errors[metric] = 0.0
        
        if count == 0:
            all_rewards_and_errors['overall'] = 0.0
            return all_rewards_and_errors
        
        avg_error = total_error / count
        reward = 1.0 - avg_error
        
        all_rewards_and_errors['overall'] = reward
        return all_rewards_and_errors
    
    def get_empty_rewards():
        all_rewards_and_errors = {}
        for metric in METRICS:
            all_rewards_and_errors[metric] = 0.0
            
        all_rewards_and_errors['overall'] = 0.0
        
        return all_rewards_and_errors

    def verify_analysis_format(text):
        if not text or not isinstance(text, str):
            return False, "Input must be a non-empty string", text
        
        if not re.search(r'\[Analysis\]', text):
            return False, "Missing [Analysis] section", text
        
        if not re.search(r'\[Scores\]', text):
            return False, "Missing [Scores] section", text
        
        analysis_pos = text.find('[Analysis]')
        scores_pos = text.find('[Scores]')
        if analysis_pos > scores_pos:
            return False, "[Analysis] section should come before [Scores] section", text
        
        scores_section = text[scores_pos:]
        
        score_pattern = re.compile(r'([^:\n]+):\s*(\d+)')
        scores = score_pattern.findall(scores_section)
        
        scores_dict = {}
        invalid_scores = []
        
        for metric, score in scores:
            metric = metric.strip()
            try:
                score_value = int(score)
                if score_value < 0 or score_value > 20:
                    invalid_scores.append(f"{metric} (score: {score})")
                else:
                    scores_dict[metric] = score_value
            except ValueError:
                invalid_scores.append(f"{metric} (score: {score})")
        
        if invalid_scores:
            return False, f"Invalid scores (must be 0-20) for: {', '.join(invalid_scores)}", text
        
        return True, scores_dict, text
    
    results = []
    for predicted, expected in zip(responses, ground_truths):
        is_valid_expected, expected_scores, _ = verify_analysis_format(expected)
        is_valid, scores, _ = verify_analysis_format(predicted)
        
        if is_valid and is_valid_expected:
            results.append(get_rewards(expected_scores, scores))
        else:
            results.append(get_empty_rewards())
        
    return results