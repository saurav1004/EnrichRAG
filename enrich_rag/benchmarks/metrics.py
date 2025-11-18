# This code is adapted from flashrag/evaluator/metrics.py 
from collections import Counter
from .utils import normalize_answer # Import our local copy

def f1_score(prediction, ground_truths):
    """Calculate F1 score."""
    normalized_prediction = normalize_answer(prediction)
    pred_tokens = normalized_prediction.split()
    
    f1_scores = []
    for gt in ground_truths:
        normalized_ground_truth = normalize_answer(gt)
        gt_tokens = normalized_ground_truth.split()
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1_scores.append(0.0)
            continue
            
        precision = 1.0 * num_same / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
        recall = 1.0 * num_same / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
    return max(f1_scores)

def exact_match(prediction, ground_truths):
    """Calculate Exact Match."""
    norm_pred = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == norm_pred:
            return 1.0
    return 0.0

def evaluate(predictions, gold_data, metrics_to_run):
    """Main evaluation function."""
    scores = {}
    if "em" in metrics_to_run:
        scores["em"] = 0.0
    if "f1" in metrics_to_run:
        scores["f1"] = 0.0
    
    gold_lookup = {item['id']: item['answers'] for item in gold_data}
    
    for pred_item in predictions:
        pred_id = pred_item['id']
        if pred_id not in gold_lookup:
            print(f"Warning: Prediction ID {pred_id} not in gold data.")
            continue
            
        pred_answer = pred_item['answer']
        gold_answers = gold_lookup[pred_id]
        
        if "em" in metrics_to_run:
            scores["em"] += exact_match(pred_answer, gold_answers)
        if "f1" in metrics_to_run:
            scores["f1"] += f1_score(pred_answer, gold_answers)
            
    total = len(predictions)
    if total == 0:
        return scores # Avoid division by zero
        
    if "em" in metrics_to_run:
        scores["em"] = (scores["em"] / total) * 100
    if "f1" in metrics_to_run:
        scores["f1"] = (scores["f1"] / total) * 100
        
    return scores