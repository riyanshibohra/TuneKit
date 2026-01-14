"""
Model Recommendation Tool
=========================
Simple 3-factor scoring: Task (50) + Size (30) + Output (20) = 100 points
"""

import os
import json
from typing import Dict, List

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.dirname(os.path.abspath(__file__)).replace('tools', 'data')
CONFIG_PATH = os.path.join(DATA_DIR, 'models.json')

def load_config() -> dict:
    """Load configuration from JSON file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback empty config if file is missing (should not happen in prod)
        return {"models": {}, "defaults": {}, "tiebreaker": {}}

CONFIG = load_config()

# Helper to access metadata easily
MODELS = {k: v['metadata'] for k, v in CONFIG['models'].items()}
DEFAULTS = CONFIG.get('defaults', {
    "task_score": 25,
    "size_score": 15,
    "output_score": 10,
    "multi_turn_bonus": 0
})
TIEBREAKER = CONFIG.get('tiebreaker', {})

# ════════════════════════════════════════════════════════════════════════════
# MAIN RECOMMENDATION FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def recommend_model(
    user_task: str,
    conversation_characteristics: Dict,
    num_examples: int,
    deployment_target: str = 'not_sure'
) -> Dict:
    """
    Recommend best SLM model using simple 3-factor scoring.
    
    Args:
        user_task: 'classify', 'qa', 'conversation', 'generation', 'extraction'
        conversation_characteristics: Dict with is_multilingual, avg_response_length, looks_like_json_output
        num_examples: Dataset size
        deployment_target: 'cloud_api', 'mobile_app', 'edge_device', etc.
    
    Returns:
        Dict with primary_recommendation, alternatives, all_scores
    """
    is_multilingual = conversation_characteristics.get('is_multilingual', False)
    avg_response_length = conversation_characteristics.get('avg_response_length', 50)
    looks_like_json = conversation_characteristics.get('looks_like_json_output', False)
    is_multi_turn = conversation_characteristics.get('is_multi_turn', False)
    
    # Hard overrides for obvious cases
    if is_multilingual:
        return build_response(
            primary_key='qwen-2.5-3b',
            score=100,
            all_scores={'qwen-2.5-3b': 100},
            reasons=['Best multilingual support (29 languages)', 'Optimized for non-English text'],
            num_examples=num_examples
        )
    
    if deployment_target == 'edge_device':
        return build_response(
            primary_key='gemma-3-270m',
            score=100,
            all_scores={'gemma-3-270m': 100, 'gemma-3-2b': 90, 'phi-4-mini': 80},
            reasons=['Smallest model (270M params)', 'Optimized for low-power devices'],
            num_examples=num_examples,
            alternatives=[{'model': 'gemma-3-2b', 'score': 90}]
        )
    
    # Score each model based on JSON config
    scores = {}
    
    for model_key, model_data in CONFIG['models'].items():
        # 1. Filter by deployment target
        allowed_deployments = model_data.get('deployment', [])
        if deployment_target != 'not_sure' and deployment_target not in allowed_deployments:
            continue
            
        model_scores = model_data.get('scores', {})
        
        # Factor 1: Task type
        task_scores_map = model_scores.get('task', {})
        task_score = task_scores_map.get(user_task, DEFAULTS['task_score'])
        
        # Factor 2: Dataset size
        size_scores_map = model_scores.get('size', {})
        if num_examples < 500:
            size_score = size_scores_map.get('small', DEFAULTS['size_score'])
        elif num_examples >= 2000:
            size_score = size_scores_map.get('large', DEFAULTS['size_score'])
        else:
            size_score = size_scores_map.get('medium', DEFAULTS['size_score'])
        
        # Factor 3: Output characteristics
        output_scores_map = model_scores.get('output', {})
        if avg_response_length > 200:
            output_score = output_scores_map.get('long', DEFAULTS['output_score'])
        elif looks_like_json:
            output_score = output_scores_map.get('json', DEFAULTS['output_score'])
        else:
            output_score = DEFAULTS['output_score']
        
        # Bonus: Multi-turn
        multi_turn_bonus = model_scores.get('multi_turn_bonus', DEFAULTS['multi_turn_bonus']) if is_multi_turn else 0
        
        scores[model_key] = task_score + size_score + output_score + multi_turn_bonus
    
    # Pick winner with tie-breaking
    priority_order = TIEBREAKER.get(user_task, TIEBREAKER.get('default', []))
    
    def get_priority(model_key):
        try:
            return priority_order.index(model_key)
        except ValueError:
            return 99
    
    # Sort by score DESC, then by priority ASC
    sorted_models = sorted(scores.items(), key=lambda x: (-x[1], get_priority(x[0])))
    
    if not sorted_models:
        # Fallback if no models match filters
        return build_response(
            primary_key='phi-4-mini', # Fallback safe default
            score=50,
            all_scores={'phi-4-mini': 50},
            reasons=['Fallback recommendation (no models matched specific criteria)'],
            num_examples=num_examples
        )

    primary_key = sorted_models[0][0]
    primary_score = sorted_models[0][1]
    
    reasons = generate_reasons(primary_key, user_task, num_examples, avg_response_length, looks_like_json, is_multi_turn, deployment_target)
    
    # Build alternatives
    alternatives = []
    for model_key, score in sorted_models[1:3]:
        model_meta = MODELS[model_key]
        alt_reasons = []
        
        if score >= primary_score - 10:
            alt_reasons.append(f"Close match ({score}/100)")
        
        # Check gated status (simplified logic based on ID pattern)
        if "meta-llama" in model_meta['id'] or "gemma" in model_meta['id']:
             alt_reasons.append("Requires HuggingFace approval (gated)")
        
        primary_meta = MODELS[primary_key]
        if model_meta.get('training_time_base', 0) < primary_meta.get('training_time_base', 0):
            time_diff = primary_meta['training_time_base'] - model_meta['training_time_base']
            alt_reasons.append(f"~{time_diff} min faster training")
        
        if model_meta.get('memory_gb', 0) < primary_meta.get('memory_gb', 0):
            alt_reasons.append(f"Lower VRAM ({model_meta['memory_gb']}GB vs {primary_meta['memory_gb']}GB)")
        
        if model_meta.get('context_window', 0) > primary_meta.get('context_window', 0):
            alt_reasons.append(f"Larger context ({model_meta['context_window']//1000}K tokens)")
        
        alternatives.append({
            'model': model_key,
            'score': score,
            'reasons': alt_reasons if alt_reasons else ['Good alternative']
        })
    
    return build_response(
        primary_key=primary_key,
        score=primary_score,
        all_scores=scores,
        reasons=reasons,
        num_examples=num_examples,
        alternatives=alternatives
    )


def generate_reasons(model_key: str, task: str, num_examples: int, avg_response: int, is_json: bool, is_multi_turn: bool, deployment: str) -> List[str]:
    """Generate human-readable reasons for the recommendation."""
    reasons = []
    model_data = CONFIG['models'].get(model_key, {})
    model_meta = model_data.get('metadata', {})
    reasons_config = model_data.get('reasons', {})
    
    # Task-based reason
    if task in reasons_config:
        reasons.append(reasons_config[task])
    else:
        reasons.append(f"Strong performance for {task} tasks")
    
    # Generic features
    reasons.extend(reasons_config.get('features', [])[:2])
    
    # Context window logic
    context_window = model_meta.get('context_window', 0)
    if context_window >= 128000:
        reasons.append('128K token context window')
    elif context_window >= 32000:
        reasons.append('32K token context window')
    
    if is_multi_turn and model_data.get('scores', {}).get('multi_turn_bonus', 0) > 5:
        reasons.append('Excellent at multi-turn context tracking')
    
    # Size-based reason
    if num_examples < 500:
        if model_key == 'gemma-3-2b':
            reasons.append('Ideal for small datasets')
        elif model_key == 'gemma-3-270m':
            reasons.append('Perfect for tiny datasets')
        elif model_key == 'phi-4-mini':
            reasons.append('Works well with limited data')
    elif num_examples > 2000:
        if model_key in ['llama-3.2-3b', 'mistral-7b']:
            reasons.append(f'Scales well with {num_examples:,} examples')
    
    # Output-based reason
    if avg_response > 200 and model_key in ['mistral-7b', 'llama-3.2-3b']:
        reasons.append('Optimized for longer outputs')
    elif is_json and model_key == 'phi-4-mini':
        reasons.append('Excellent JSON/structured output')
    
    # Deployment reason
    if deployment in ['mobile_app', 'ios_app', 'android_app'] and model_key in ['phi-4-mini', 'gemma-3-2b', 'gemma-3-270m']:
        reasons.append('Optimized for mobile deployment')
    elif deployment == 'web_browser' and model_key in ['gemma-3-2b', 'phi-4-mini', 'gemma-3-270m']:
        reasons.append('Runs efficiently in browser')
    elif deployment == 'edge_device' and model_key in ['gemma-3-2b', 'gemma-3-270m']:
        reasons.append('Designed for edge devices')
    
    # Deduplicate and limit
    unique_reasons = []
    seen = set()
    for r in reasons:
        if r not in seen:
            unique_reasons.append(r)
            seen.add(r)
            
    return unique_reasons[:5]


def build_response(
    primary_key: str,
    score: int,
    all_scores: Dict,
    reasons: List[str],
    num_examples: int,
    alternatives: List[Dict] = None
) -> Dict:
    """Build the final response with model recommendation."""
    model = MODELS[primary_key]
    confidence = 'high' if score >= 80 else ('medium' if score >= 60 else 'low')
    
    def format_context_window(tokens):
        if tokens >= 1000:
            return f"{tokens // 1000}K"
        return str(tokens)
    
    formatted_alternatives = []
    if alternatives:
        for alt in alternatives:
            alt_model = MODELS.get(alt.get('model'))
            if alt_model:
                ctx_window = alt_model.get('context_window', 0)
                formatted_alternatives.append({
                    'model_id': alt_model['id'],
                    'model_name': alt_model['name'],
                    'size': alt_model['size'],
                    'score': round(alt.get('score', 0) / 100, 2),
                    'reasons': alt.get('reasons', ['Good alternative']),
                    'context_window': ctx_window,
                    'context_window_formatted': format_context_window(ctx_window)
                })
    
    return {
        'primary_recommendation': {
            'model_id': model['id'],
            'model_name': model['name'],
            'size': model['size'],
            'score': round(score / 100, 2),
            'confidence': confidence,
            'reasons': reasons,
            'context_window': model.get('context_window', 0),
            'context_window_formatted': format_context_window(model.get('context_window', 0))
        },
        'alternatives': formatted_alternatives,
        'all_scores': {k: round(v / 100, 2) for k, v in all_scores.items()}
    }
