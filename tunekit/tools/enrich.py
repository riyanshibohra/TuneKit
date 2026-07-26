"""
Enrich Data Tool
================
Utilities for improving fine-tuning examples:
1. Quality Metrics (Complexity, Diversity, etc.)
2. Prioritization (Scoring examples)
3. Augmentation (Class balancing, Synthetic generation placeholders)
"""

from typing import TYPE_CHECKING, List, Dict, Any, Tuple
import random
import re
import math
import json
from collections import Counter

if TYPE_CHECKING:
    from tunekit.state import TuneKitState

# ============================================================================
# METRICS
# ============================================================================

def _calculate_lexical_diversity(text: str) -> float:
    """Calculate Type-Token Ratio (TTR) as a measure of lexical diversity."""
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def _calculate_instruction_complexity(text: str) -> float:
    """
    Estimate instruction complexity based on length and structure.
    Returns 0.0 to 1.0
    """
    words = text.split()
    word_count = len(words)
    
    # Heuristic: Instructions between 10 and 50 words are often "good" complexity
    # Too short = simple command
    # Too long = might be confusing or few-shot
    
    score = 0.0
    if word_count < 5:
        score = 0.2
    elif word_count < 10:
        score = 0.5
    elif word_count < 50:
        score = 0.9
    else:
        score = 0.7 # Still good, but maybe verbose
        
    # Bonus for punctuation indicating structure
    if "?" in text: score += 0.1
    if "\n" in text: score += 0.1 # Lists or formatting
    
    return min(1.0, score)

def _score_conversation_quality(messages: List[Dict]) -> float:
    """
    Score a conversation based on multiple heuristics.
    Returns 0.0 to 1.0
    """
    if not messages:
        return 0.0
        
    score = 0.5 # Base score
    
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    asst_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    
    if not user_msgs or not asst_msgs:
        return 0.0
    
    # 1. Balance check (User vs Assistant length)
    avg_user_len = sum(len(m) for m in user_msgs) / len(user_msgs)
    avg_asst_len = sum(len(m) for m in asst_msgs) / len(asst_msgs)
    
    # Assistant should generally be helpful (not too short) but not hallucinatingly long
    if avg_asst_len < 10:
        score -= 0.2
    elif avg_asst_len > 2000:
        score -= 0.1 # Very long might be okay, but slight penalty for potential rambling
    else:
        score += 0.1
        
    # 2. Lexical Diversity of Assistant
    # We want varied vocabulary
    diversity = _calculate_lexical_diversity(" ".join(asst_msgs))
    score += (diversity * 0.2)
    
    # 3. Instruction Complexity
    # We want complex/interesting user queries
    complexity = _calculate_instruction_complexity(" ".join(user_msgs))
    score += (complexity * 0.2)
    
    return max(0.0, min(1.0, score))

# ============================================================================
# PRIORITIZATION & FILTERING
# ============================================================================

def prioritize_examples(raw_data: List[Dict], top_n: int = None, threshold: float = 0.3) -> List[Dict]:
    """
    Sort examples by quality score and filter out low-quality ones.
    """
    scored_data = []
    for entry in raw_data:
        quality = _score_conversation_quality(entry.get("messages", []))
        # Store score in the entry for debugging/analysis
        entry["_quality_score"] = quality
        scored_data.append(entry)
    
    # Sort by score descending
    scored_data.sort(key=lambda x: x["_quality_score"], reverse=True)
    
    # Filter by threshold
    filtered_data = [x for x in scored_data if x["_quality_score"] >= threshold]
    
    # Return top N
    if top_n and top_n < len(filtered_data):
        return filtered_data[:top_n]
    
    return filtered_data

# ============================================================================
# AUGMENTATION
# ============================================================================

def balance_classes(raw_data: List[Dict], target_col: str = None) -> List[Dict]:
    """
    Simple augmentation: Oversample underrepresented classes for classification tasks.
    (Requires identifying a 'label' which is tricky in pure chat, 
     but we can try to infer from short assistant responses if they look like classes)
    """
    # Heuristic: If assistant responses are short (<50 chars) and repetitive, 
    # treat them as classification labels.
    
    # 1. Extract potential labels
    labels = []
    for entry in raw_data:
        # Check last assistant message
        msgs = entry.get("messages", [])
        if not msgs: 
            labels.append(None)
            continue
        last_msg = msgs[-1]
        if last_msg["role"] == "assistant":
            content = last_msg["content"].strip().lower()
            if len(content) < 50:
                labels.append(content)
            else:
                labels.append(None) # Not a classification-style response
        else:
            labels.append(None)
    
    # If too many None, skip balancing
    if not labels or labels.count(None) / len(labels) > 0.5:
        return raw_data # Probably not a classification dataset
        
    # 2. Count classes
    valid_labels = [l for l in labels if l is not None]
    if not valid_labels:
        return raw_data
        
    counts = Counter(valid_labels)
    max_count = max(counts.values())
    
    # 3. Oversample
    augmented_data = list(raw_data)
    
    for label, count in counts.items():
        if count < max_count:
            # Find examples with this label
            examples = [
                entry for entry, l in zip(raw_data, labels) 
                if l == label
            ]
            
            if not examples: continue
            
            # Calculate how many to add
            needed = max_count - count
            
            # Add random copies (simple oversampling)
            # In a real scenario, we'd use an LLM to rephrase here
            for _ in range(needed):
                # Deep copy to avoid reference issues
                copy_entry = json.loads(json.dumps(random.choice(examples)))
                augmented_data.append(copy_entry)
                
    return augmented_data

def generate_variations(state: "TuneKitState") -> dict:
    """
    Placeholder for LLM-based data generation.
    """
    # TODO: Implement LLM integration for true synthetic data generation
    return {
        "warning": "LLM-based generation requires API configuration. Using heuristic balancing instead."
    }

# ============================================================================
# MAIN TOOL
# ============================================================================

def enrich_dataset(state: "TuneKitState") -> dict:
    """
    Enrich the dataset with scores, prioritization, and basic augmentation.
    
    Inputs (from state):
        - raw_data: List[Dict]
        - enrich_config: Dict (optional)
            - top_n: int
            - balance: bool
    
    Outputs (to state):
        - enriched_data: List[Dict]
        - enrichment_stats: Dict
    """
    raw_data = state.get("raw_data", [])
    config = state.get("enrich_config", {})
    
    if not raw_data:
        return {"enriched_data": [], "enrichment_stats": {"error": "No data"}}
    
    # 1. Scoring & Prioritization
    top_n = config.get("top_n", None)
    
    # Use 0.0 threshold to keep everything by default unless very bad
    prioritized_data = prioritize_examples(raw_data, top_n=top_n, threshold=0.1)
    
    # 2. Balancing (Augmentation)
    if config.get("balance", True):
        final_data = balance_classes(prioritized_data)
    else:
        final_data = prioritized_data
        
    # Stats
    stats = {
        "original_count": len(raw_data),
        "enriched_count": len(final_data),
        "avg_quality_score": sum(x.get("_quality_score", 0) for x in final_data) / len(final_data) if final_data else 0
    }
    
    return {
        "enriched_data": final_data,
        "enrichment_stats": stats
    }
