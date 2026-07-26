#!/usr/bin/env python3
"""
TuneKit CLI
===========
Manage models and configuration for TuneKit.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

# Add local path to import tunekit modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tunekit.tools.enrich import enrich_dataset
from tunekit.tools.ingest import ingest_data

# Path to configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tunekit', 'data')
CONFIG_PATH = os.path.join(DATA_DIR, 'models.json')

def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {CONFIG_PATH}")
        sys.exit(1)

def save_config(config: Dict[str, Any]):
    """Save configuration to JSON file."""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Configuration saved to {CONFIG_PATH}")

def list_models(args):
    """List all configured models."""
    config = load_config()
    models = config.get('models', {})
    
    print(f"\n📦 Configured Models ({len(models)}):")
    print("=" * 60)
    print(f"{'Key':<20} | {'Name':<25} | {'Size':<10}")
    print("-" * 60)
    
    for key, data in models.items():
        meta = data.get('metadata', {})
        print(f"{key:<20} | {meta.get('name', 'N/A'):<25} | {meta.get('size', 'N/A'):<10}")
    print("=" * 60)

def input_default(prompt: str, default: Any) -> Any:
    """Get user input with a default value."""
    user_input = input(f"{prompt} [{default}]: ").strip()
    if not user_input:
        return default
    return user_input

import requests

def get_hf_metadata(model_id: str) -> Dict[str, Any]:
    """Fetch model metadata from HuggingFace API."""
    print(f"   🔍 Querying HuggingFace API for '{model_id}'...")
    try:
        # Get basic info
        url = f"https://huggingface.co/api/models/{model_id}"
        response = requests.get(url, timeout=5)
        
        if response.status_code != 200:
            print("      ⚠️ Could not fetch metadata (Model not found or API error)")
            return {}
            
        data = response.json()
        
        # Extract config to find context window
        config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        config_resp = requests.get(config_url, timeout=5)
        config_data = config_resp.json() if config_resp.status_code == 200 else {}
        
        metadata = {
            "id": data.get("modelId", model_id),
            "tags": data.get("tags", []),
            "downloads": data.get("downloads", 0),
        }
        
        # Try to guess size from tags or safetensors
        for tag in metadata["tags"]:
            if tag.endswith("b") or tag.endswith("m"):
                # Rough heuristic
                pass

        # Try to find context window
        ctx = config_data.get("max_position_embeddings") or config_data.get("seq_length") or config_data.get("n_positions")
        if ctx:
            metadata["context_window"] = ctx
            
        # Try to guess params count from safetensors index if available
        # (This is complex, so we'll stick to manual entry if not obvious)
        
        print("      ✅ Metadata found!")
        return metadata
        
    except Exception as e:
        print(f"      ⚠️ API Error: {e}")
        return {}

def add_model(args):
    """Interactive wizard to add a new model."""
    config = load_config()
    defaults = config.get('defaults', {})
    
    print("\n🤖 TuneKit AI Assistant")
    print("=======================")
    print("I'll help you add a new model to the ecosystem.")
    print("Please provide the details below.\n")
    
    # 1. Metadata
    print("📝 Step 1: Model Metadata")
    
    while True:
        key = input("   Unique Key (e.g., deepseek-1.3b): ").strip()
        if not key:
            print("   Key cannot be empty.")
            continue
        if key in config['models']:
            print(f"   ❌ Model '{key}' already exists. Choose another key.")
            continue
        break
        
    hf_id = input("   HuggingFace ID (e.g., deepseek-ai/deepseek-coder-1.3b-instruct): ").strip()
    
    # Auto-fetch metadata
    hf_meta = {}
    if hf_id:
        hf_meta = get_hf_metadata(hf_id)
    
    name_default = hf_id.split('/')[-1].replace('-', ' ').title() if hf_id else ""
    name = input_default("   Display Name", name_default)
    
    size = input("   Size (e.g., 1.3B): ").strip()
    
    ctx_default = hf_meta.get("context_window", 8192)
    context_window = int(input_default("   Context Window (tokens)", ctx_default))
    gpu_tier = input_default("   Recommended GPU", "T4")
    
    # 2. Scores
    print("\n📊 Step 2: Scoring Configuration")
    print("   Rate the model's performance (0-100) for each task.")
    
    task_scores = {}
    for task in ['classify', 'qa', 'conversation', 'generation', 'extraction']:
        task_scores[task] = int(input_default(f"   Score for '{task}'", defaults.get('task_score', 25)))
        
    print("\n   Set bonus points for special capabilities.")
    multi_turn = int(input_default("   Multi-turn Bonus (0-10)", 0))
    json_score = int(input_default("   JSON Output Score (0-20)", 10))
    long_output = int(input_default("   Long Output Score (0-20)", 10))
    
    # 3. Reasons
    print("\n🗣️ Step 3: Assistant Reasoning")
    print("   What makes this model special? (Comma separated features)")
    features_input = input("   Features: ").strip()
    features = [f.strip() for f in features_input.split(',')] if features_input else []
    
    # Construct model object
    new_model = {
        "metadata": {
            "id": hf_id,
            "name": name,
            "size": size,
            "context_window": context_window,
            "training_time_base": 3,  # Default
            "cost_base": 0.0,
            "gpu_tier": gpu_tier,
            "memory_gb": 8,  # Conservative default
            "accuracy_baseline": 80
        },
        "scores": {
            "task": task_scores,
            "size": {
                "small": 20,
                "medium": 20,
                "large": 20
            },
            "output": {
                "long": long_output,
                "json": json_score
            },
            "multi_turn_bonus": multi_turn
        },
        "deployment": ["cloud_api", "desktop_app", "not_sure"],
        "reasons": {
            "features": features
        }
    }
    
    # Save
    config['models'][key] = new_model
    
    # Update tiebreakers (append to end)
    tiebreaker = config.get('tiebreaker', {})
    for cat in tiebreaker:
        if key not in tiebreaker[cat]:
            tiebreaker[cat].append(key)
    
    save_config(config)
    print(f"\n✅ Model '{name}' ({key}) successfully added to TuneKit!")

def remove_model(args):
    """Remove a model."""
    config = load_config()
    key = args.key
    
    if key not in config['models']:
        print(f"❌ Error: Model '{key}' not found.")
        return
        
    confirm = input(f"⚠️ Are you sure you want to remove '{key}'? (y/N): ").lower()
    if confirm == 'y':
        del config['models'][key]
        
        # Remove from tiebreakers
        tiebreaker = config.get('tiebreaker', {})
        for cat in tiebreaker:
            if key in tiebreaker[cat]:
                tiebreaker[cat].remove(key)
                
        save_config(config)
        print(f"🗑️ Model '{key}' removed.")
    else:
        print("Cancelled.")

def enrich_cmd(args):
    """Run data enrichment on a file."""
    file_path = args.file
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return
        
    print(f"\n🚀 Enriching dataset: {file_path}")
    print("   Analysis started...")
    
    # 1. Ingest
    state = {"file_path": file_path}
    ingest_res = ingest_data(state)
    
    if ingest_res.get("error_msg"):
        print(f"❌ Ingestion Error: {ingest_res['error_msg']}")
        return
        
    state.update(ingest_res)
    raw_count = len(state["raw_data"])
    print(f"   ✓ Loaded {raw_count} examples.")
    
    # 2. Enrich
    print(f"   Applying metrics (Top N: {args.top_n}, Balance: {args.balance})...")
    state["enrich_config"] = {
        "top_n": args.top_n,
        "balance": args.balance
    }
    
    enrich_res = enrich_dataset(state)
    enriched_data = enrich_res.get("enriched_data", [])
    stats = enrich_res.get("enrichment_stats", {})
    
    # 3. Save Output
    if args.output:
        out_path = args.output
    else:
        name, ext = os.path.splitext(file_path)
        out_path = f"{name}_enriched{ext}"
        
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for entry in enriched_data:
                f.write(json.dumps(entry) + '\n')
                
        print("\n✅ Enrichment Complete!")
        print(f"   Input:  {stats.get('original_count')} examples")
        print(f"   Output: {stats.get('enriched_count')} examples")
        print(f"   Score:  {stats.get('avg_quality_score', 0):.4f} (Avg Quality)")
        print(f"   Saved:  {out_path}")
        
    except Exception as e:
        print(f"❌ Save Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="TuneKit CLI Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    subparsers.add_parser("list", help="List configured models")
    
    # Add command
    subparsers.add_parser("add", help="Add a new model via AI Assistant")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a model")
    remove_parser.add_argument("key", help="Model key to remove")
    
    # Enrich command
    enrich_parser = subparsers.add_parser("enrich", help="Enrich a dataset")
    enrich_parser.add_argument("file", help="Path to JSONL file")
    enrich_parser.add_argument("--top_n", type=int, default=None, help="Keep only top N examples")
    enrich_parser.add_argument("--no-balance", action="store_false", dest="balance", help="Disable class balancing")
    enrich_parser.add_argument("-o", "--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_models(args)
    elif args.command == "add":
        add_model(args)
    elif args.command == "remove":
        remove_model(args)
    elif args.command == "enrich":
        enrich_cmd(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
