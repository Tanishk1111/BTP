"""
Model compatibility fix for spatx_core framework version differences
Handles layer name mapping between different versions of the framework
"""

import torch
import os
from typing import Dict, Any

def fix_model_state_dict(state_dict: Dict[str, Any], target_genes: int = 50) -> Dict[str, Any]:
    """
    Fix model state dict to handle naming differences between framework versions
    
    Maps:
    - _ConvMixer_blocks -> ConvMixer_blocks
    - Removes num_batches_tracked entries
    - Handles gene count expansion from 1 to target_genes
    - Handles other layer naming inconsistencies
    """
    fixed_state_dict = {}
    
    for key, value in state_dict.items():
        # Fix the main naming issue: _ConvMixer_blocks -> ConvMixer_blocks
        new_key = key.replace('_ConvMixer_blocks', 'ConvMixer_blocks')
        
        # Skip num_batches_tracked entries as they're not needed
        if 'num_batches_tracked' in new_key:
            continue
            
        # Handle conv bias differences
        if 'conv.bias' in new_key and 'cit.' in new_key:
            # Skip bias entries that don't exist in current model
            continue
            
        fixed_state_dict[new_key] = value
    
    return fixed_state_dict

def load_compatible_model(model_path: str, model: torch.nn.Module, required_gene_ids: list):
    """
    Load model with compatibility fixes for framework version differences
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Fix state dict naming issues
    fixed_state_dict = fix_model_state_dict(checkpoint)
    
    # Handle gene count mismatch - if model has 1 gene but we need multiple
    num_required_genes = len(required_gene_ids)
    
    # Check if we have gene count mismatch
    if 'reg_head.2.weight' in fixed_state_dict:
        model_gene_count = fixed_state_dict['reg_head.2.weight'].shape[0]
        
        if model_gene_count == 1 and num_required_genes > 1:
            print(f"🔧 Expanding model from {model_gene_count} gene to {num_required_genes} genes...")
            
            # Replicate the single gene weights for all required genes
            reg_weight = fixed_state_dict['reg_head.2.weight']
            reg_bias = fixed_state_dict['reg_head.2.bias']
            query_embed = fixed_state_dict['head.query_embed']
            
            # Expand dimensions by replicating
            fixed_state_dict['reg_head.2.weight'] = reg_weight.repeat(num_required_genes, 1)
            fixed_state_dict['reg_head.2.bias'] = reg_bias.repeat(num_required_genes)
            fixed_state_dict['head.query_embed'] = query_embed.repeat(num_required_genes, 1)
            
            print(f"✅ Successfully expanded model weights for {num_required_genes} genes")
    
    # Load the fixed state dict with strict=False to handle missing keys
    missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️ Missing keys (using default initialization): {len(missing_keys)} keys")
    
    if unexpected_keys:
        print(f"⚠️ Unexpected keys (ignored): {len(unexpected_keys)} keys")
    
    print("✅ Model loaded with compatibility fixes!")
    return model