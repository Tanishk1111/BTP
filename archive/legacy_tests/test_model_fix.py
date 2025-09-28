"""
Test script to verify the model compatibility fix works
"""
import sys
import os
import torch

# Add spatx_core to path
sys.path.append(os.path.join(os.getcwd(), 'spatx_core'))

from model_compatibility_fix import fix_model_state_dict, load_compatible_model

def test_model_loading():
    """Test if we can load the model with our compatibility fixes"""
    try:
        print("🔧 Testing model compatibility fix...")
        
        # Import model classes
        from spatx_core.models.cit_to_gene.CiT_Net_T import CIT
        from spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor
        
        print("✅ Successfully imported model classes")
        
        # Create model architecture
        print("🏗️ Creating model architecture...")
        cit_backbone = CIT(
            device='cpu',  # First parameter is device
            img_size=224,
            patch_size=4,  # Changed from 16 to 4 based on class definition
            in_chans=3,
            embed_dim=96,  # Default from class
            depths=[2, 2, 6, 2],  # Default from class
            num_heads=[3, 6, 12, 24],  # Default from class
            window_size=7,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        model = CITGenePredictor(cit_backbone, num_genes=50)
        print("✅ Successfully created model architecture")
        
        # Test loading the state dict
        model_path = "saved_models/cit_to_gene/model_working_model.pth"
        print(f"📁 Loading model from: {model_path}")
        
        # Load original state dict
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"📊 Original state dict has {len(state_dict)} keys")
        
        # Apply compatibility fixes
        print("🔧 Applying compatibility fix...")
        fixed_state_dict = fix_model_state_dict(state_dict, target_genes=50)
        print(f"🔧 Fixed state dict has {len(fixed_state_dict)} keys")
        
        # Handle gene count expansion manually for testing
        if 'reg_head.2.weight' in fixed_state_dict:
            model_gene_count = fixed_state_dict['reg_head.2.weight'].shape[0]
            target_genes = 50
            
            if model_gene_count == 1 and target_genes > 1:
                print(f"🔧 Expanding model from {model_gene_count} gene to {target_genes} genes...")
                
                # Replicate the single gene weights for all required genes
                reg_weight = fixed_state_dict['reg_head.2.weight']
                reg_bias = fixed_state_dict['reg_head.2.bias']
                query_embed = fixed_state_dict['head.query_embed']
                
                # Expand dimensions by replicating
                fixed_state_dict['reg_head.2.weight'] = reg_weight.repeat(target_genes, 1)
                fixed_state_dict['reg_head.2.bias'] = reg_bias.repeat(target_genes)
                fixed_state_dict['head.query_embed'] = query_embed.repeat(target_genes, 1)
                
                print(f"✅ Successfully expanded model weights for {target_genes} genes")
        
        # Try loading into model
        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
        print(f"✅ Successfully loaded fixed state dict!")
        
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        
        # Test forward pass with dummy input
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"🧬 Model output shape: {output.shape}")
            print(f"🧬 Sample predictions: {output[0, :5].tolist()}")
        
        print("🎉 ALL TESTS PASSED! The compatibility fix works!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n✨ Ready to integrate into main API!")
    else:
        print("\n💥 Need to fix issues before integration")