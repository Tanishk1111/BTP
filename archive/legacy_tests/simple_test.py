import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'spatx_core'))

try:
    print("🔧 Starting compatibility test...")
    from model_compatibility_fix import fix_model_state_dict
    print("✅ Imported compatibility fix")
    
    from spatx_core.models.cit_to_gene.CiT_Net_T import CIT
    print("✅ Imported CIT")
    
    from spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor  
    print("✅ Imported CITGenePredictor")
    
    import torch
    print("✅ Imported PyTorch")
    
    # Load model state dict
    print("📁 Loading model file...")
    model_path = "saved_models/cit_to_gene/model_working_model.pth"
    state_dict = torch.load(model_path, map_location='cpu')
    print(f"📊 Loaded {len(state_dict)} keys from model file")
    
    # Show some key names to understand the structure
    print("🔍 Sample keys from model:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {i+1}. {key}")
    
    # Test our fix
    print("🔧 Applying compatibility fix...")
    fixed_dict = fix_model_state_dict(state_dict, target_genes=50)
    print(f"✅ Fixed dict has {len(fixed_dict)} keys")
    
    print("🎉 COMPATIBILITY FIX TEST PASSED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()