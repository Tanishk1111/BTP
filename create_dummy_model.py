import torch
import torch.nn as nn

# Create a simple dummy model that matches CITGenePredictor structure
dummy_state_dict = {
    'dummy_param': torch.randn(1)
}

# Save the dummy model
torch.save(dummy_state_dict, 'spatx_core/saved_models/cit_to_gene/model_test_model.pth')
print('✅ Created dummy model file: model_test_model.pth')



