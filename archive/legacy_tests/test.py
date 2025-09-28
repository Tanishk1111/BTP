import torch, os, json, textwrap
pth='saved_models/cit_to_gene/model_working_model.pth'
if not os.path.exists(pth):
    print('MISSING', pth)
    raise SystemExit
ckpt=torch.load(pth,map_location='cpu')
# Detect if raw state dict
if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
    sd=ckpt
else:
    sd=ckpt.get('state_dict', {})
print('Total tensors:', len(sd))
head_keys=[k for k in sd.keys() if any(t in k for t in ['reg_head','head','fc'])]
print('Head-related keys count:', len(head_keys))
for k in head_keys[:25]:
    t=sd[k]
    print(k, tuple(t.shape))
# Try to identify final linear weight
candidates=[(k,v) for k,v in sd.items() if v.ndim==2 and v.shape[0] <= 256 and v.shape[1] > 10]
print('\nCandidate final layers (first 5):')
for k,v in candidates[:5]:
    print('  ',k, tuple(v.shape))
# Specifically check known names
for probe in ['reg_head.2.weight','reg_head.2.bias','reg_head.0.weight','fc.weight']:
    if probe in sd:
        print('FOUND', probe, tuple(sd[probe].shape))
print('\nSummary:')
if 'reg_head.2.weight' in sd:
    out_dim=sd['reg_head.2.weight'].shape[0]
    print('Output neurons (reg_head.2.weight rows):', out_dim)
else:
    print('Could not find reg_head.2.weight; need manual inspection')