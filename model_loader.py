import os
import torch
from typing import List, Tuple, Dict

try:
    # Preferred import when spatx_core installed as a package (pip/editable)
    from spatx_core.models.cit_to_gene.CiT_Net_T import CIT  # type: ignore
    from spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor  # type: ignore
except ModuleNotFoundError:
    # Fallback to relative nested structure inside repo (spatx_core/spatx_core/...)
    from spatx_core.spatx_core.models.cit_to_gene.CiT_Net_T import CIT  # type: ignore
    from spatx_core.spatx_core.models.cit_to_gene.CiTGene import CITGenePredictor  # type: ignore

GENE_FILE_SUFFIX = "_genes.txt"
MODEL_DIR = "saved_models/cit_to_gene"

class ModelLoadError(RuntimeError):
    pass

def _model_base_paths(model_id: str) -> Tuple[str, str]:
    safe_id = model_id.strip().replace(" ", "_")
    model_path = os.path.join(MODEL_DIR, f"{safe_id}.pth")
    gene_path = os.path.join(MODEL_DIR, f"{safe_id}_genes.txt")
    return model_path, gene_path

def load_gene_ids(model_id: str) -> List[str]:
    _, gene_path = _model_base_paths(model_id)
    if not os.path.exists(gene_path):
        # Fallback: try a python module with same model_id exposing gene_ids list (e.g. working_model.py)
        module_candidate = os.path.join(MODEL_DIR, f"{model_id}.py")
        if os.path.exists(module_candidate):
            try:
                import importlib.util, sys
                spec = importlib.util.spec_from_file_location(f"genes_{model_id}", module_candidate)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[spec.name] = mod
                    spec.loader.exec_module(mod)  # type: ignore
                    if hasattr(mod, 'gene_ids') and isinstance(mod.gene_ids, list) and mod.gene_ids:
                        return [str(g).strip() for g in mod.gene_ids if str(g).strip()]
            except Exception as imp_ex:
                raise ModelLoadError(f"Gene ID file missing and fallback import failed: {imp_ex}")
        raise ModelLoadError(f"Gene ID file not found for model_id '{model_id}' at {gene_path}")
    with open(gene_path, 'r', encoding='utf-8') as f:
        genes = [ln.strip() for ln in f if ln.strip()]
    if not genes:
        raise ModelLoadError(f"Gene ID file empty for model_id '{model_id}'")
    return genes

def build_backbone(device: str = 'cpu') -> CIT:
    # Mirrors training configuration
    return CIT(img_size=224, in_chans=3,
               embed_dim=96,
               depths=[2,2,6,2],
               num_heads=[3,6,12,24],
               device=device)

def build_model(num_genes: int, device: str = 'cpu') -> CITGenePredictor:
    backbone = build_backbone(device=device)
    model = CITGenePredictor(backbone, num_genes=num_genes)
    return model.to(device)

def load_state_dict(model_id: str) -> Dict[str, torch.Tensor]:
    model_path, _ = _model_base_paths(model_id)
    if not os.path.exists(model_path):
        # Fallback: legacy naming 'model_<model_id>.pth'
        safe_id = model_id.strip().replace(" ", "_")
        legacy_path = os.path.join(MODEL_DIR, f"model_{safe_id}.pth")
        if os.path.exists(legacy_path):
            model_path = legacy_path
        else:
            raise ModelLoadError(
                f"Model weights not found for model_id '{model_id}' at {model_path} (also tried {legacy_path})"
            )
    ckpt = torch.load(model_path, map_location='cpu')
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']  # type: ignore
    raise ModelLoadError(f"Unsupported checkpoint format for model_id '{model_id}'")

def expand_single_head_state_dict(sd: Dict[str, torch.Tensor], target_genes: int) -> Dict[str, torch.Tensor]:
    """Demo-only expansion: replicate single gene head weights to target_genes.
    This produces synthetic identical outputs and should be replaced with a real multi-gene model.
    """
    if 'reg_head.2.weight' not in sd or 'reg_head.2.bias' not in sd:
        raise ModelLoadError("Cannot locate regression head layers (reg_head.2.*) in state dict")
    w = sd['reg_head.2.weight']
    b = sd['reg_head.2.bias']
    if w.shape[0] != 1:
        raise ModelLoadError("Expansion only allowed when original model has a single output neuron")
    sd = dict(sd)  # shallow copy
    sd['reg_head.2.weight'] = w.repeat(target_genes, 1).clone()
    sd['reg_head.2.bias'] = b.repeat(target_genes).clone()
    # NOTE: GeneTransformerHead parameters (query_embed etc.) will mismatch; we cannot faithfully expand them.
    # We therefore remove them so load_state_dict with strict=False can skip, and the new head will random-init.
    for k in list(sd.keys()):
        if k.startswith('head.'):
            sd.pop(k)
    return sd

def load_or_expand_model(model_id: str, requested_genes: List[str], device: str = 'cpu', allow_expand: bool = True) -> Tuple[CITGenePredictor, List[str], str]:
    """Load model; if underlying state dict only has 1 output and multiple genes requested,
    optionally expand for demo purposes.
    Returns (model, model_gene_ids, mode) where mode is 'real' or 'expanded'."""
    model_gene_ids = load_gene_ids(model_id)
    full_gene_count = len(model_gene_ids)
    sd = load_state_dict(model_id)
    head_rows = sd.get('reg_head.2.weight').shape[0] if 'reg_head.2.weight' in sd else None  # type: ignore
    target_subset = requested_genes if requested_genes else model_gene_ids

    if head_rows == 1 and full_gene_count > 1 and len(target_subset) > 1:
        if not allow_expand:
            raise ModelLoadError("Model is single-gene; multi-gene prediction requested and expansion disabled.")
        model = build_model(num_genes=len(target_subset), device=device)
        expanded_sd = expand_single_head_state_dict(sd, len(target_subset))
        missing, unexpected = model.load_state_dict(expanded_sd, strict=False)
        mode = 'expanded'
    else:
        model = build_model(num_genes=full_gene_count, device=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        mode = 'real'

    return model, model_gene_ids, mode

def map_requested_genes(model_gene_ids: List[str], requested: List[str]) -> List[int]:
    indices = []
    for g in requested:
        if g not in model_gene_ids:
            raise ModelLoadError(f"Requested gene '{g}' not in model gene set")
        indices.append(model_gene_ids.index(g))
    return indices
