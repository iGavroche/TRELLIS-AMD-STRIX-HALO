import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _remap_spconv_to_torchsparse(state_dict):
    """
    Remap spconv state dict keys to torchsparse format and reshape kernel weights.
    
    spconv uses:
      - Key: 'conv.weight' 
      - Shape: [out_channels, k, k, k, in_channels]
    
    torchsparse uses:
      - Key: 'conv.kernel'
      - Shape: [k*k*k, in_channels, out_channels]
    """
    import torch
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        new_value = value
        
        # Remap spconv weight naming to torchsparse kernel naming and reshape
        if '.conv.weight' in key:
            new_key = key.replace('.conv.weight', '.conv.kernel')
            # Reshape: spconv [out_ch, k, k, k, in_ch] -> torchsparse [k*k*k, in_ch, out_ch]
            if value.dim() == 5:
                # Standard 3D sparse conv kernel
                out_ch, k1, k2, k3, in_ch = value.shape
                # Transpose from [out_ch, k, k, k, in_ch] to [k, k, k, in_ch, out_ch]
                value = value.permute(1, 2, 3, 4, 0)
                # Reshape to [k*k*k, in_ch, out_ch]
                new_value = value.reshape(k1 * k2 * k3, in_ch, out_ch)
                # For 1x1x1 kernels, torchsparse expects [in_ch, out_ch] (2D, squeezed)
                if k1 * k2 * k3 == 1:
                    new_value = new_value.squeeze(0)  # [1, in_ch, out_ch] -> [in_ch, out_ch]
            elif value.dim() == 2:
                # 1x1x1 kernel stored as [out_ch, in_ch] -> needs to be [in_ch, out_ch]
                out_ch, in_ch = value.shape
                new_value = value.t()  # [out_ch, in_ch] -> [in_ch, out_ch]
        
        new_state_dict[new_key] = new_value
    return new_state_dict


# Models that use sparse convolutions (SparseConv3d) and need key remapping for torchsparse
_SPARSE_CONV_MODELS = {
    'SLatEncoder', 'SLatGaussianDecoder', 'SLatRadianceFieldDecoder', 'SLatMeshDecoder',
    'ElasticSLatEncoder', 'ElasticSLatGaussianDecoder', 'ElasticSLatRadianceFieldDecoder', 'ElasticSLatMeshDecoder',
    'SLatFlowModel', 'ElasticSLatFlowModel'
}


def from_pretrained(path: str, base_repo_id: str = None, revision: str = None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        base_repo_id: Optional base repository ID to use when path is relative (e.g., "microsoft/TRELLIS-image-large")
        revision: Optional revision/branch to use (e.g., "refs/pr/16" or "main"). If None, tries main first, then common PR branches.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        path_parts = path.split('/')
        
        # Determine repo_id and model_name
        # Check if path already contains repo_id (starts with owner/repo)
        if len(path_parts) >= 2 and base_repo_id and path.startswith(base_repo_id):
            # Path already contains repo_id, extract it
            repo_id = f'{path_parts[0]}/{path_parts[1]}'
            model_name = '/'.join(path_parts[2:]) if len(path_parts) > 2 else ''
        elif base_repo_id:
            # Use base_repo_id and treat path as relative
            repo_id = base_repo_id
            model_name = path
        elif len(path_parts) >= 2:
            # Absolute path: repo_owner/repo_name/path/to/model
            repo_id = f'{path_parts[0]}/{path_parts[1]}'
            model_name = '/'.join(path_parts[2:]) if len(path_parts) > 2 else ''
        else:
            # Fallback: assume it's a relative path in the default repo
            # This shouldn't happen, but handle it gracefully
            raise ValueError(f"Invalid model path format: {path}. Expected format: 'repo_owner/repo_name/path/to/model' or provide base_repo_id for relative paths.")
        
        # Try to download with specified revision, or try common revisions
        revisions_to_try = [revision] if revision else ["main", "refs/pr/16", "refs/pr/15", "refs/pr/14"]
        
        config_file = None
        model_file = None
        last_error = None
        
        for rev in revisions_to_try:
            if rev is None:
                continue
            try:
                config_file = hf_hub_download(repo_id, f"{model_name}.json" if model_name else "model.json", revision=rev)
                model_file = hf_hub_download(repo_id, f"{model_name}.safetensors" if model_name else "model.safetensors", revision=rev)
                if config_file and model_file:
                    break
            except (EntryNotFoundError, Exception) as e:
                last_error = e
                continue
        
        if not config_file or not model_file:
            raise EntryNotFoundError(f"Could not find model files for {repo_id}/{model_name} in any revision. Last error: {last_error}")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model_name = config['name']
    model = __getattr__(model_name)(**config['args'], **kwargs)
    
    # Load state dict and remap keys if using torchsparse backend AND model uses sparse convolutions
    state_dict = load_file(model_file)
    sparse_backend = os.environ.get('SPARSE_BACKEND', 'spconv')
    if sparse_backend == 'torchsparse' and model_name in _SPARSE_CONV_MODELS:
        state_dict = _remap_spconv_to_torchsparse(state_dict)
    
    model.load_state_dict(state_dict)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )
