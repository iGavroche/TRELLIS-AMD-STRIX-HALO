from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        # Extract base repo_id from path for relative model paths
        # If path looks like a HuggingFace repo (has /), use it as base
        if '/' in path and not os.path.exists(f"{path}/pipeline.json"):
            base_repo_id = path
        else:
            base_repo_id = None
        
        for k, v in args['models'].items():
            # Try with relative path first if we have base_repo_id (avoids duplicating repo_id)
            if base_repo_id and v:
                try:
                    # Use relative path with base_repo_id
                    _models[k] = models.from_pretrained(v, base_repo_id=base_repo_id)
                except Exception as e:
                    # Check if it's a download/URL error vs a runtime error
                    from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
                    from requests.exceptions import HTTPError
                    
                    is_download_error = isinstance(e, (RepositoryNotFoundError, EntryNotFoundError, HTTPError)) or \
                                       "404" in str(e) or "Not Found" in str(e) or "Repository Not Found" in str(e)
                    
                    if is_download_error:
                        # Try with full path as fallback
                        try:
                            full_path = f"{path}/{v}" if v else path
                            print(f"Warning: Failed to load model {k} from {base_repo_id}/{v}, trying full path {full_path}")
                            _models[k] = models.from_pretrained(full_path)
                        except Exception as e2:
                            # Last resort: try as absolute path
                            print(f"Warning: Failed to load model {k} from {full_path}, trying absolute path: {e2}")
                            _models[k] = models.from_pretrained(v)
                    else:
                        # If it's not a download error (e.g., HIP error), re-raise it
                        print(f"Error loading model {k} from {base_repo_id}/{v}: {e}")
                        raise
            else:
                # No base_repo_id or empty v, try as absolute path
                full_path = f"{path}/{v}" if v else path
                try:
                    _models[k] = models.from_pretrained(full_path)
                except Exception as e:
                    print(f"Error loading model {k} from {full_path}: {e}")
                    raise

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
