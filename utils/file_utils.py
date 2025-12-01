
import torch
#import tempfile
from pathlib import Path
import os
import uuid
import folder_paths  # ComfyUI's helper

def save_tensor_to_file(tensor, prefix="tensor"):
    # Get ComfyUI's temp directory
    temp_dir = Path(folder_paths.get_temp_directory())
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename
    file_name = temp_dir / f"{prefix}_{uuid.uuid4().hex}.pt"
    temp_path = file_name

    # Save the tensor
    torch.save(tensor, temp_path)

    # Return path relative to ComfyUI temp folder
    return str(file_name)

def load_tensor_from_file(file_path, map_location='cpu'):
    return torch.load(file_path, map_location=map_location)
