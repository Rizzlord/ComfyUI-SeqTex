
import torch
import tempfile
import os

def save_tensor_to_file(tensor, prefix="tensor"):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt", prefix=f"{prefix}_")
    torch.save(tensor, temp_file.name)
    temp_file.close()
    return temp_file.name

def load_tensor_from_file(file_path, map_location='cpu'):
    return torch.load(file_path, map_location=map_location)
