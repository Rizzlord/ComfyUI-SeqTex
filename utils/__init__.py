import numpy as np
import torch
from einops import rearrange
from PIL import Image


def tensor_to_pil(tensor, mask=None, normalize: bool = True):
    """
    Convert tensor to PIL Image.
    :param tensor: torch.Tensor or str (file path to tensor), shape can be (Nv, H, W, C), (Nv, C, H, W), (H, W, C), (C, H, W)
    :param mask: torch.Tensor or str (file path to tensor), shape same as tensor, effective when C=3
    :return: PIL.Image
    """
    # If input is a file path, load the tensor
    if isinstance(tensor, str):
        from utils.file_utils import load_tensor_from_file
        tensor = load_tensor_from_file(tensor, map_location="cpu")
    if mask is not None and isinstance(mask, str):
        from utils.file_utils import load_tensor_from_file
        mask = load_tensor_from_file(mask, map_location="cpu")
    # Move to cpu
    tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if mask is not None and mask.is_cuda:
        mask = mask.cpu()

    # Convert to float32
    tensor = tensor.float()
    if mask is not None:
        mask = mask.float()

    if normalize:
        tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)
    if mask is not None:
        if mask.shape[-1] not in [1, 3]:
            mask = mask.unsqueeze(-1)
        tensor = torch.cat([tensor, mask], dim=-1)

    shape = tensor.shape
    # 4D: (Nv, H, W, C) or (Nv, C, H, W)
    if len(shape) == 4:
        Nv = shape[0]
        if shape[-1] in [3, 4]:  # (Nv, H, W, C)
            tensor = rearrange(tensor, 'nv h w c -> h (nv w) c')
        else:  # (Nv, C, H, W)
            tensor = rearrange(tensor, 'nv c h w -> h (nv w) c')
    # 3D: (H, W, C) or (C, H, W)
    elif len(shape) == 3:
        if shape[-1] in [3, 4]:  # (H, W, C)
            tensor = rearrange(tensor, 'h w c -> h w c')
        else:  # (C, H, W)
            tensor = rearrange(tensor, 'c h w -> h w c')
    else:
        raise ValueError(f"Unsupported tensor shape: {shape}")

    # Convert to numpy
    np_img = (tensor.numpy() * 255).round().astype(np.uint8)

    # Create PIL Image
    if np_img.shape[2] == 3:
        return Image.fromarray(np_img, mode="RGB")
    elif np_img.shape[2] == 4:
        return Image.fromarray(np_img, mode="RGBA")
    else:
        raise ValueError("Only support 3 or 4 channel images.")