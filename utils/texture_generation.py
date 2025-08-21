import os
import threading
from dataclasses import dataclass
from urllib.parse import urlparse
from tqdm import tqdm
import gradio as gr
import numpy as np
import spaces
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from . import tensor_to_pil
from .file_utils import save_tensor_to_file, load_tensor_from_file

from ..wan.pipeline_wan_t2tex_extra import WanT2TexPipeline
from ..wan.wan_t2tex_transformer_3d_extra import WanT2TexTransformer3DModel

TEX_PIPE = None
VAE = None
LATENTS_MEAN, LATENTS_STD = None, None
TEX_PIPE_LOCK = threading.Lock()

@dataclass
class Config:
    video_base_name: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    seqtex_transformer_path: str = "VAST-AI/SeqTex-Transformer"
    min_noise_level_index: int = 15 # refer to paper [WorldMem](https://arxiv.org/pdf/2504.12369v1)

    num_views: int = 4
    uv_num_views: int = 1
    mv_height: int = 512
    mv_width: int = 512
    uv_height: int = 1024
    uv_width: int = 1024

    flow_shift: float = 5.0
    eval_guidance_scale: float = 1.0
    eval_num_inference_steps: int = 30
    eval_seed: int = 42

    # VRAM-friendly chunking (does NOT change results)
    encode_frame_chunk: int = 1   # frames per VAE pass during encoding
    decode_frame_chunk: int = 1   # frames per VAE pass during decoding

cfg = Config()

def get_seqtex_pipe(min_noise_level_index, flow_shift, rope_max_seq_len):
    """
    Lazy load the SeqTex pipeline for texture generation.
    Must be called within @spaces.GPU context.
    """
    global TEX_PIPE, VAE, LATENTS_MEAN, LATENTS_STD
    if TEX_PIPE is not None:
        return TEX_PIPE

    gr.Info("First called, loading SeqTex pipeline... It may take about 1 minute.")
    with TEX_PIPE_LOCK:
        if TEX_PIPE is not None:
            return TEX_PIPE

        auth_token = os.getenv("SEQTEX_SPACE_TOKEN")
        if auth_token is None:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! WARNING: Hugging Face token not found in environment variables.         !!!")
            print("!!! Please set the SEQTEX_SPACE_TOKEN environment variable with your token. !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Load transformer with auto-configured LoRA adapter first (CPU by default)
        transformer = WanT2TexTransformer3DModel.from_pretrained(
            cfg.seqtex_transformer_path,
            token=auth_token,
            rope_max_seq_len=rope_max_seq_len
        )

        # Pipeline - pass the pre-loaded transformer to avoid re-loading
        TEX_PIPE = WanT2TexPipeline.from_pretrained(
            cfg.video_base_name,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )
        del transformer

        # Load VAE with real weights on CPU (avoid meta tensors)
        VAE = AutoencoderKLWan.from_pretrained(
            cfg.video_base_name,
            subfolder="vae",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False  # <-- ensure weights are materialized
        )
        TEX_PIPE.vae = VAE

        # Enable safe VAE memory features when available (no change to outputs)
        try:
            if hasattr(TEX_PIPE.vae, "enable_tiling"):
                TEX_PIPE.vae.enable_tiling()
        except Exception:
            pass
        try:
            if hasattr(TEX_PIPE.vae, "enable_slicing"):
                TEX_PIPE.vae.enable_slicing()
        except Exception:
            pass

        # Keep pipeline on CPU and offload submodules on demand
        try:
            TEX_PIPE.enable_attention_slicing()
        except Exception:
            pass
        try:
            TEX_PIPE.enable_sequential_cpu_offload()  # requires accelerate
        except Exception:
            pass

        # Precompute constants on CPU; move per-chunk as needed
        LATENTS_MEAN = torch.tensor(VAE.config.latents_mean).view(
            1, VAE.config.z_dim, 1, 1, 1
        ).to(torch.float32)
        LATENTS_STD = 1.0 / torch.tensor(VAE.config.latents_std).view(
            1, VAE.config.z_dim, 1, 1, 1
        ).to(torch.float32)

        scheduler: FlowMatchEulerDiscreteScheduler = (
            FlowMatchEulerDiscreteScheduler.from_config(
                TEX_PIPE.scheduler.config, shift=flow_shift
            )
        )
        min_noise_level_index_result = scheduler.config.num_train_timesteps - min_noise_level_index
        setattr(TEX_PIPE, "min_noise_level_index", min_noise_level_index_result)
        min_noise_level_timestep = scheduler.timesteps[min_noise_level_index_result]
        setattr(TEX_PIPE, "min_noise_level_timestep", min_noise_level_timestep)
        setattr(TEX_PIPE, "min_noise_level_sigma", min_noise_level_timestep / 1000.)

        # IMPORTANT: do NOT move the full pipeline or VAE to CUDA here.
        return TEX_PIPE

@torch.amp.autocast('cuda', dtype=torch.float32)
def encode_images(
    images: Float[Tensor, "B F H W C"], encode_as_first: bool = False
) -> Float[Tensor, "B C' F H/8 W/8"]:
    """
    Encode images to latent space using VAE.
    Every frame is seen as a separate image, without any awareness of the temporal dimension.
    :param images: Input images tensor with shape [B, F, H, W, C].
                   (Allowed to be on CPU; slices are moved to device lazily.)
    :param encode_as_first: Whether to encode all frames as the first frame.
    :return: Encoded latents with shape [B, C', F, H/8, W/8].
    """
    global VAE, LATENTS_MEAN, LATENTS_STD

    # Normalize range only once; keep on CPU if possible
    if images.min() < -0.1:
        images = (images + 1.0) / 2.0  # [-1,1] -> [0,1]

    if encode_as_first:
        B, F, H, W, C = images.shape
        latents_out = None
        chunk = max(1, int(cfg.encode_frame_chunk))

        # Prefer CUDA if available; we rely on offload hooks to move VAE as needed.
        prefer_cuda = torch.cuda.is_available()
        target_device = torch.device("cuda") if prefer_cuda else torch.device("cpu")

        for b in range(B):
            latents_frames = []

            # tqdm over frame chunks
            for f0 in tqdm(range(0, F, chunk), desc=f"Encoding batch {b+1}/{B}", leave=False):
                f1 = min(F, f0 + chunk)

                # chunk to device
                img_chunk = images[b, f0:f1]  # CPU
                img_chunk = rearrange(img_chunk, "F H W C -> F C 1 H W").to(target_device, non_blocking=True)

                # move stats to same device
                lat_mean = LATENTS_MEAN.to(target_device, non_blocking=True)
                lat_std = LATENTS_STD.to(target_device, non_blocking=True)

                # VAE forward: offload hooks (if enabled) will bring weights to the right device.
                lat_dist = VAE.encode(img_chunk).latent_dist
                lat_chunk = (lat_dist.sample() - lat_mean) * lat_std  # [Fchunk, C, 1, H/8, W/8]
                latents_frames.append(lat_chunk)

                # free as we go
                del img_chunk, lat_mean, lat_std, lat_dist, lat_chunk
                if prefer_cuda:
                    torch.cuda.empty_cache()

            latents_b = torch.cat(latents_frames, dim=0)  # [F, C, 1, H/8, W/8]
            latents_b = rearrange(latents_b, "F C 1 H W -> 1 C F H W")
            if latents_out is None:
                latents_out = latents_b
            else:
                latents_out = torch.cat([latents_out, latents_b], dim=0)

            del latents_frames, latents_b
            if prefer_cuda:
                torch.cuda.empty_cache()

        return latents_out

    else:
        raise NotImplementedError("Currently only support encode as first frame.")

@torch.amp.autocast('cuda', dtype=torch.float32)
def decode_images(latents: Float[Tensor, "B C F H W"], decode_as_first: bool = False):
    """
    Decode latents back to images using VAE.
    :param latents: Input latents with shape [B, C, F, H, W].
    :param decode_as_first: Whether to decode all frames as the first frame.
    :return: Decoded images with shape [B, C, F*Nv, H*8, W*8].
    """
    global VAE, LATENTS_MEAN, LATENTS_STD

    if decode_as_first:
        B, C, F, H, W = latents.shape
        out_list = []
        chunk = max(1, int(cfg.decode_frame_chunk))

        prefer_cuda = torch.cuda.is_available()
        target_device = torch.device("cuda") if prefer_cuda else torch.device("cpu")

        for b in range(B):
            imgs_frames = []
            for f0 in range(0, F, chunk):
                f1 = min(F, f0 + chunk)
                lat_chunk = latents[b : b + 1, :, f0:f1].to(target_device, non_blocking=True)
                # invert normalization on same device
                lat_mean = LATENTS_MEAN.to(target_device, non_blocking=True)
                lat_std = LATENTS_STD.to(target_device, non_blocking=True)
                lat_chunk = lat_chunk.to(VAE.dtype)
                lat_chunk = lat_chunk / lat_std + lat_mean
                lat_chunk = rearrange(lat_chunk, "B C F H W -> (B F) C 1 H W")

                imgs = VAE.decode(lat_chunk, return_dict=False)[0]      # [(B F) C Nv H W]
                Fchunk = f1 - f0
                imgs = rearrange(imgs, "(B F) C Nv H W -> B C (F Nv) H W", B=1, F=Fchunk)
                imgs_frames.append(imgs)

                del lat_chunk, imgs, lat_mean, lat_std
                if prefer_cuda:
                    torch.cuda.empty_cache()

            imgs_b = torch.cat(imgs_frames, dim=2)                      # concat on frame dim
            out_list.append(imgs_b)
            del imgs_frames, imgs_b
            if prefer_cuda:
                torch.cuda.empty_cache()

        images = torch.cat(out_list, dim=0)
    else:
        raise NotImplementedError("Currently only support decode as first frame.")

    return images

def convert_img_to_tensor(image: Image.Image, device="cuda") -> Float[Tensor, "H W C"]:
    """
    Convert a PIL Image to a tensor. If Image is RGBA, mask it with black background using a-channel mask.
    :param image: PIL Image to convert. [0, 255]
    :param device: Target device for the tensor.
    :return: Tensor representation of the image. [0.0, 1.0], still [H, W, C]
    """
    # Convert to RGBA to ensure alpha channel exists
    image = image.convert("RGBA")
    np_img = np.array(image)
    rgb = np_img[..., :3]
    alpha = np_img[..., 3:4] / 255.0  # Normalize alpha to [0, 1]
    # Blend with black background using alpha mask
    rgb = rgb * alpha
    rgb = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
    tensor = torch.from_numpy(rgb)
    if device != "cpu":
        tensor = tensor.to(device)
    return tensor
