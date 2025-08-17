import os
import threading
from dataclasses import dataclass
from urllib.parse import urlparse

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

from ..wan.pipeline_wan_t2tex_extra import WanT2TexPipeline
from ..wan.wan_t2tex_transformer_3d_extra import WanT2TexTransformer3DModel

from . import tensor_to_pil
from .file_utils import save_tensor_to_file, load_tensor_from_file

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

def get_seqtex_pipe():
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

        transformer = WanT2TexTransformer3DModel.from_pretrained(
            cfg.seqtex_transformer_path,
            token=auth_token
        )

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
                TEX_PIPE.scheduler.config, shift=cfg.flow_shift
            )
        )
        min_noise_level_index = scheduler.config.num_train_timesteps - cfg.min_noise_level_index
        setattr(TEX_PIPE, "min_noise_level_index", min_noise_level_index)
        min_noise_level_timestep = scheduler.timesteps[min_noise_level_index]
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
            for f0 in range(0, F, chunk):
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

@spaces.GPU(duration=90)
@torch.no_grad
@torch.inference_mode
def generate_texture_b(position_map_path, normal_map_path, position_images_path, normal_images_path, condition_image, text_prompt, selected_view, negative_prompt=None, device="cuda", progress=gr.Progress()):
    """
    Use SeqTex to generate texture for the mesh based on the image condition.
    :param position_images_path: File path to position images tensor
    :param normal_images_path: File path to normal images tensor
    :param condition_image: Image condition generated from the selected view.
    :param text_prompt: Text prompt for texture generation.
    :param selected_view: The view selected for generating the image condition.
    :return: File paths of generated texture map and multi-view frames, and PIL images
    """
    # Keep big tensors on CPU until needed
    position_map = load_tensor_from_file(position_map_path, map_location="cpu")
    normal_map = load_tensor_from_file(normal_map_path, map_location="cpu")
    position_images = load_tensor_from_file(position_images_path, map_location="cpu")
    normal_images = load_tensor_from_file(normal_images_path, map_location="cpu")

    progress(0, desc="Loading SeqTex pipeline...")
    tex_pipe = get_seqtex_pipe()
    progress(0.2, desc="SeqTex pipeline loaded successfully.")

    view_id_map = {
        "First View": 0,
        "Second View": 1,
        "Third View": 2,
        "Fourth View": 3
    }
    view_id = view_id_map[selected_view]

    # --- Encode position/normal sequentially to reduce peak VRAM (identical results) ---
    progress(0.3, desc="Encoding position and normal images (sequentially)...")
    nat_pos_seq = position_images.unsqueeze(0)  # [1, F, H, W, C]
    nat_pos_latents = encode_images(nat_pos_seq, encode_as_first=True)  # [1, C, F, H/8, W/8]
    del nat_pos_seq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nat_norm_seq = normal_images.unsqueeze(0)
    nat_norm_latents = encode_images(nat_norm_seq, encode_as_first=True)
    del nat_norm_seq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    uv_pos_seq = position_map.unsqueeze(0)      # [1, F', H', W', C]
    uv_pos_latents = encode_images(uv_pos_seq, encode_as_first=True)
    del uv_pos_seq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    uv_norm_seq = normal_map.unsqueeze(0)
    uv_norm_latents = encode_images(uv_norm_seq, encode_as_first=True)
    del uv_norm_seq
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nat_geo_latents = torch.cat([nat_pos_latents, nat_norm_latents], dim=1)
    uv_geo_latents = torch.cat([uv_pos_latents, uv_norm_latents], dim=1)

    del nat_pos_latents, nat_norm_latents, uv_pos_latents, uv_norm_latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cond_model_latents = (nat_geo_latents, uv_geo_latents)

    num_frames = cfg.num_views * (2 ** sum(tex_pipe.vae.config.temperal_downsample))
    uv_num_frames = cfg.uv_num_views * (2 ** sum(tex_pipe.vae.config.temperal_downsample))
    
    progress(0.4, desc="Encoding condition image...")
    if isinstance(condition_image, Image.Image):
        print(f"converting and resizing image...")
        condition_image = condition_image.resize((cfg.mv_width, cfg.mv_height), Image.LANCZOS)
        # Convert PIL Image to tensor; keep on CPU until VAE time
        condition_image = convert_img_to_tensor(condition_image, device="cpu")
        condition_image = condition_image.unsqueeze(0)
    else:
        print(f"could not find input ref image")
    gt_latents = (encode_images(condition_image, encode_as_first=True), None)

    del condition_image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress(0.5, desc="Generating texture with SeqTex...")
    latents = tex_pipe(
        prompt=text_prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        generator=torch.Generator(device=device).manual_seed(cfg.eval_seed),
        num_inference_steps=cfg.eval_num_inference_steps,
        guidance_scale=cfg.eval_guidance_scale,
        height=cfg.mv_height,
        width=cfg.mv_width,
        output_type="latent",

        cond_model_latents=cond_model_latents,
        # mask_indices=test_mask_indices,
        uv_height=cfg.uv_height,
        uv_width=cfg.uv_width,
        uv_num_frames=uv_num_frames,
        treat_as_first=True,
        gt_condition=gt_latents,
        inference_img_cond_frame=view_id,
        use_qk_geometry=True,
        task_type="img2tex", # img2tex
        progress=progress,
    ).frames

    # Free large inputs to reclaim VRAM before decode
    del cond_model_latents, gt_latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mv_latents, uv_latents = latents
    
    progress(0.9, desc="Decoding generated latents to images...")
    mv_frames = decode_images(mv_latents, decode_as_first=True) # B C 4 H W
    del mv_latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    uv_frames = decode_images(uv_latents, decode_as_first=True) # B C 1 H W
    del uv_latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    uv_map_pred = uv_frames[:, :, -1, ...]
    uv_map_pred.squeeze_(0)
    mv_out = rearrange(mv_frames[:, :, :cfg.num_views, ...], "B C (F N) H W -> N C (B H) (F W)", N=1)[0]

    mv_out = torch.clamp(mv_out, 0.0, 1.0)
    uv_map_pred = torch.clamp(uv_map_pred, 0.0, 1.0)

    del uv_frames, mv_frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress(1, desc="Texture generated successfully.")
    uv_map_pred_path = save_tensor_to_file(uv_map_pred, prefix="uv_map_pred")

    uv_map_pred_cpu = uv_map_pred.cpu()
    mv_out_cpu = mv_out.cpu()
    del uv_map_pred, mv_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return uv_map_pred_path, tensor_to_pil(uv_map_pred_cpu, normalize=False), tensor_to_pil(mv_out_cpu, normalize=False), "Step 3: Texture generated successfully."
