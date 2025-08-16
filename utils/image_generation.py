import os
import threading

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn.functional as F
# Add FLUX imports
from diffusers import (AutoencoderKL, EulerAncestralDiscreteScheduler,
                       FluxControlNetModel, FluxControlNetPipeline)
from einops import rearrange
from PIL import Image
from torchvision.transforms import ToPILImage

from .controlnet_union import ControlNetModel_Union
from .pipeline_controlnet_union_sd_xl import \
    StableDiffusionXLControlNetUnionPipeline
from .render_utils import get_silhouette_image
from .file_utils import load_tensor_from_file

IMG_PIPE = None
IMG_PIPE_LOCK = threading.Lock()
FLUX_PIPE = None
FLUX_PIPE_LOCK = threading.Lock()
FLUX_SUFFIX = None
FLUX_NEGATIVE = None
CPU_OFFLOAD = False

def get_flux_pipe():
    global FLUX_PIPE, FLUX_SUFFIX, FLUX_NEGATIVE
    if FLUX_PIPE is not None:
        return FLUX_PIPE
    gr.Info("First called, loading FLUX pipeline... It may take about 1 minute.")
    with FLUX_PIPE_LOCK:
        if FLUX_PIPE is not None:
            return FLUX_PIPE
        FLUX_SUFFIX = ", albedo texture, high-quality, 8K, flat shaded, diffuse color only, orthographic view, seamless texture pattern, detailed surface texture."
        FLUX_NEGATIVE = "ugly, PBR, lighting, shadows, highlights, specular, reflections, ambient occlusion, global illumination, bloom, glare, lens flare, glow, shiny, glossy, noise, grain, blurry, bokeh, depth of field."
        base_model = 'black-forest-labs/FLUX.1-dev'
        controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'
        
        controlnet = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
        assert os.environ["SEQTEX_SPACE_TOKEN"] != "", "Please set the SEQTEX_SPACE_TOKEN environment variable with your Hugging Face token, which has access to black-forest-labs/FLUX.1-dev."
        FLUX_PIPE = FluxControlNetPipeline.from_pretrained(
            base_model, 
            controlnet=controlnet, 
            torch_dtype=torch.bfloat16,
            token=os.environ["SEQTEX_SPACE_TOKEN"]
        )
        if CPU_OFFLOAD:
            FLUX_PIPE.enable_model_cpu_offload()
        else:
            FLUX_PIPE.to("cuda")
    return FLUX_PIPE

def get_sdxl_pipe():
    global IMG_PIPE
    if IMG_PIPE is not None:
        return IMG_PIPE
    gr.Info("First called, loading SDXL pipeline... It may take about 20 seconds.")
    with IMG_PIPE_LOCK:
        if IMG_PIPE is not None:
            return IMG_PIPE
        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        controlnet_model = ControlNetModel_Union.from_pretrained("xinsir/controlnet-union-sdxl-1.0", torch_dtype=torch.float16, use_safetensors=True)
        IMG_PIPE = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet_model, 
            vae=vae,
            torch_dtype=torch.float16,
            scheduler=eulera_scheduler,
        )
        if CPU_OFFLOAD:
            IMG_PIPE.enable_model_cpu_offload()
        else:
            IMG_PIPE.to("cuda")
    return IMG_PIPE
    

def generate_sdxl_condition(depth_img, normal_img, text_prompt, mask, seed=42, edge_refinement=False, image_height=1024, image_width=1024, progress=gr.Progress()) -> Image.Image:
    progress(0.1, desc="Loading SDXL pipeline...")
    pipeline = get_sdxl_pipe()
    progress(0.3, desc="SDXL pipeline loaded successfully.")

    positive_prompt = text_prompt + ", photo-realistic style, high quality, 8K, highly detailed texture, soft diffuse lighting, uniform lighting, flat lighting, even illumination, matte surface, low contrast, uniform color, foreground"
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, harsh lighting, high contrast, bright highlights, specular reflections, shiny surface, glossy, reflective, strong shadows, dramatic lighting, spotlight, direct sunlight, glare, bloom, lens flare'
    
    img_generation_resolution = 1024
    image = pipeline(prompt=[positive_prompt]*1,
                image_list=[0, depth_img, 0, 0, normal_img, 0], 
                negative_prompt=[negative_prompt]*1,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                width=img_generation_resolution, 
                height=img_generation_resolution,
                num_inference_steps=50,
                union_control=True,
                union_control_type=torch.Tensor([0, 1, 0, 0, 1, 0]).to("cuda"),
                progress=progress,
            ).images[0]
    progress(0.9, desc="Condition tensor generated successfully.")
    
    rgb_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0).to(pipeline.device)
    mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(pipeline.device)
    mask_tensor = mask_tensor / 255.0

    rgb_tensor = F.interpolate(rgb_tensor, (image_height, image_width), mode="bilinear", align_corners=False)
    mask_tensor = F.interpolate(mask_tensor, (image_height, image_width), mode="bilinear", align_corners=False)
    
    if edge_refinement:
        rgb_tensor_cuda = rgb_tensor.to("cuda")
        mask_tensor_cuda = mask_tensor.to("cuda")
        rgb_tensor_cuda = refine_image_edges(rgb_tensor_cuda, mask_tensor_cuda)
        rgb_tensor = rgb_tensor_cuda.to(pipeline.device)
    
    background_tensor = torch.zeros_like(rgb_tensor)
    rgb_tensor = torch.lerp(background_tensor, rgb_tensor, mask_tensor)
    rgb_tensor = rearrange(rgb_tensor, "1 C H W -> C H W")
    rgb_tensor = rgb_tensor / 255.
    to_img = ToPILImage()
    condition_image = to_img(rgb_tensor.cpu())

    progress(1, desc="Condition image generated successfully.")
    return condition_image

def generate_flux_condition(depth_img, text_prompt, mask, seed=42, edge_refinement=False, image_height=1024, image_width=1024, progress=gr.Progress()) -> Image.Image:
    progress(0.1, desc="Loading FLUX pipeline...")
    pipeline = get_flux_pipe()
    progress(0.3, desc="FLUX pipeline loaded successfully.")

    positive_prompt = text_prompt + FLUX_SUFFIX
    negative_prompt = FLUX_NEGATIVE
    
    width, height = depth_img.size
    
    progress(0.5, desc="Generating image with FLUX (including onload and cpu offload)...")
    
    image = pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        control_image=depth_img,
        width=width,
        height=height,
        controlnet_conditioning_scale=0.8,
        control_guidance_end=0.8,
        num_inference_steps=30,
        guidance_scale=3.5,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    
    progress(0.9, desc="Applying mask and resizing...")
    
    rgb_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0).to("cuda")
    mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to("cuda")
    mask_tensor = mask_tensor / 255.0
    
    rgb_tensor = F.interpolate(rgb_tensor, (image_height, image_width), mode="bilinear", align_corners=False)
    mask_tensor = F.interpolate(mask_tensor, (image_height, image_width), mode="bilinear", align_corners=False)
    
    background_tensor = torch.zeros_like(rgb_tensor)
    if edge_refinement:
        rgb_tensor = refine_image_edges(rgb_tensor, mask_tensor)

    rgb_tensor = torch.lerp(background_tensor, rgb_tensor, mask_tensor)
    
    rgb_tensor = rearrange(rgb_tensor, "1 C H W -> C H W")
    rgb_tensor = rgb_tensor / 255.0
    to_img = ToPILImage()
    condition_image = to_img(rgb_tensor.cpu())

    progress(1, desc="FLUX condition image generated successfully.")
    return condition_image

def refine_image_edges(rgb_tensor, mask_tensor):
    rgb_np = rgb_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    mask_np = mask_tensor.squeeze().cpu().numpy()
    original_mask_np = (mask_np * 255).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    eroded_mask_np = cv2.erode(original_mask_np, kernel, iterations=3)
    
    double_eroded_mask_np = cv2.erode(eroded_mask_np, kernel, iterations=5)
    
    circle_valid_mask_np = cv2.bitwise_xor(eroded_mask_np, double_eroded_mask_np)
    
    circle_valid_mask_3c = cv2.cvtColor(circle_valid_mask_np, cv2.COLOR_GRAY2BGR) / 255.0
    circle_rgb_np = (rgb_np * circle_valid_mask_3c).astype(np.uint8)
    
    dilated_circle_rgb_np = cv2.dilate(circle_rgb_np, kernel, iterations=8)
    
    double_eroded_mask_3c = cv2.cvtColor(double_eroded_mask_np, cv2.COLOR_GRAY2BGR) / 255.0
    
    refined_rgb_np = (rgb_np * double_eroded_mask_3c + 
                     dilated_circle_rgb_np * (1 - double_eroded_mask_3c)).astype(np.uint8)
    
    refined_rgb_tensor = torch.from_numpy(refined_rgb_np).float().permute(2, 0, 1).unsqueeze(0).to("cuda")
    
    return refined_rgb_tensor

@spaces.GPU()
def generate_image_condition(position_imgs, normal_imgs, mask_imgs, w2c, text_prompt, selected_view="First View", seed=42, model="SDXL", edge_refinement=True, progress=gr.Progress()):
    if isinstance(position_imgs, str):
        position_imgs = load_tensor_from_file(position_imgs, map_location="cuda")
    if isinstance(normal_imgs, str):
        normal_imgs = load_tensor_from_file(normal_imgs, map_location="cuda")
    if isinstance(mask_imgs, str):
        mask_imgs = load_tensor_from_file(mask_imgs, map_location="cuda")
    if isinstance(w2c, str):
        w2c = load_tensor_from_file(w2c, map_location="cuda")

    position_imgs = position_imgs.to("cuda")
    normal_imgs = normal_imgs.to("cuda")
    mask_imgs = mask_imgs.to("cuda")
    w2c = w2c.to("cuda")

    progress(0, desc="Handling geometry information...")
    silhouette = get_silhouette_image(position_imgs, normal_imgs, mask_imgs=mask_imgs, w2c=w2c, selected_view=selected_view)
    depth_img = silhouette[0]
    normal_img = silhouette[1]
    mask = silhouette[2]

    try:
        if model == "SDXL":
            condition = generate_sdxl_condition(depth_img, normal_img, text_prompt, mask, seed, edge_refinement=edge_refinement, progress=progress)
            return condition, "SDXL condition generated successfully."
        elif model == "FLUX":
            raise NotImplementedError("FLUX model not supported in HF space, please delete it and use it locally")
            condition = generate_flux_condition(depth_img, text_prompt, mask, seed, edge_refinement=edge_refinement, progress=progress)
            return condition, "FLUX condition generated successfully (depth-only control)."
        else:
            raise ValueError(f"Unsupported image generation model type: {model}. Supported models: 'SDXL', 'FLUX'.")
    finally:
        torch.cuda.empty_cache()