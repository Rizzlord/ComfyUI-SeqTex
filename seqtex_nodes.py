import torch
import numpy as np
from PIL import Image
import os
import gc
import imageio
import spaces
from dataclasses import dataclass
import comfy.utils
import comfy.model_management as mm
import folder_paths
from huggingface_hub import hf_hub_download
from einops import rearrange
from .utils.mesh_utils import Mesh
from .utils.render_utils import get_mvp_matrix, render_geo_map, render_geo_views_tensor, get_pure_texture, render_views
from .utils.image_generation import get_silhouette_image
from .wan.pipeline_wan_t2tex_extra import WanT2TexPipeline
from .wan.wan_t2tex_transformer_3d_extra import WanT2TexTransformer3DModel
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from .utils.controlnet_union import ControlNetModel_Union
from .utils.pipeline_controlnet_union_sd_xl import StableDiffusionXLControlNetUnionPipeline
from .utils.texture_generation import cfg as seqtex_cfg
from .utils.texture_generation import encode_images,  decode_images, convert_img_to_tensor, get_seqtex_pipe
from .utils.file_utils import save_tensor_to_file, load_tensor_from_file
from .utils import tensor_to_pil
import trimesh 
from PIL import ImageDraw

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

seqtex_global_pipe = None

class SeqTex_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_noise_level_index": ("INT", {"default": 15, "min": 0, "max": 999}),
                "flow_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 999.99}),
            }
        }
    RETURN_TYPES = ("SEQTEX_PIPE",)
    FUNCTION = "load_pipe"
    CATEGORY = "SeqTex/Loaders"

    def load_pipe(self, min_noise_level_index, flow_shift):
        global seqtex_global_pipe
        if seqtex_global_pipe is not None:
            print("Using cached SeqTex pipe")
            return (seqtex_global_pipe,)

        seqtex_global_pipe = get_seqtex_pipe(min_noise_level_index=min_noise_level_index, flow_shift=flow_shift)
        return (seqtex_global_pipe,)
    
class SeqTex_Load_Mesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "load_path": ("STRING", {"default": "", "tooltip": "The path of the mesh to load."}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The model of the mesh to load.",)
    
    FUNCTION = "load"
    CATEGORY = "SeqTex"
    DESCRIPTION = "Loads a model from the given path."

    def load(self, load_path, file_format):
        mesh = trimesh.load(load_path, force="mesh", format=file_format, process=False)

        #new_vertices, new_faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
        #mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

        return (mesh,)
    
class SeqTex_Step1_ProcessMesh:
    @classmethod
    @spaces.GPU(duration=90)
    @torch.no_grad
    @torch.inference_mode
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_trimesh": ("TRIMESH",),
                "y2z": ("BOOLEAN", {"default": True}),
                "y2x": ("BOOLEAN", {"default": False}),
                "z2x": ("BOOLEAN", {"default": False}),
                "upside_down": ("BOOLEAN", {"default": False}),
                "uv_size": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "mv_size": ("INT", {"default": 512, "min": 0, "max": 8192}),
                "enable_xatlas": ("BOOLEAN", {"default": False}),
                "enable_uv_preview": ("BOOLEAN", {"default": False}),
                "smooth_normals": ("BOOLEAN", {"default": True}),
                "fix_normals": ("BOOLEAN", {"default": True}),
                "camera_elevation": ("INT", {"default": 25}),
                "camera_lens": ("INT", {"default": 50}),
                "camera_sensor_width": ("INT", {"default": 36})
            },
            "optional": {
                "uv_preview": ("IMAGE"),
            },
        }

    RETURN_TYPES = ( "TRIMESH", "IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ( "trimesh", "uv_preview", "position_map_path", "normal_map_path", "position_images_path",
                    "normal_images_path", "mask_images_path", "w2c_matrix_path", "mvp_matrix_path")
    FUNCTION = "process_mesh"
    CATEGORY = "SeqTex"

    def process_mesh(self, input_trimesh, y2z, y2x, z2x, upside_down, uv_size=1024, mv_size=512, enable_uv_preview=False, enable_xatlas = False, smooth_normals=True, fix_normals=True, camera_elevation=30, camera_lens=50, camera_sensor_width=36):
        
        def smooth_normals_across_uv(mesh):
            """
            Smooth normals across duplicated vertices caused by UV seams.
            mesh: trimesh.Trimesh with attributes vertices, faces, and vertex normals
            """
            verts = mesh.vertices
            normals = mesh.vertex_normals

            # Find unique vertex positions (merge duplicates)
            unique_verts, inverse = np.unique(verts, axis=0, return_inverse=True)

            # Accumulate normals per unique vertex
            smooth_normals = np.zeros_like(unique_verts)
            counts = np.zeros(len(unique_verts), dtype=np.int32)

            for i, idx in enumerate(inverse):
                smooth_normals[idx] += normals[i]
                counts[idx] += 1

            smooth_normals /= counts[:, None]
            smooth_normals = smooth_normals / np.linalg.norm(smooth_normals, axis=1, keepdims=True)

            # Map back to original vertex order
            vertex_normals = smooth_normals[inverse]

            return torch.tensor(vertex_normals, dtype=torch.float32)
    
        if fix_normals == True:
            input_trimesh.fix_normals()

      

        if smooth_normals == True:
            smooth_normals_across_uv(input_trimesh)

        mesh = Mesh.__new__(Mesh)
        mesh._device = 'cuda'
        mesh._parts = None
        mesh._upside_down_applied = False
  
        mesh._v_pos = torch.tensor(input_trimesh.vertices, dtype=torch.float32)
        mesh._t_pos_idx = torch.tensor(input_trimesh.faces, dtype=torch.int32)
        mesh._v_normal = torch.tensor(input_trimesh.vertex_normals, dtype=torch.float32)
        
        raw_uv = getattr(input_trimesh.visual, 'uv', None)
        if raw_uv is not None and np.asarray(raw_uv).size > 0 and np.asarray(raw_uv).shape[0] > 0:
            mesh._v_tex = torch.tensor(raw_uv, dtype=torch.float32)
            mesh._t_tex_idx = torch.tensor(input_trimesh.faces, dtype=torch.int32)
        else:
            mesh._v_tex = None
            mesh._t_tex_idx = None

   
        if mesh._v_tex is None:
            raise AssertionError("Your mesh is missing UVs. Enable xatlas or manually UV your input mesh.")
        
        if enable_xatlas is True:
            print(f"SeqTex Node: Input mesh has no UVs. Running xatlas parameterization...")
            mesh.uv_xatlas_mapping(size=uv_size)
            input_trimesh.visual.uv = mesh.v_tex.cpu().numpy()

        mesh.to(mesh._device)
        if y2z:
            mesh.vertex_transform()
        if y2x:
            mesh.vertex_transform_y2x()
        if z2x:
            mesh.vertex_transform_z2x()
        if upside_down:
            mesh.vertex_transform_upsidedown()


        mesh.normalize()


        
        mvp_matrix, w2c = get_mvp_matrix(mesh, default_elevation=camera_elevation, default_camera_lens=camera_lens, default_camera_sensor_width=camera_sensor_width, width=mv_size, height=mv_size)
        position_images, normal_images, mask_images = render_geo_views_tensor(mesh, mvp_matrix, img_size=(mv_size, mv_size))
        position_map, normal_map = render_geo_map(mesh, map_size=(uv_size,uv_size))

        print(f"Mesh processing completed.")
        position_map_path = save_tensor_to_file(position_map, prefix="position_map")
        normal_map_path = save_tensor_to_file(normal_map, prefix="normal_map")
        position_images_path = save_tensor_to_file(position_images, prefix="position_images")
        normal_images_path = save_tensor_to_file(normal_images, prefix="normal_images")
        mask_images_path = save_tensor_to_file(mask_images.squeeze(-1), prefix="mask_images")
        w2c_path = save_tensor_to_file(w2c, prefix="w2c")
        mvp_matrix_path = save_tensor_to_file(mvp_matrix, prefix="mvp_matrix")

        default_texture = Image.new("RGB", (uv_size, uv_size), (200, 200, 200))
        material = trimesh.visual.texture.SimpleMaterial(image=default_texture)

        if hasattr(mesh, '_vmapping') and mesh._vmapping is not None:
            vertices = mesh.v_pos[mesh._vmapping].cpu().numpy()
            faces = mesh.t_tex_idx.cpu().numpy()
            uvs = mesh.v_tex.cpu().numpy()
        else:
            vertices = mesh.v_pos.cpu().numpy()
            faces = mesh.t_pos_idx.cpu().numpy()
            uvs = mesh.v_tex.cpu().numpy()

        if hasattr(mesh, '_upside_down_applied') and mesh._upside_down_applied:
            faces_corrected = faces.copy()
            faces_corrected[:, [1, 2]] = faces[:, [2, 1]]
            faces = faces_corrected

        vertices_export = vertices.copy()
        vertices_export[:, 1] = vertices[:, 2]
        vertices_export[:, 2] = -vertices[:, 1]
        print(f"SeqTex Step 1: Mesh processing complete.")

        mesh_export = trimesh.Trimesh(vertices=vertices_export, faces=faces, process=False)
        mesh_export.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

        uv_preview_image = Image.new("RGB", (uv_size, uv_size), (0, 0, 0))
        def fast_uv_preview(uvs, faces, image_size=(512,512), background=(0,0,0), line_color=(255,255,255), line_width=1):
            
            w, h = image_size
            img = Image.new("RGB", image_size, color=background)
            draw = ImageDraw.Draw(img)

            uv_pixels = np.clip((uvs * np.array([w-1, h-1])).astype(int), 0, [w-1, h-1])

            for f in faces:
                coords = [tuple(uv_pixels[i]) for i in f]
                coords.append(coords[0])
                draw.line(coords, fill=line_color, width=line_width)

            return img

        if mesh.has_uv and enable_uv_preview == True:
            uv_preview_image = fast_uv_preview(uvs, faces, image_size=(uv_size,uv_size))
            print(f"Done Generating atlas")
                
        uv_preview_image = pil2tensor(uv_preview_image.transpose(Image.FLIP_TOP_BOTTOM))

        return (mesh_export, uv_preview_image, position_map_path, normal_map_path, position_images_path,
                    normal_images_path, mask_images_path, w2c_path, mvp_matrix_path)


class SeqTex_Step4_SaveMesh:
    @classmethod
    @spaces.GPU(duration=90)
    @torch.no_grad
    @torch.inference_mode
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh_input": ("TRIMESH",),
                "texture_map": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "SeqTex_"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ( "STRING",)
    RETURN_NAMES = ( "save_path",)
    FUNCTION = "save_mesh"
    CATEGORY = "SeqTex"
    OUTPUT_NODE = True

    def save_mesh(self, trimesh_input, texture_map, file_format, filename_prefix, save_file=False):

        import tempfile
        #assert not trimesh_input.has_multi_parts, "Mesh should be processed and merged to single part"
        #assert trimesh_input.has_uv, "Mesh should have UV mapping after processing"
        
        #if save_path is None:
        #    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
        #    save_path = temp_file.name
        #    temp_file.close()
        print(f"Step 4. Combining mesh with image result")
        if texture_map is not None:
            texture_map = tensor2pil(texture_map)
            if type(texture_map) is np.ndarray:
                texture_map = Image.fromarray(texture_map)
            assert type(texture_map) is Image.Image, "texture_map should be a PIL.Image"
            #texture_map = texture_map.transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_map,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
            print(f"OK Added material")
        else:
            default_texture = Image.new("RGB", (1024, 1024), (200, 200, 200))
            material = trimesh.visual.texture.SimpleMaterial(image=default_texture)

        # Check if UVs exist
        if hasattr(trimesh_input.visual, "uv") and trimesh_input.visual.uv is not None:
            uv_coords = trimesh_input.visual.uv   # numpy array (N, 2)
            texture_size = np.array(texture_map.size)   # (W, H)
            uvs_pixel = uv_coords * texture_size

            # Assign back (if you want them in pixel space — but note many renderers expect normalized [0,1])
            trimesh_input.visual.uv = uvs_pixel

            print("UVs shape:", uv_coords.shape)
            print(uv_coords[:5])  # first few coords
        else:
            assert(f"mesh has no UV")
        
        trimesh_input.visual = trimesh.visual.TextureVisuals(uv=uv_coords,material=material)

        #merge vertices
        trimesh_input.merge_vertices()

        from pathlib import Path

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh_input.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        else:
            temp_file = Path(full_output_folder, f'_.{file_format}')
            trimesh_input.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f'_.{file_format}'
        
        return (str(relative_path), )
    
class SeqTex_Step2_GenerateCondition:
    @classmethod
    @spaces.GPU()
    def INPUT_TYPES(s):
        return {
            "required": {
                "position_images": ("STRING",),
                "normal_images": ("STRING",),
                "mask_images": ("STRING",),
                "w2c_matrix": ("STRING",),
                "prompt": ("STRING", {"default": "a birdhouse"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "selected_view": (["First View", "Second View", "Third View", "Fourth View"],),
                "steps": ("INT", {"default": 50, "min": 0, "max": 100}),
                "edge_refinement": ("BOOLEAN", {"default": False}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "model_type": (["SDXL"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_condition"
    CATEGORY = "SeqTex"

    def _create_sdxl_pipe(self, model, clip_g, clip_l, vae, controlnet):
        unet = model.model.diffusion_model
        
        text_encoder_one = clip_l.cond_stage_model
        text_encoder_two = clip_g.cond_stage_model
        
        tokenizer_one = clip_l.tokenizer
        tokenizer_two = clip_g.tokenizer
        
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
        vae.first_stage_model.to("cuda")
        unet.to("cuda")
        controlnet.control_model.to("cuda")
        clip_l.cond_stage_model.to("cuda")
        clip_g.cond_stage_model.to("cuda")
        pipe = StableDiffusionXLControlNetUnionPipeline(
            vae=vae.first_stage_model,
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            unet=unet,
            controlnet=controlnet.control_model,
            scheduler=scheduler,
        )
        return pipe

    def generate_condition(self, position_images, normal_images, mask_images, w2c_matrix, prompt, seed, selected_view, steps=50, edge_refinement=False, cpu_offload=False, model_type=0):

        import torch.nn.functional as F
        import cv2
        import threading

        IMG_PIPE = None
        IMG_PIPE_LOCK = threading.Lock()
        FLUX_PIPE = None
        FLUX_PIPE_LOCK = threading.Lock()
        FLUX_SUFFIX = None
        FLUX_NEGATIVE = None
        CPU_OFFLOAD = cpu_offload

        pipe = None
        pbar = comfy.utils.ProgressBar(6)
        if isinstance(position_images, str):
            position_imgs = load_tensor_from_file(position_images, map_location="cuda")
        if isinstance(normal_images, str):
            normal_imgs = load_tensor_from_file(normal_images, map_location="cuda")
        if isinstance(mask_images, str):
            mask_imgs = load_tensor_from_file(mask_images, map_location="cuda")
        if isinstance(w2c_matrix, str):
            w2c = load_tensor_from_file(w2c_matrix, map_location="cuda")

        position_imgs = position_imgs.to("cuda")
        normal_imgs = normal_imgs.to("cuda")
        mask_imgs = mask_imgs.to("cuda")
        w2c = w2c.to("cuda")

        print(f"Handling geometry information...")
        silhouette = get_silhouette_image(position_imgs, normal_imgs, mask_imgs=mask_imgs, w2c=w2c, selected_view=selected_view)
        depth_img = silhouette[0]
        normal_img = silhouette[1]
        mask = silhouette[2]

        def generate_sdxl_condition(depth_img, normal_img, text_prompt, mask, seed=42, edge_refinement=False, image_height=1024, image_width=1024, progress=pbar) -> Image.Image:

            from torchvision.transforms import ToPILImage

            print(f"Loading SDXL pipeline...")
            def get_sdxl_pipe():
                #global IMG_PIPE
                #if IMG_PIPE is not None:
                #    return IMG_PIPE
                print(f"First called, loading SDXL pipeline... It may take about 20 seconds.")
                with IMG_PIPE_LOCK:
                    #if IMG_PIPE is not None:
                    #    return IMG_PIPE
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

            pipeline = get_sdxl_pipe()
            print(f"SDXL pipeline loaded successfully.")

            positive_prompt = text_prompt + ", photo-realistic style, high quality, 8K, highly detailed texture, soft diffuse lighting, uniform lighting, flat lighting, even illumination, matte surface, low contrast, uniform color, foreground"
            negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, harsh lighting, high contrast, bright highlights, specular reflections, shiny surface, glossy, reflective, strong shadows, dramatic lighting, spotlight, direct sunlight, glare, bloom, lens flare'
            
            img_generation_resolution = 1024
            image = pipeline(prompt=[positive_prompt]*1,
                        image_list=[0, depth_img, 0, 0, normal_img, 0], 
                        negative_prompt=[negative_prompt]*1,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        width=img_generation_resolution, 
                        height=img_generation_resolution,
                        num_inference_steps=steps,
                        union_control=True,
                        union_control_type=torch.Tensor([0, 1, 0, 0, 1, 0]).to("cuda"),
                        progress=progress,
                    ).images[0]
            print(f"Condition tensor generated successfully.")
            
            rgb_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1).unsqueeze(0).to(pipeline.device)
            mask_tensor = torch.from_numpy(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(pipeline.device)
            mask_tensor = mask_tensor / 255.0

            rgb_tensor = F.interpolate(rgb_tensor, (image_height, image_width), mode="bicubic", align_corners=False)
            mask_tensor = F.interpolate(mask_tensor, (image_height, image_width), mode="bicubic", align_corners=False)
            
            if edge_refinement:
                rgb_tensor_cuda = rgb_tensor.to("cuda")
                mask_tensor_cuda = mask_tensor.to("cuda")
  
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
            
                rgb_tensor_cuda = refine_image_edges(rgb_tensor_cuda, mask_tensor_cuda)
                rgb_tensor = rgb_tensor_cuda.to(pipeline.device)
            
            background_tensor = torch.zeros_like(rgb_tensor)
            rgb_tensor = torch.lerp(background_tensor, rgb_tensor, mask_tensor)
            rgb_tensor = rearrange(rgb_tensor, "1 C H W -> C H W")
            rgb_tensor = rgb_tensor / 255.
            to_img = ToPILImage()
            condition_image = to_img(rgb_tensor.cpu())

            print(f"Condition image generated successfully.")
            return (condition_image )
        try:
            if model_type == "SDXL":
                model = "SDXL"
            else:
                model = "FLUX"
                
            if model == "SDXL":
                condition = generate_sdxl_condition(depth_img, normal_img, prompt, mask, seed, edge_refinement=edge_refinement, progress=pbar)
                condition = pil2tensor(condition)
                return (condition, "SDXL condition generated successfully.")
            elif model == "FLUX":
                raise NotImplementedError("FLUX model not supported in HF space, please delete it and use it locally")
                condition = generate_flux_condition(depth_img, prompt, mask, seed, edge_refinement=True, progress=pbar)
                return condition, "FLUX condition generated successfully (depth-only control)."
            else:
                raise ValueError(f"Unsupported image generation model type: {model}. Supported models: 'SDXL', 'FLUX'.")
        finally:
            torch.cuda.empty_cache()

class SeqTex_TensorsToImages:
    @classmethod
    @spaces.GPU()
    def INPUT_TYPES(s):
        return {
            "required": {
                "position_map_path": ("STRING",),
                "normal_map_path": ("STRING",),
                "position_images_path": ("STRING",),
                "normal_images_path": ("STRING",),
                "mask_images_path": ("STRING",),
                "w2c_matrix_path": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE","MASK","IMAGE","IMAGE")
    RETURN_NAMES = ("normal_mv_images", "depth_mv_images", "mask_mv_images","position_map", "normal_map")
    FUNCTION = "get_images"
    CATEGORY = "SeqTex"

    def get_images(self, position_map_path, normal_map_path, position_images_path, normal_images_path,mask_images_path,w2c_matrix_path):

        import torch.nn.functional as F
        import threading

        if isinstance(position_images_path, str):
            position_map = load_tensor_from_file(position_map_path, map_location="cuda")
        if isinstance(normal_map_path, str):
            normal_map = load_tensor_from_file(normal_map_path, map_location="cuda")
        if isinstance(position_images_path, str):
            position_imgs = load_tensor_from_file(position_images_path, map_location="cuda")
        if isinstance(normal_images_path, str):
            normal_imgs = load_tensor_from_file(normal_images_path, map_location="cuda")
        if isinstance(mask_images_path, str):
            mask_imgs = load_tensor_from_file(mask_images_path, map_location="cuda")
        if isinstance(w2c_matrix_path, str):
            w2c = load_tensor_from_file(w2c_matrix_path, map_location="cuda")

        position_imgs = position_imgs.to("cuda")
        normal_imgs = normal_imgs.to("cuda")
        mask_imgs = mask_imgs.to("cuda")
        w2c = w2c.to("cuda")

        view_id_map = {
            0: "First View",
            1: "Second View",
            2: "Third View",
            3: "Fourth View"
        }

        depth_images = []
        normal_images = []
        mask_images = []

        for i in range(4):
            view_id = str(view_id_map[i])

            silhouette = get_silhouette_image(position_imgs, normal_imgs, mask_imgs=mask_imgs, w2c=w2c, selected_view=view_id)
            depth_img = pil2tensor(silhouette[0])
            depth_images.append(depth_img)
            normal_img = pil2tensor(silhouette[1])
            normal_images.append(normal_img)
            mask_img = pil2tensor(silhouette[2])
            mask_images.append(mask_img)

        normal_images = torch.cat(normal_images, dim=0)
        depth_images  = torch.cat(depth_images,  dim=0)
        mask_images  = torch.cat(mask_images,  dim=0)
        position_map = pil2tensor(tensor2pil(position_map).transpose(Image.FLIP_TOP_BOTTOM))
        normal_map_img = tensor2pil(normal_map).transpose(Image.FLIP_TOP_BOTTOM)
    
        normal_map = pil2tensor(normal_map_img)
        return normal_images, depth_images, mask_images, position_map, normal_map,

class SeqTex_Step3_GenerateTexture:
    @classmethod
    @spaces.GPU(duration=90)
    @torch.no_grad
    @torch.inference_mode
    def INPUT_TYPES(s):
        return {
            "required": {
                "seqtex_pipe": ("SEQTEX_PIPE",),
                "position_map_path": ("STRING",),
                "normal_map_path": ("STRING",),
                "position_images_path": ("STRING",),
                "normal_images_path": ("STRING",),
                "condition_image_front": ("IMAGE",),
                "text_prompt": ("STRING", {"default": "a birdhouse"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 0.0, "max": 100.0}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0}),
                "uv_size": ("INT", {"default": 1024, "min": 0, "max": 8192}),
                "mv_size": ("INT", {"default": 512, "min": 0, "max": 8192}),
                "num_views": ("INT", {"default": 4, "min": 0, "max": 256}),
                "uv_num_views":("INT", {"default": 1, "min": 0, "max": 256}),
                "task": (["img2tex", "geo+mv2tex"], {"default": "img2tex"}),
                "selected_view": (["First View", "Second View", "Third View", "Fourth View"],),
                #"cpu_offload": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "condition_image_left": ("IMAGE", {"default": None}),
                "condition_image_back": ("IMAGE", {"default": None}),
                "condition_image_right": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("uv_texture", "mv_preview",)
    FUNCTION = "generate_texture"
    CATEGORY = "SeqTex"

    def generate_texture(self, seqtex_pipe, position_map_path, normal_map_path, position_images_path, normal_images_path, condition_image_front, text_prompt, seed, selected_view, condition_image_left=None, condition_image_back=None, condition_image_right=None, negative_prompt=None, steps=30, guidance_scale=1.0, uv_size=1024, mv_size=512, num_views=4, uv_num_views=1, task="img2tex"):
        
        device="cuda"
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

        print(f"Loading SeqTex pipeline...")
        tex_pipe = seqtex_pipe
        print(f"SeqTex pipeline loaded successfully.")

        view_id_map = {
            "First View": 0,
            "Second View": 1,
            "Third View": 2,
            "Fourth View": 3
        }
        view_id = view_id_map[selected_view]
        #condition_image = tensor2pil(condition_image)
        
        # --- Encode position/normal sequentially to reduce peak VRAM (identical results) ---
        print(f"Encoding position and normal images (sequentially)...")
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

        num_frames = num_views * (2 ** sum(tex_pipe.vae.config.temperal_downsample))
        uv_num_frames = uv_num_views * (2 ** sum(tex_pipe.vae.config.temperal_downsample))
        
        condition_images_list = []

        if task != "img2tex":
            # ensure all 4 images are provided
            assert all(img is not None for img in [
                condition_image_front, condition_image_left, condition_image_back, condition_image_right]
            ), "You must provide all four condition images"

            for img in [condition_image_front, condition_image_left, condition_image_back, condition_image_right]:
                img = tensor2pil(img)
                img = img.resize((mv_size, mv_size), Image.NEAREST)
                img_tensor = convert_img_to_tensor(img, device="cpu").unsqueeze(0).unsqueeze(0)
                condition_images_list.append(img_tensor)

            # free memory
            del condition_image_front, condition_image_left, condition_image_back, condition_image_right

        else:
            if condition_image_front is None:
                raise ValueError("You must provide a front condition image for img2tex task")

            img = tensor2pil(condition_image_front)
            img = img.resize((mv_size, mv_size), Image.NEAREST)
            img_tensor = convert_img_to_tensor(img, device="cpu").unsqueeze(0).unsqueeze(0)
            condition_images_list.append(img_tensor)

            del condition_image_front  # free memory for consistency

        # concatenate once
        condition_images = torch.cat(condition_images_list, dim=0)     
        gt_latents = encode_images(condition_images, encode_as_first=True)

        del condition_images
  
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from tqdm import tqdm

        # total steps should match num_inference_steps or however many calls tex_pipe will make
        with tqdm(total=steps) as pbar:
            def progress_callback(step: int, **kwargs):
                # If tex_pipe calls progress with current step
                # you can set pbar.n = step and refresh
                pbar.n = step
                pbar.refresh()
        
        print(f"Generating texture with SeqTex...")
        latents = tex_pipe(
            prompt=text_prompt,
            negative_prompt=None,
            num_frames=num_frames,
            generator=torch.Generator(device=device).manual_seed(seed),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=mv_size,
            width=mv_size,
            output_type="latent",
            cond_model_latents=cond_model_latents,
            # mask_indices=test_mask_indices,
            uv_height=uv_size,
            uv_width=uv_size,
            uv_num_frames=uv_num_frames,
            treat_as_first=True,
            gt_condition=gt_latents,
            inference_img_cond_frame=view_id,
            use_qk_geometry=True,
            max_sequence_length=1024,
            task_type=task,#"geo+mv2tex", # img2tex
            progress=progress_callback,
        ).frames

        # Free large inputs to reclaim VRAM before decode
        del cond_model_latents, gt_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mv_latents, uv_latents = latents
        
        print(f"Decoding generated latents to images...")
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
        mv_out = rearrange(mv_frames[:, :, :num_views, ...], "B C (F N) H W -> N C (B H) (F W)", N=1)[0]

        mv_out = torch.clamp(mv_out, 0.0, 1.0)
        uv_map_pred = torch.clamp(uv_map_pred, 0.0, 1.0)

        del uv_frames, mv_frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Texture generated successfully.")
        uv_map_pred_path = save_tensor_to_file(uv_map_pred, prefix="uv_map_pred")
        uv_map_pred_cpu = uv_map_pred.cpu()

        mv_out_cpu = mv_out.cpu()
        del uv_map_pred, mv_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        uv_map_pred_cpu = tensor_to_pil(uv_map_pred_cpu, normalize=False)
        mv_out_cpu = tensor_to_pil(mv_out_cpu, normalize=False)

        uv_map_pred_cpu = uv_map_pred_cpu.transpose(Image.FLIP_TOP_BOTTOM)

        uv_map_pred_cpu = pil2tensor(uv_map_pred_cpu)
        mv_out_cpu = pil2tensor(mv_out_cpu)


        return uv_map_pred_cpu, mv_out_cpu,


NODE_CLASS_MAPPINGS = {
    "SeqTex_TensorsToImages": SeqTex_TensorsToImages,
    "SeqTex_Load_Mesh": SeqTex_Load_Mesh,
    "SeqTex_Loader": SeqTex_Loader,
    "SeqTex_Step1_ProcessMesh": SeqTex_Step1_ProcessMesh,
    "SeqTex_Step2_GenerateCondition": SeqTex_Step2_GenerateCondition,
    "SeqTex_Step3_GenerateTexture": SeqTex_Step3_GenerateTexture,
    "SeqTex_Step4_SaveMesh": SeqTex_Step4_SaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeqTex_TensorsToImages": "SeqTex Tensors to MV",
    "SeqTex_Load_Mesh": "SeqTex Load Mesh",
    "SeqTex_Loader": "SeqTex Loader",
    "SeqTex_Step1_ProcessMesh": "SeqTex Step 1: Process Mesh",
    "SeqTex_Step2_GenerateCondition": "SeqTex Step 2: Generate Condition",
    "SeqTex_Step3_GenerateTexture": "SeqTex Step 3: Generate Texture",
    "SeqTex_Step4_SaveMesh": "SeqTex Step 4: Apply Texture to Trimesh"
}