import math
from functools import cache
from typing import Dict, Union

import numpy as np
import spaces
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToPILImage

from .rasterize import (NVDiffRasterizerContext,
                        rasterize_position_and_normal_maps,
                        render_geo_from_mesh,
                        render_rgb_from_texture_mesh_with_mask)
from .file_utils import load_tensor_from_file

_CTX_INSTANCE = None

@spaces.GPU
def get_rasterizer_context():
    global _CTX_INSTANCE
    if _CTX_INSTANCE is None:
        _CTX_INSTANCE = NVDiffRasterizerContext('cuda', 'cuda')
    return _CTX_INSTANCE

def setup_lights():
    raise NotImplementedError("setup_lights function is not implemented yet.")

@spaces.GPU
def render_views(mesh, texture, mvp_matrix, lights=None, img_size=(512, 512)) -> Image.Image:
    if isinstance(texture, str):
        texture = load_tensor_from_file(texture, map_location="cuda")
    if isinstance(mvp_matrix, str):
        mvp_matrix = load_tensor_from_file(mvp_matrix, map_location="cuda")
    mesh = mesh.to("cuda")
    texture = texture.to("cuda")
    mvp_matrix = mvp_matrix.to("cuda")
    
    print("Trying to render views...")
    ctx = get_rasterizer_context()
    if texture.shape[-1] != 3:
        texture = texture.permute(1, 2, 0)
    image_height, image_width = img_size
    rgb_cond, mask = render_rgb_from_texture_mesh_with_mask(
            ctx, mesh, texture, mvp_matrix, image_height, image_width, torch.tensor([0.0, 0.0, 0.0], device=texture.device))
    
    if mvp_matrix.shape[0] == 0:
        return None

    pil_images = []
    for i in range(mvp_matrix.shape[0]):
        rgba_img = torch.cat([rgb_cond[i], mask[i].unsqueeze(-1)], dim=-1)
        rgba_img = (rgba_img * 255).to(torch.uint8)
        rgba_img = rgba_img.cpu().numpy()
        pil_images.append(Image.fromarray(rgba_img, mode='RGBA'))

    if not pil_images:
        return None

    total_width = sum(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)

    concatenated_image = Image.new('RGBA', (total_width, max_height))

    current_x = 0
    for img in pil_images:
        concatenated_image.paste(img, (current_x, 0))
        current_x += img.width
    
    return concatenated_image

@spaces.GPU
def render_geo_views_tensor(mesh, mvp_matrix, img_size=(512, 512)) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = get_rasterizer_context()
    image_height, image_width = img_size
    position_images, normal_images, mask_images = render_geo_from_mesh(ctx, mesh, mvp_matrix, image_height, image_width)
    return position_images, normal_images, mask_images

@spaces.GPU
def render_geo_map(mesh, map_size=(1024, 1024)) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = get_rasterizer_context()
    map_height, map_width = map_size
    position_images, normal_images, mask = rasterize_position_and_normal_maps(ctx, mesh, map_height, map_width)
    return position_images, normal_images

@cache
def get_pure_texture(uv_size, color=(int("0x55", 16), int("0x55", 16), int("0x55", 16))) -> torch.Tensor:
    height, width = uv_size
    
    color = torch.tensor(color, dtype=torch.float32).view(1, 1, 3) / 255.0
    texture = color.repeat(height, width, 1)

    return texture

def get_c2w(
        azimuth_deg,
        elevation_deg,
        camera_distances,):
    assert len(azimuth_deg) == len(elevation_deg) == len(camera_distances)
    n_views = len(azimuth_deg)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w

def camera_strategy_test_4_90deg(
        mesh: Dict,
        num_views: int = 4,
        **kwargs) -> Dict:
    default_elevation = 25
    default_camera_lens = 50
    default_camera_sensor_width = 36
    default_fovy = 2 * np.arctan(default_camera_sensor_width / (2 * default_camera_lens))

    bbox_size = mesh.v_pos.max(dim=0)[0] - mesh.v_pos.min(dim=0)[0]
    distance = default_camera_lens / default_camera_sensor_width * \
               math.sqrt(bbox_size[0] ** 2 + bbox_size[1] ** 2 + bbox_size[2] ** 2)

    all_azimuth_deg = torch.linspace(0, 360.0, num_views + 1)[:num_views] - 90

    all_elevation_deg = torch.full_like(all_azimuth_deg, default_elevation)

    view_idxs = torch.arange(0, num_views)
    azimuth = all_azimuth_deg[view_idxs]
    elevation = all_elevation_deg[view_idxs]
    camera_distances = torch.full_like(elevation, distance)
    c2w = get_c2w(azimuth, elevation, camera_distances)
    
    if c2w.ndim == 2:
        w2c: Float[Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0

    fovy = torch.full_like(azimuth, default_fovy)

    return {
        'cond_sup_view_idxs': view_idxs,
        'cond_sup_c2w': c2w,
        'cond_sup_w2c': w2c,
        'cond_sup_fovy': fovy,
    }

def _get_projection_matrix(
    fovy: Union[float, Float[Tensor, "B"]], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "*B 4 4"]:
    if isinstance(fovy, float):
        proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
        proj_mtx[0, 0] = 1.0 / (math.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[1, 1] = -1.0 / math.tan(
            fovy / 2.0
        )
        proj_mtx[2, 2] = -(far + near) / (far - near)
        proj_mtx[2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[3, 2] = -1.0
    else:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def _get_mvp_matrix(
    c2w: Float[Tensor, "*B 4 4"], proj_mtx: Float[Tensor, "*B 4 4"]
) -> Float[Tensor, "*B 4 4"]:
    if c2w.ndim == 2:
        assert proj_mtx.ndim == 2
        w2c: Float[Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def get_mvp_matrix(mesh, num_views=4, width=512, height=512, strategy="strategy_test_4_90deg"):
    if strategy == "strategy_test_4_90deg":
        camera_info = camera_strategy_test_4_90deg(
            mesh=mesh,
            num_views=num_views,
        )
        cond_sup_fovy = camera_info["cond_sup_fovy"]
        cond_sup_c2w = camera_info["cond_sup_c2w"]
        cond_sup_w2c = camera_info["cond_sup_w2c"]
    else:
        raise ValueError(f"Unsupported camera strategy: {strategy}")
    cond_sup_proj_mtx: Float[Tensor, "B 4 4"] = _get_projection_matrix(
        cond_sup_fovy, width / height, 0.1, 1000.0
    )
    mvp_mtx: Float[Tensor, "B 4 4"] = _get_mvp_matrix(cond_sup_c2w, cond_sup_proj_mtx)
    return mvp_mtx, cond_sup_w2c

@torch.cuda.amp.autocast(enabled=False)
def _get_depth_noraml_map_with_mask(xyz_map, normal_map, mask, w2c, device="cuda", background_color=(0, 0, 0)):
    w2c = w2c.to(device)

    B, Nv, H, W, C = xyz_map.shape
    assert Nv == 1
    xyz_map = rearrange(xyz_map, "B Nv H W C -> (B Nv) (H W) C")
    normal_map = rearrange(normal_map, "B Nv H W C -> (B Nv) (H W) C")
    w2c = rearrange(w2c, "B Nv C1 C2 -> (B Nv) C1 C2")

    B_Nv, N, C = xyz_map.shape
    ones = torch.ones(B_Nv, N, 1, dtype=xyz_map.dtype, device=xyz_map.device)
    homogeneous_xyz = torch.cat([xyz_map, ones], dim=2)
    zeros = torch.zeros(B_Nv, N, 1, dtype=xyz_map.dtype, device=xyz_map.device)
    homogeneous_normal = torch.cat([normal_map, zeros], dim=2)
    
    camera_coords = torch.bmm(homogeneous_xyz, w2c.transpose(1, 2))
    camera_normals = torch.bmm(homogeneous_normal, w2c.transpose(1, 2))

    depth_map = camera_coords[..., 2:3]
    depth_map = rearrange(depth_map, "(B Nv) (H W) 1 -> B Nv H W", B=B, Nv=Nv, H=H, W=W)
    normal_map = camera_normals[..., :3]
    normal_map = rearrange(normal_map, "(B Nv) (H W) c -> B Nv H W c", B=B, Nv=Nv, H=H, W=W)
    assert depth_map.dtype == torch.float32, f"depth_map must be float32, otherwise there will be artifact in controlnet generated pictures, but got {depth_map.dtype}"

    min_depth = depth_map.amin((1,2,3), keepdim=True)
    max_depth = depth_map.amax((1,2,3), keepdim=True)

    depth_map = (depth_map - min_depth) / (max_depth - min_depth + 1e-6)
    
    depth_map = depth_map.repeat(1, 3, 1, 1)
    normal_map = normal_map * 0.5 + 0.5
    normal_map = normal_map[:,0].permute(0, 3, 1, 2)

    rgb_background_batched = torch.tensor(background_color, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    
    mask_for_lerp = mask.squeeze(-1)

    depth_map = torch.lerp(rgb_background_batched, depth_map, mask_for_lerp)
    normal_map = torch.lerp(rgb_background_batched, normal_map, mask_for_lerp)

    return depth_map, normal_map, mask

@spaces.GPU
def get_silhouette_image(position_imgs, normal_imgs, mask_imgs, w2c, selected_view="First View") -> tuple[Image.Image, Image.Image]:
    view_id_map = {
        "First View": 0,
        "Second View": 1,
        "Third View": 2,
        "Fourth View": 3
    }
    view_id = view_id_map[selected_view]
    position_view = position_imgs[view_id: view_id + 1]
    normal_view = normal_imgs[view_id: view_id + 1]
    mask_view = mask_imgs[view_id: view_id + 1]
    w2c = w2c[view_id: view_id + 1]

    depth_img, normal_img, mask = _get_depth_noraml_map_with_mask(
        position_view.unsqueeze(0),
        normal_view.unsqueeze(0),
        mask_view.unsqueeze(0),
        w2c.unsqueeze(0),
    )

    to_img = ToPILImage()
    return to_img(depth_img.squeeze(0)), to_img(normal_img.squeeze(0)), to_img(mask.squeeze())