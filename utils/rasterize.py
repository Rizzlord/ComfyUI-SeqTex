# This file uses nvdiffrast library, which is licensed under the NVIDIA Source Code License (1-Way Commercial).
# nvdiffrast is available for non-commercial use (research or evaluation purposes only).
# For commercial use, please contact NVIDIA for licensing: https://www.nvidia.com/en-us/research/inquiries/
#
# nvdiffrast copyright: Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
# Full license: https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt

from typing import Tuple, Union

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float, Integer
from torch import Tensor


class NVDiffRasterizerContext:
    def __init__(self, context_type: str, device) -> None:
        self.device = device
        self.ctx = self.initialize_context(context_type, device)

    def init(self, context_type: str, device) -> None:
        self.device = device
        self.ctx = self.initialize_context(context_type, device)    
        
        
    def initialize_context(
        self, context_type: str, device
    ) -> Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]:
        if context_type == "gl":
            return dr.RasterizeGLContext(device=device)
        elif context_type == "cuda":
            return dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f"Unknown rasterizer context type: {context_type}")

    def vertex_transform(
        self, verts: Float[Tensor, "Nv 3"], mvp_mtx: Float[Tensor, "B 4 4"]
    ) -> Float[Tensor, "B Nv 4"]:
        with torch.amp.autocast("cuda", enabled=False):
            verts_homo = torch.cat(
                [verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1
            )
            verts_clip = torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))
        return verts_clip

    def rasterize(
        self,
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        # rasterize in instance mode (single topology)
        pos = pos.to(device="cuda")
        tri = tri.to(device="cuda")
        print(f"pos tri on CUDA")
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(
        self,
        pos: Float[Tensor, "Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        pos = pos.to(device="cuda" )
        tri = tri.to(device="cuda")
        print(f"pos tri on CUDA")
        # rasterize one single mesh under a single viewpoint
        rast, rast_db = self.rasterize(pos[None, ...], tri, resolution)
        return rast[0], rast_db[0]

    def antialias(
        self,
        color: Float[Tensor, "B H W C"],
        rast: Float[Tensor, "B H W 4"],
        pos: Float[Tensor, "B Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
    ) -> Float[Tensor, "B H W C"]:
        
        pos = pos.to(device="cuda")
        tri = tri.to(device="cuda")
        print(f"pos tri on CUDA")
    
        return dr.antialias(color.float(), rast, pos.float(), tri.int())

    def interpolate(
        self,
        attr: Float[Tensor, "B Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        tri = tri.to(device="cuda")
        rast = rast.to(device="cuda")
        attr = attr.to("cuda")
        print(f"pos tri on CUDA")
        return dr.interpolate(
            attr.float(), rast, tri.int(), rast_db=rast_db, diff_attrs=diff_attrs
        )

    def interpolate_one(
        self,
        attr: Float[Tensor, "Nv C"],
        rast: Float[Tensor, "B H W 4"],
        tri: Integer[Tensor, "Nf 3"],
        rast_db=None,
        diff_attrs=None,
    ) -> Float[Tensor, "B H W C"]:
        tri = tri.to(device="cuda")
        attr = attr.to("cuda")
        print(f"pos tri on CUDA")
        return self.interpolate(attr[None, ...], rast, tri, rast_db, diff_attrs)

def texture_map_to_rgb(tex_map, uv_coordinates):
    return dr.texture(tex_map.float(), uv_coordinates)

def render_rgb_from_texture_mesh_with_mask(
    ctx,
    mesh,
    tex_map: Float[Tensor, "1 H W C"],
    mvp_matrix: Float[Tensor, "batch 4 4"],
    image_height: int,
    image_width: int,
    background_color: Tensor = torch.tensor([0.0, 0.0, 0.0]),
):
    batch_size = mvp_matrix.shape[0]
    tex_map = tex_map.contiguous()
    if tex_map.dim() == 3:
        tex_map = tex_map.unsqueeze(0)  # Add batch dimension if missing
    mesh.to(device="cuda")
    vertex_positions_clip = ctx.vertex_transform(mesh.v_pos, mvp_matrix)
    rasterized_output, _ = ctx.rasterize(vertex_positions_clip, mesh.t_pos_idx, (image_height, image_width))
    mask = rasterized_output[..., 3:] > 0
    mask_antialiased = ctx.antialias(mask.float(), rasterized_output, vertex_positions_clip, mesh.t_pos_idx)

    interpolated_texture_coords, _ = ctx.interpolate_one(mesh._v_tex, rasterized_output, mesh._t_tex_idx)
    rgb_foreground = texture_map_to_rgb(tex_map.float(), interpolated_texture_coords)
    rgb_foreground_batched = torch.zeros(batch_size, image_height, image_width, 3).to(rgb_foreground)
    rgb_background_batched = torch.zeros(batch_size, image_height, image_width, 3).to(rgb_foreground)
    rgb_background_batched += background_color.view(1, 1, 1, 3).to(rgb_foreground)

    selector = mask[..., 0]
    rgb_foreground_batched[selector] = rgb_foreground[selector]

    # Use the anti-aliased mask for blending
    final_rgb = torch.lerp(rgb_background_batched, rgb_foreground_batched, mask_antialiased)
    final_rgb_aa = ctx.antialias(final_rgb, rasterized_output, vertex_positions_clip, mesh.t_pos_idx)

    return final_rgb_aa, selector


def render_geo_from_mesh(ctx, mesh, mvp_matrix, image_height, image_width):
    device = mvp_matrix.device
    mesh = mesh.to(device=device)
    vertex_positions_clip = ctx.vertex_transform(mesh.v_pos, mvp_matrix)
    rasterized_output, _ = ctx.rasterize(vertex_positions_clip, mesh.t_pos_idx, (image_height, image_width))
    interpolated_positions, _ = ctx.interpolate_one(mesh.v_pos, rasterized_output, mesh.t_pos_idx)
    interpolated_normals, _ = ctx.interpolate_one(mesh.v_normal.contiguous(), rasterized_output, mesh.t_pos_idx)

    mask = rasterized_output[..., 3:] > 0
    mask_antialiased = ctx.antialias(mask.float(), rasterized_output, vertex_positions_clip, mesh.t_pos_idx)

    batch_size = mvp_matrix.shape[0]
    rgb_foreground_pos_batched = torch.zeros(batch_size, image_height, image_width, 3).to(interpolated_positions)
    rgb_foreground_norm_batched = torch.zeros(batch_size, image_height, image_width, 3).to(interpolated_positions)
    rgb_background_batched = torch.zeros(batch_size, image_height, image_width, 3).to(interpolated_positions)

    selector = mask[..., 0]
    rgb_foreground_pos_batched[selector] = interpolated_positions[selector]
    rgb_foreground_norm_batched[selector] = interpolated_normals[selector]

    final_pos_rgb = torch.lerp(rgb_background_batched, rgb_foreground_pos_batched, mask_antialiased)
    final_norm_rgb = torch.lerp(rgb_background_batched, rgb_foreground_norm_batched, mask_antialiased)
    final_pos_rgb_aa = ctx.antialias(final_pos_rgb, rasterized_output, vertex_positions_clip, mesh.t_pos_idx)
    final_norm_rgb_aa = ctx.antialias(final_norm_rgb, rasterized_output, vertex_positions_clip, mesh.t_pos_idx)

    return final_pos_rgb_aa, final_norm_rgb_aa, mask_antialiased

import torch
import torch.nn.functional as F
def gaussian_kernel(channels, kernel_size=15, sigma=5.0, device='cuda'):
    """Creates a 2D Gaussian kernel for convolution."""
    import math
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel

def apply_gaussian_blur(tensor, kernel_size=15, sigma=5.0):
    """Apply Gaussian blur via conv2d."""
    channels = tensor.shape[1]
    kernel = gaussian_kernel(channels, kernel_size, sigma, device=tensor.device)
    padding = kernel_size // 2
    return F.conv2d(tensor, kernel, padding=padding, groups=channels)


import torch
import torch.nn.functional as F

def bleed_edges_only(tensor, mask, bleed_radius=5):
    """
    Bleed edges outward by repeating nearest valid pixel, without mixing colors.

    tensor: (B, H, W, C) -> position_map or normal_map
    mask: (B, 1, H, W) boolean tensor where True = valid pixels
    bleed_radius: number of pixels to extend
    """
    B, H, W, C = tensor.shape
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
    mask = mask.bool()

    for _ in range(bleed_radius):
        mask_float = mask.float()  # cast to float for padding

        # Shift tensor in 4 directions
        up    = F.pad(tensor[:, :, :-1, :], (0,0,1,0), mode='replicate')
        down  = F.pad(tensor[:, :, 1:, :], (0,0,0,1), mode='replicate')
        left  = F.pad(tensor[:, :, :, :-1], (1,0,0,0), mode='replicate')
        right = F.pad(tensor[:, :, :, 1:], (0,1,0,0), mode='replicate')

        # Shift mask (use float for padding)
        mask_up    = F.pad(mask_float[:, :, :-1, :], (0,0,1,0), mode='replicate') > 0.5
        mask_down  = F.pad(mask_float[:, :, 1:, :], (0,0,0,1), mode='replicate') > 0.5
        mask_left  = F.pad(mask_float[:, :, :, :-1], (1,0,0,0), mode='replicate') > 0.5
        mask_right = F.pad(mask_float[:, :, :, 1:], (0,1,0,0), mode='replicate') > 0.5

        # Combine neighbors only where current pixel is empty
        update_mask = ~mask
        nearest_val = torch.zeros_like(tensor)

        for neighbor, neighbor_mask in zip([up, down, left, right],
                                           [mask_up, mask_down, mask_left, mask_right]):
            valid = neighbor_mask & update_mask
            nearest_val = torch.where(valid.expand_as(tensor), neighbor, nearest_val)
            mask = mask | neighbor_mask  # expand mask as we fill

        tensor = torch.where(update_mask.expand_as(tensor), nearest_val, tensor)

    return tensor.permute(0, 2, 3, 1)


def compute_smooth_vertex_normals(v_pos: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
    """
    Compute smooth vertex normals by averaging face normals.

    v_pos: [Nv, 3] vertex positions
    t_idx: [Nf, 3] triangle indices
    """
    device = v_pos.device
    Nv = v_pos.shape[0]

    # Get triangle vertices
    v0 = v_pos[t_idx[:, 0]]
    v1 = v_pos[t_idx[:, 1]]
    v2 = v_pos[t_idx[:, 2]]

    # Compute face normals
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # [Nf, 3]

    # Normalize face normals
    face_normals = F.normalize(face_normals, dim=1)

    # Accumulate face normals to vertices
    vertex_normals = torch.zeros_like(v_pos)
    vertex_normals.index_add_(0, t_idx[:, 0], face_normals)
    vertex_normals.index_add_(0, t_idx[:, 1], face_normals)
    vertex_normals.index_add_(0, t_idx[:, 2], face_normals)

    # Normalize vertex normals
    vertex_normals = F.normalize(vertex_normals, dim=1)

    return vertex_normals

def rasterize_position_and_normal_maps(ctx, mesh, rasterize_height, rasterize_width, dilate_pixels=10):

    import numpy as np


    device = ctx.device
    dilate_pixels = 50
    # Convert mesh data to torch tensors
    mesh_v = mesh.v_pos.to(device)
    mesh_f = mesh.t_pos_idx.to(device)
    uvs_tensor = mesh._v_tex.to(device)
    indices_tensor = mesh._t_tex_idx.to(device)
    normal_v = mesh.v_normal.to(device).contiguous()
    normal_v = F.normalize(normal_v, dim=1)



    # Convert UVs to clip space and pad for rasterization
    uv_clip = uvs_tensor[None, ...] * 2.0 - 1.0
    uv_clip_padded = torch.cat(
        (uv_clip, torch.zeros_like(uv_clip[..., :1]), torch.ones_like(uv_clip[..., :1])), dim=-1
    )

    # Rasterize UVs
    rasterized_output, _ = ctx.rasterize(uv_clip_padded, indices_tensor.int(), (rasterize_height, rasterize_width))

    # Interpolate positions and normals
    position_map, _ = ctx.interpolate_one(mesh_v, rasterized_output, mesh_f.int())
    normal_map, _ = ctx.interpolate_one(normal_v, rasterized_output, mesh_f.int())

    # Create rasterization mask
    rasterization_mask = (rasterized_output[..., 3:4] > 0).float()
    if rasterization_mask.dim() == 4:
        rasterization_mask = rasterization_mask.permute(0, 3, 1, 2)  # [B, 1, H, W]
    else:
        rasterization_mask = rasterization_mask.unsqueeze(0).permute(0, 3, 1, 2)
    # Original mask
    orig_mask = rasterization_mask.clone()

    # Bleed edges using the original mask
    if dilate_pixels > 0:
        position_map = bleed_edges_only(position_map, orig_mask, bleed_radius=dilate_pixels)
        normal_map   = bleed_edges_only(normal_map, orig_mask, bleed_radius=dilate_pixels)

    # Optionally dilate mask for later use
    if dilate_pixels > 0:
        kernel = torch.ones(1, 1, 2*dilate_pixels+1, 2*dilate_pixels+1, device=device)
        rasterization_mask = F.conv2d(rasterization_mask, kernel, padding=dilate_pixels)
        rasterization_mask = torch.clamp(rasterization_mask, 0, 1)
    

    return position_map, normal_map, rasterization_mask
