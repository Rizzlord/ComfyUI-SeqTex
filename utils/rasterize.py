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
        pos = pos.to(device="cuda", dtype=torch.float32)
        tri = tri.to(device="cuda", dtype=torch.int32)
        print(f"pos tri on CUDA")
        return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

    def rasterize_one(
        self,
        pos: Float[Tensor, "Nv 4"],
        tri: Integer[Tensor, "Nf 3"],
        resolution: Union[int, Tuple[int, int]],
    ):
        pos = pos.to(device="cuda", dtype=torch.float32)
        tri = tri.to(device="cuda", dtype=torch.int32)
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
        
        pos = pos.to(device="cuda", dtype=torch.float32)
        tri = tri.to(device="cuda", dtype=torch.int32)
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
        tri = tri.to(device="cuda", dtype=torch.int32)
        rast = rast.to(device="cuda")
        attr = attr.to("cuda", dtype=torch.float32)
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
        tri = tri.to(device="cuda", dtype=torch.int32)
        attr = attr.to("cuda", dtype=torch.float32)
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
    mesh = mesh.to(device="cpu")
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

def rasterize_position_and_normal_maps(ctx, mesh, rasterize_height, rasterize_width):
    device = ctx.device
    # Convert mesh data to torch tensors
    mesh_v = mesh.v_pos.to(device)
    mesh_f = mesh.t_pos_idx.to(device)
    uvs_tensor = mesh._v_tex.to(device)
    indices_tensor = mesh._t_tex_idx.to(device)
    normal_v = mesh.v_normal.to(device).contiguous()

    # Interpolate mesh data
    uv_clip = uvs_tensor[None, ...] * 2.0 - 1.0
    uv_clip_padded = torch.cat((uv_clip, torch.zeros_like(uv_clip[..., :1]), torch.ones_like(uv_clip[..., :1])), dim=-1)
    rasterized_output, _ = ctx.rasterize(uv_clip_padded, indices_tensor.int(), (rasterize_height, rasterize_width))

    # Interpolate positions.
    position_map, _ = ctx.interpolate_one(mesh_v, rasterized_output, mesh_f.int())
    normal_map, _ = ctx.interpolate_one(normal_v, rasterized_output, mesh_f.int())
    rasterization_mask = rasterized_output[..., 3:4] > 0

    return position_map, normal_map, rasterization_mask