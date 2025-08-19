import tempfile

import gradio as gr
import numpy as np
import spaces
import torch
import trimesh
import xatlas
from PIL import Image

from .render_utils import (get_mvp_matrix, get_pure_texture, render_geo_map,
                           render_geo_views_tensor, render_views, setup_lights)
from .file_utils import save_tensor_to_file


class Mesh:
    def __init__(self, mesh_path=None, uv_tool="xAtlas", device='cuda', progress=gr.Progress()):
        self._device = device
        if mesh_path is not None:
            self._parts = {}
            
            if mesh_path.endswith('.obj'):
                progress(0., f"Loading mesh in .obj format...")
                mesh_data = trimesh.load(mesh_path, process=False)
                
                if isinstance(mesh_data, list):
                    progress(0.1, f"Handling part list...")
                    for i, mesh_part in enumerate(mesh_data):
                        self._add_part_to_parts(f"part_{i}", mesh_part)
                elif isinstance(mesh_data, trimesh.Scene):
                    progress(0.1, f"Handling Scenes...")
                    geometry = mesh_data.geometry
                    if len(geometry) > 0:
                        for key, mesh_part in geometry.items():
                            self._add_part_to_parts(key, mesh_part)
                    else:
                        raise ValueError("Empty scene, no mesh data found.")
                else:
                    progress(0.1, f"Handling single part...")
                    self._add_part_to_parts("part_0", mesh_data)
            
            elif mesh_path.endswith('.glb'):
                progress(0., f"Loading mesh in .glb format...")
                mesh_loaded = trimesh.load(mesh_path)
                
                if isinstance(mesh_loaded, trimesh.Scene):
                    progress(0.1, f"Handling Scenes...")
                    geometry = mesh_loaded.geometry
                    if len(geometry) > 0:
                        for key, mesh_part in geometry.items():
                            self._add_part_to_parts(key, mesh_part)
                    else:
                        raise ValueError("Empty scene, no mesh data found.")
                else:
                    progress(0.1, f"Handling single part...")
                    self._add_part_to_parts("part_0", mesh_loaded)
            else:
                raise ValueError(f"Unsupported file format: {mesh_path}")
            
            progress(0.2, f"Merging if the mesh have multiple parts.")
            self._merge_parts_internal()
        else:
            raise ValueError("Mesh path cannot be None.")
        self.to(self.device)
        
        self._upside_down_applied = False
        
        if self.has_multi_parts or not self.has_uv:
            progress(0.4, f"Using {uv_tool} for UV parameterization. It may take quite a while (several minutes), if there are many faces. We STRONLY recommend using a mesh with UV parameterization.")
            if uv_tool == "xAtlas":
                self.uv_xatlas_mapping()
            elif uv_tool == "UVAtlas":
                raise NotImplementedError("UVAtlas parameterization is not implemented yet.")
            else:
                raise ValueError("Unsupported UV parameterization tool.")
            print("UV parameterization completed.")
        else:
            progress(0.4, f"The model has SINGLE UV parameterization, no need to reparameterize.")
            self._vmapping = None
    
    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = device
        self._v_pos = self._v_pos.to(device)
        self._t_pos_idx = self._t_pos_idx.to(device)
        if self._v_tex is not None:
            self._v_tex = self._v_tex.to(device)
            self._t_tex_idx = self._t_tex_idx.to(device)
        if hasattr(self, '_vmapping') and self._vmapping is not None:
            self._vmapping = self._vmapping.to(device)
        self._v_normal = self._v_normal.to(device)
        return self

    @property
    def has_multi_parts(self):
        if self._parts is None:
            return False
        return len(self._parts) > 1

    @property
    def v_pos(self):
        return self._v_pos
    
    @v_pos.setter
    def v_pos(self, value):
        self._v_pos = value
    
    @property
    def t_pos_idx(self):
        return self._t_pos_idx
    
    @t_pos_idx.setter
    def t_pos_idx(self, value):
        self._t_pos_idx = value
    
    @property
    def v_tex(self):
        return self._v_tex
    
    @v_tex.setter
    def v_tex(self, value):
        self._v_tex = value
    
    @property
    def t_tex_idx(self):
        return self._t_tex_idx
    
    @t_tex_idx.setter
    def t_tex_idx(self, value):
        self._t_tex_idx = value
    
    @property
    def v_normal(self):
        return self._v_normal
    
    @v_normal.setter
    def v_normal(self, value):
        self._v_normal = value
    
    @property
    def has_uv(self):
        return self.v_tex is not None
    
    def uv_xatlas_mapping(self, size=1024, xatlas_chart_options: dict = {}, xatlas_pack_options: dict = {}):
        atlas = xatlas.Atlas()
        v_pos_np = self.v_pos.detach().cpu().numpy()
        t_pos_idx_np = self.t_pos_idx.cpu().numpy()
        atlas.add_mesh(v_pos_np, t_pos_idx_np)

        co = xatlas.ChartOptions()
        po = xatlas.PackOptions()
        if 'resolution' not in xatlas_pack_options:
            po.resolution = size
        if 'padding' not in xatlas_pack_options:
            po.padding = 2
        for k, v in xatlas_chart_options.items():
            setattr(co, k, v)
        for k, v in xatlas_pack_options.items():
            setattr(po, k, v)
        atlas.generate(co, po)

        vmapping, indices, uvs = atlas.get_mesh(0)
        device = self.v_pos.device
        vmapping = torch.from_numpy(vmapping.astype(np.uint64, casting="same_kind").view(np.int64)).to(device).long()
        uvs = torch.from_numpy(uvs).to(device).float()
        indices = torch.from_numpy(indices.astype(np.uint64, casting="same_kind").view(np.int64)).to(device).long()

        self.v_tex = uvs
        self.t_tex_idx = indices
        self._vmapping = vmapping
    
    def normalize(self):
        vertices = self.v_pos
        bounding_box_max = vertices.max(0)[0]
        bounding_box_min = vertices.min(0)[0]
        mesh_scale = 2.0
        scale = mesh_scale / ((bounding_box_max - bounding_box_min).max() + 1e-6)
        center_offset = (bounding_box_max + bounding_box_min) * 0.5
        self.v_pos = (vertices - center_offset) * scale
    
    def vertex_transform(self):
        pre_normals = self.v_normal
        normals = torch.clone(pre_normals)
        normals[:, 1] = -pre_normals[:, 2]
        normals[:, 2] = pre_normals[:, 1]
        
        pre_vertices = self.v_pos
        vertices = torch.clone(pre_vertices)
        vertices[:, 1] = -pre_vertices[:, 2]
        vertices[:, 2] = pre_vertices[:, 1]
        
        self.v_normal = normals
        self.v_pos = vertices

    def vertex_transform_y2x(self):
        pre_normals = self.v_normal
        normals = torch.clone(pre_normals)
        normals[:, 1] = -pre_normals[:, 0]
        normals[:, 0] = pre_normals[:, 1]
        
        pre_vertices = self.v_pos
        vertices = torch.clone(pre_vertices)
        vertices[:, 1] = -pre_vertices[:, 0]
        vertices[:, 0] = pre_vertices[:, 1]
        
        self.v_normal = normals
        self.v_pos = vertices

    def vertex_transform_z2x(self):
        pre_normals = self.v_normal
        normals = torch.clone(pre_normals)
        normals[:, 2] = -pre_normals[:, 0]
        normals[:, 0] = pre_normals[:, 2]
        
        pre_vertices = self.v_pos
        vertices = torch.clone(pre_vertices)
        vertices[:, 2] = -pre_vertices[:, 0]
        vertices[:, 0] = pre_vertices[:, 2]
        
        self.v_normal = normals
        self.v_pos = vertices

    def vertex_transform_upsidedown(self):
        pre_normals = self.v_normal
        normals = torch.clone(pre_normals)
        normals[:, 2] = -pre_normals[:, 2]

        pre_vertices = self.v_pos
        vertices = torch.clone(pre_vertices)
        vertices[:, 2] = -pre_vertices[:, 2]

        self.v_normal = normals
        self.v_pos = vertices
        
        self._upside_down_applied = True
    
    def _add_part_to_parts(self, key, mesh_part):
        if hasattr(mesh_part, 'vertices') and hasattr(mesh_part, 'faces') and len(mesh_part.vertices) > 0 and len(mesh_part.faces) > 0:
            raw_uv = getattr(mesh_part.visual, 'uv', None)
            processed_v_tex = None
            processed_t_tex_idx = None

            if raw_uv is not None and np.asarray(raw_uv).size > 0 and np.asarray(raw_uv).shape[0] > 0:
                processed_v_tex = torch.tensor(raw_uv, dtype=torch.float32)
                processed_t_tex_idx = torch.tensor(mesh_part.faces, dtype=torch.int32)
            
            self._parts[key] = {
                'v_pos': torch.tensor(mesh_part.vertices, dtype=torch.float32),
                't_pos_idx': torch.tensor(mesh_part.faces, dtype=torch.int32),
                'v_tex': processed_v_tex,
                't_tex_idx': processed_t_tex_idx,
                'v_normal': torch.tensor(mesh_part.vertex_normals, dtype=torch.float32)
            }
    
    def _merge_parts_internal(self):
        if not self._parts:
            raise ValueError("No mesh parts.")
        elif len(self._parts) == 1:
            key = next(iter(self._parts))
            part = self._parts[key]
            self._v_pos = part['v_pos']
            self._t_pos_idx = part['t_pos_idx']
            self._v_tex = part['v_tex']
            self._t_tex_idx = part['t_tex_idx']
            self._v_normal = part['v_normal']
            self._parts = None
            return

        vertices = []
        faces = []
        normals = []
        
        v_count = 0
        
        for key, part in self._parts.items():
            vertices.append(part['v_pos'])
            
            if len(faces) > 0:
                adjusted_faces = part['t_pos_idx'] + v_count
                faces.append(adjusted_faces)
            else:
                faces.append(part['t_pos_idx'])
            
            normals.append(part['v_normal'])
            
            v_count += part['v_pos'].shape[0]
        
        self._parts = None

        self._v_pos = torch.cat(vertices, dim=0)
        self._t_pos_idx = torch.cat(faces, dim=0)
        self._v_normal = torch.cat(normals, dim=0)
        self._v_tex = None
        self._t_tex_idx = None
        self._vmapping = None

    @classmethod
    def export(cls, mesh, save_path=None, texture_map: Image.Image = None):
        assert not mesh.has_multi_parts, "Mesh should be processed and merged to single part"
        assert mesh.has_uv, "Mesh should have UV mapping after processing"
        
        if save_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
            save_path = temp_file.name
            temp_file.close()

        if texture_map is not None:
            if type(texture_map) is np.ndarray:
                texture_map = Image.fromarray(texture_map)
            assert type(texture_map) is Image.Image, "texture_map should be a PIL.Image"
            texture_map = texture_map.transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture_map,
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
        else:
            default_texture = Image.new("RGB", (1024, 1024), (200, 200, 200))
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

        mesh_export = trimesh.Trimesh(vertices=vertices_export, faces=faces, process=False)
        mesh_export.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)

        mesh_export.export(file_obj=save_path, file_type='glb')
        
        return save_path

    @classmethod
    @spaces.GPU(duration=30)
    def process(cls, mesh_file, uv_tool="xAtlas", y2z=True, y2x=False, z2x=False, upside_down=False, img_size=(512, 512), uv_size=(1024, 1024), device='cuda', progress=gr.Progress()):
        mesh: Mesh = cls(mesh_file, uv_tool, device, progress=progress)

        progress(0.7, f"Handling transformation and normalization...")
        if y2z:
            mesh.vertex_transform()
        if y2x:
            mesh.vertex_transform_y2x()
        if z2x:
            mesh.vertex_transform_z2x()
        if upside_down:
            mesh.vertex_transform_upsidedown()
        mesh.normalize()
        
        texture = get_pure_texture(uv_size).to(device)
        lights = None
        mvp_matrix, w2c = get_mvp_matrix(mesh)
        mvp_matrix = mvp_matrix.to(device)
        w2c = w2c.to(device)
        
        progress(0.8, f"Rendering clay model views...")
        print(f"Rendering geometry views...")
        position_images, normal_images, mask_images = render_geo_views_tensor(mesh, mvp_matrix, img_size)
        progress(0.9, f"Rendering geometry maps...")
        print(f"Rendering geometry maps...")
        position_map, normal_map = render_geo_map(mesh)

        progress(1, f"Mesh processing completed.")
        position_map_path = save_tensor_to_file(position_map, prefix="position_map")
        normal_map_path = save_tensor_to_file(normal_map, prefix="normal_map")
        position_images_path = save_tensor_to_file(position_images, prefix="position_images")
        normal_images_path = save_tensor_to_file(normal_images, prefix="normal_images")
        mask_images_path = save_tensor_to_file(mask_images.squeeze(-1), prefix="mask_images")
        w2c_path = save_tensor_to_file(w2c, prefix="w2c")
        mvp_matrix_path = save_tensor_to_file(mvp_matrix, prefix="mvp_matrix")
        return position_map_path, normal_map_path, position_images_path, normal_images_path, mask_images_path, w2c_path, mesh.to("cpu"), mvp_matrix_path, "Mesh processing completed."