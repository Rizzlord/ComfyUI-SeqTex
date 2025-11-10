# SeqTex ComfyUI Wrapper
## [VAST-AI](https://huggingface.co/spaces/VAST-AI) SeqTex: Generate Mesh Textures in Video Sequence
([Arxiv Paper](https://arxiv.org/abs/2507.04285)) ([HuggingFace](https://huggingface.co/spaces/VAST-AI/SeqTex))

**New Features!!!**
 - 10.11.2025: Added SeqTex Project Texture node for re-projecting edited multiview images back to UV space without running the WAN generator
 - 20.08.2025: Fixed seam bleed in normal and position map bake
 - 19.08.2025: Added support for Multi-view Image Input including SDXL workflow example

## Installation

```
git clone https://github.com/Rizzlord/ComfyUI-SeqTex
[path_to_python_embeded]/python.exe -m pip install -r requirements.txt
```

## SeqTex Project Texture

Need to skip the WAN texture generation and simply bake your edited multiview renders back onto the mesh?  
Hook the SeqTex Step 1 mesh plus its `mvp_matrix` / `w2c` tensor paths and your multiview RGB batch into **SeqTex Project Texture**.  
The node replays the exact camera rig with `nvdiffrast`, handles preset-based view trimming (2/4/6/12), optional per-view masks, and outputs a UV texture tensor you can save or feed downstream.

## Authors

[Rizzlord](https://github.com/Rizzlord)
[easymode](https://github.com/Easymode-ai)
