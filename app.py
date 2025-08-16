import os
import gradio as gr

from utils import tensor_to_pil
from utils.image_generation import generate_image_condition, get_flux_pipe, get_sdxl_pipe
from utils.mesh_utils import Mesh
from utils.render_utils import render_views
from utils.texture_generation import generate_texture, get_seqtex_pipe

EXAMPLES = [
    ["examples/birdhouse.glb", True, False, False, False, 42, "First View", "SDXL", False, "A rustic birdhouse featuring a snow-covered roof, wood textures, and two decorative cardinal birds. It has a circular entryway and conveys a winter-themed aesthetic."],
    ["examples/shoe.glb", True, False, False, False, 42, "Second View", "SDXL", False, "Modern sneaker exhibiting a mesh upper and wavy rubber outsole. Features include lacing for adjustability and padded components for comfort. Normal maps emphasize geometric detail."],
    # ["examples/mario.glb", False, False, False, True, 6666, "Third View", "FLUX", True, "Mario, a cartoon character wearing a red cap and blue overalls, with brown hair and a mustache, and white gloves, in a fighting pose. The clothes he wears are not in a reflection mode."],
]
LOAD_FIRST = True
    

with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("# 🎨 SeqTex: Generate Mesh Textures in Video Sequence")
    
    gr.Markdown("""
    ## 🚀 Welcome to SeqTex!
    **SeqTex** is a cutting-edge AI system that generates high-quality textures for 3D meshes using image prompts (here we use image generator to get them from textual prompts). 
    
    Choose to either **try our example models** below or **upload your own 3D mesh** to create stunning textures.
    """)

    gr.Markdown("---")

    gr.Markdown("## 🔧 Step 1: Upload & Process 3D Mesh")
    gr.Markdown("""
    **📋 How to prepare your 3D mesh:**
    - Upload your 3D mesh in **.obj** or **.glb** format
    - **💡 Pro Tip**: 
        - For optimal results, ensure your mesh includes only one part with <span style="color:#e74c3c; font-weight:bold;">UV parameterization</span>
        - Otherwise, we'll combine all parts and generate UV parameterization using *xAtlas* (may take longer for high-poly meshes; may also fail for certain meshes)
    - **⚠️ Important**: We recommend adjusting your model using *Mesh Orientation Adjustments* to be **Z-UP oriented** for best results
    """)
    position_map_tensor_path = gr.State()
    normal_map_tensor_path = gr.State()
    position_images_tensor_path = gr.State()
    normal_images_tensor_path = gr.State()
    mask_images_tensor_path = gr.State()
    w2c_tensor_path = gr.State()
    mesh = gr.State()
    mvp_matrix_tensor_path = gr.State()

    # fixed_texture_map = Image.open("image.webp").convert("RGB")
    # Step 1
    with gr.Row():
        with gr.Column():
            mesh_upload = gr.File(label="📁 Upload 3D Mesh", file_types=[".obj", ".glb"])
            # uv_tool = gr.Radio(["xAtlas", "UVAtlas"], label="UV parameterizer", value="xAtlas")
            
            gr.Markdown("**🔄 Mesh Orientation Adjustments** (if needed):")
            y2z = gr.Checkbox(label="Y → Z Transform", value=False, info="Rotate: Y becomes Z, -Z becomes Y")
            y2x = gr.Checkbox(label="Y → X Transform", value=False, info="Rotate: Y becomes X, -X becomes Y")
            z2x = gr.Checkbox(label="Z → X Transform", value=False, info="Rotate: Z becomes X, -X becomes Z")
            upside_down = gr.Checkbox(label="🔃 Flip Vertically", value=False, info="Fix upside-down mesh orientation")
            step1_button = gr.Button("🔄 Process Mesh & Generate Views", variant="primary")
            step1_progress = gr.Textbox(label="📊 Processing Status", interactive=False)

        with gr.Column():
            model_input = gr.Model3D(label="📐 Processed 3D Model", height=500)

    with gr.Row(equal_height=True):
        rgb_views = gr.Image(label="📷 Generated Views", type="pil", scale=3)
        position_map = gr.Image(label="🗺️ Position Map", type="pil", scale=1)
        normal_map = gr.Image(label="🧭 Normal Map", type="pil", scale=1)

    step1_button.click(
        Mesh.process,
        inputs=[mesh_upload, gr.State("xAtlas"), y2z, y2x, z2x, upside_down],
        outputs=[position_map_tensor_path, normal_map_tensor_path, position_images_tensor_path, normal_images_tensor_path, mask_images_tensor_path, w2c_tensor_path, mesh, mvp_matrix_tensor_path, step1_progress]
    ).success(
        tensor_to_pil,
        inputs=[normal_images_tensor_path, mask_images_tensor_path],
        outputs=[rgb_views]
    ).success(
        tensor_to_pil,
        inputs=[position_map_tensor_path],
        outputs=[position_map]
    ).success(
        tensor_to_pil,
        inputs=[normal_map_tensor_path],
        outputs=[normal_map]
    ).success(
        Mesh.export,
        inputs=[mesh, gr.State(None), gr.State(None)],
        outputs=[model_input]
    )

    # Step 2
    gr.Markdown("---")
    gr.Markdown("## 👁️ Step 2: Select View & Generate Image Condition")
    gr.Markdown("""
    **📋 How to generate image condition:**
    - Your mesh will be rendered from **four viewpoints** (front, back, left, right)
    - Choose **one view** as your image condition
    - Enter a **descriptive text prompt** for the desired texture
    - Select your preferred AI model:
        - <span style="color:#27ae60; font-weight:bold;">🎯 SDXL</span>: Fast generation with depth + normal control, better details (often suffer from wrong highlights)
        - <span style="color:#3498db; font-weight:bold;">⚡ FLUX</span>: ~~High-quality generation with depth control (slower due to CPU offloading). Better work with **Edge Refinement**~~ (Not supported due to the memory limit of HF Space. You can try it locally)
    """)
    with gr.Row():
        with gr.Column():
            img_condition_seed = gr.Number(label="🎲 Random Seed", minimum=0, maximum=9999, step=1, value=42, info="Change for different results")
            selected_view = gr.Radio(["First View", "Second View", "Third View", "Fourth View"], label="📐 Camera View", value="First View", info="Choose which viewpoint to use as reference")
            with gr.Row():
                # model_choice = gr.Radio(["SDXL", "FLUX"], label="🤖 AI Model", value="SDXL", info="SDXL: Fast, depth+normal control | FLUX: High-quality, slower processing")
                model_choice = gr.Radio(["SDXL"], label="🤖 AI Model", value="SDXL", info="SDXL: Fast, depth+normal control | FLUX: High-quality, slower processing (Not supported due to the memory limit of HF Space)")
                edge_refinement = gr.Checkbox(label="✨ Edge Refinement", value=True, info="Smooth boundary artifacts (recommended for delightning highlights in the boundary)")
            text_prompt = gr.Textbox(label="💬 Texture Description", placeholder="Describe the desired texture appearance (e.g., 'rustic wooden surface with weathered paint')", lines=2)
            step2_button = gr.Button("🎯 Generate Image Condition", variant="primary")
            step2_progress = gr.Textbox(label="📊 Generation Status", interactive=False)
            
        with gr.Column():
            condition_image = gr.Image(label="🖼️ Generated Image Condition", type="pil") # , interactive=False

    step2_button.click(
        generate_image_condition,
        inputs=[position_images_tensor_path, normal_images_tensor_path, mask_images_tensor_path, w2c_tensor_path, text_prompt, selected_view, img_condition_seed, model_choice, edge_refinement],
        outputs=[condition_image, step2_progress],
    )

    # Step 3
    gr.Markdown("---")
    gr.Markdown("## 🎨 Step 3: Generate Final Texture")
    gr.Markdown("""
    **📋 How to generate final texture:**
    - The **SeqTex pipeline** will create a complete texture map for your model
    - View the results from multiple angles and download your textured 3D model (the viewport is a little bit dark)
    """)
    texture_map_tensor_path = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            step3_button = gr.Button("🎨 Generate Final Texture", variant="primary")
            step3_progress = gr.Textbox(label="📊 Texture Generation Status", interactive=False)
            texture_map = gr.Image(label="🏆 Generated Texture Map", interactive=False)
        with gr.Column(scale=2):
            rendered_imgs = gr.Image(label="🖼️ Final Rendered Views")
            mv_branch_imgs = gr.Image(label="🖼️ SeqTex Direct Output")
        with gr.Column(scale=1.5):
            model_display = gr.Model3D(label="🏆 Final Textured Model", height=500)
            # model_display = LitModel3D(label="Model with Texture", 
            #                             exposure=30.0, 
            #                             height=500)

    step3_button.click(
        generate_texture,  
        inputs=[position_map_tensor_path, normal_map_tensor_path, position_images_tensor_path, normal_images_tensor_path, condition_image, text_prompt, selected_view],
        outputs=[texture_map_tensor_path, texture_map, mv_branch_imgs, step3_progress],
    ).success(
        render_views,
        inputs=[mesh, texture_map_tensor_path, mvp_matrix_tensor_path],
        outputs=[rendered_imgs]
    ).success(
        Mesh.export,
        inputs=[mesh, gr.State(None), texture_map],
        outputs=[model_display]
    )

    # Add example inputs for user convenience
    gr.Markdown("---")
    gr.Markdown("## 🚀 Try Our Examples")
    gr.Markdown("**Quick Start**: Click on any example below to see SeqTex in action with pre-configured settings!")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[mesh_upload, y2z, y2x, z2x, upside_down, img_condition_seed, selected_view, model_choice, edge_refinement, text_prompt],
        cache_examples=False
    )

    # Acknowledgments
    gr.Markdown("---")
    gr.Markdown("## 🙏 Acknowledgments")
    gr.Markdown("""
    **Special thanks to [Toshihiro Hayashi](mailto:toshihiro@huggingface.co)** for his valuable support and assistance in fixing bugs for this demo.
    """)

if LOAD_FIRST is True:
    import gc
    get_seqtex_pipe()
    print("SeqTex pipeline loaded successfully.")
    get_sdxl_pipe()
    print("SDXL pipeline loaded successfully.")
    # get_flux_pipe()
    # Note: FLUX pipeline is available in code but not loaded due to GPU memory constraints on HF Space
    print("Note: FLUX and other models are available for local deployment.")
    gc.collect()

assert os.environ["OPENCV_IO_ENABLE_OPENEXR"] == "1", "OpenEXR support is required for this demo."
demo.launch(server_name="0.0.0.0")