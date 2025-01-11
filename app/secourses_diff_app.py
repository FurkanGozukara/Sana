from __future__ import annotations

import argparse
import glob
import os
import platform
import random
import re
import time
import uuid
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from PIL import Image
import tqdm

from diffusers import SanaPipeline
from transformers import AutoTokenizer

MAX_SEED = np.iinfo(np.int32).max

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
DEMO_PORT = int(os.getenv("DEMO_PORT", "15432"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ASPECT_RATIOS = {
    "1:1":  (1024, 1024),
    "4:3":  (1152, 896),
    "3:4":  (896, 1152),
    "3:2":  (1216, 832),
    "2:3":  (832, 1216),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "21:9": (1536, 640),
    "9:21": (640, 1536),
    "4:5":  (896, 1088),
    "5:4":  (1088, 896),
}

ASPECT_RATIOS_2K = {
    "1:1":  (2048, 2048),
    "4:3":  (2304, 1792),
    "3:4":  (1792, 2304),
    "3:2":  (2432, 1664),
    "2:3":  (1664, 2432),
    "16:9": (2688, 1536),
    "9:16": (1536, 2688),
    "21:9": (3072, 1280),
    "9:21": (1280, 3072),
    "4:5":  (1792, 2240),
    "5:4":  (2240, 1792),
}

ASPECT_RATIOS_4K = {
    "1:1":  (4096, 4096),
    "4:3":  (4608, 3584),
    "3:4":  (3584, 4608),
    "3:2":  (4864, 3328),
    "2:3":  (3328, 4864),
    "16:9": (5376, 3072),
    "9:16": (3072, 5376),
    "21:9": (6144, 2560),
    "9:21": (2560, 6144),
    "4:5":  (3584, 4480),
    "5:4":  (4480, 3584),
}

style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, "
        "glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, "
        "disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"

pipe = None
generation_cancel_status = False
INFER_SPEED = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    return parser.parse_known_args()[0]

args = get_args()

def load_model(model_choice):
    global pipe
    model_paths = {
        "Sana 1K (1024x1024)": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        "Sana 2K (2048x2048)": "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers",
        "Sana 4K (4096x4096)": "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers",
    }
    model_path = model_paths.get(model_choice)
    
    if model_path:
        if pipe is None or pipe.model_path != model_path:
            if torch.cuda.is_available():
                try:
                    pipe = SanaPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        variant="bf16",
                    )
                    pipe.to("cuda")
                    pipe.vae.to(torch.float32)  # Ensure VAE is in float32
                    pipe.enable_vae_tiling()
                    pipe.model_path = model_path  # Track the current model path
                    return True, f"{model_choice} model loaded successfully"
                except Exception as e:
                    return False, f"Error loading {model_choice} model: {str(e)}"
            else:
                return False, "CUDA is not available"
        return True, f"{model_choice} model already loaded"
    return False, "Invalid model choice"

def apply_vram_optimizations(enable_vae_tiling, enable_vae_slicing, enable_model_cpu_offload, enable_sequential_cpu_offload, tile_sample_min_size):
    global pipe
    if enable_vae_tiling:
        pipe.vae.enable_tiling(tile_sample_min_height=tile_sample_min_size, tile_sample_min_width=tile_sample_min_size)
    else:
        pipe.vae.disable_tiling()

    if enable_vae_slicing:
        pipe.enable_vae_slicing()
    else:
        pipe.disable_vae_slicing()

    if enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    elif enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.disable_attention_slicing()

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

def save_image(img, seed="", save_img=True):
    save_path = os.path.join(f"output/online_demo_img/{datetime.now().date()}")
    os.makedirs(save_path, exist_ok=True)

    existing_files = glob.glob(os.path.join(save_path, "img_*.png"))
    if existing_files:
        numbers = [int(re.search(r'img_(\d+)', f).group(1)) for f in existing_files if re.search(r'img_(\d+)', f)]
        next_number = max(numbers, default=0) + 1
    else:
        next_number = 1

    unique_name = f"img_{str(next_number).zfill(4)}.png"
    unique_name = os.path.join(save_path, unique_name)

    if save_img:
        img.save(unique_name)

    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@torch.no_grad()
def generate(
    model_choice: str,
    prompt: str,
    negative_prompt: str,
    style: str,
    use_negative_prompt: bool,
    num_imgs: int,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    enable_vae_tiling: bool,
    enable_vae_slicing: bool,
    enable_model_cpu_offload: bool,
    enable_sequential_cpu_offload: bool,
    tile_sample_min_size: int,
    randomize_seed: bool,
):
    global pipe, INFER_SPEED

    if pipe is None or pipe.model_path != model_choice:
        success, message = load_model(model_choice)
        if not success:
            return [], 0, f"Error: {message}"

    pipe.vae.to(torch.float32)
    if not use_negative_prompt:
        negative_prompt = None

    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    apply_vram_optimizations(enable_vae_tiling, enable_vae_slicing, enable_model_cpu_offload, enable_sequential_cpu_offload, tile_sample_min_size)

    seed = randomize_seed_fn(seed, randomize_seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    time_start = time.time()
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_imgs,
        generator=generator,
    ).images
    INFER_SPEED = (time.time() - time_start) / num_imgs

    saved_images = [save_image(img, f"{seed}_{i}") for i, img in enumerate(images)]

    return (
        saved_images,
        seed,
        f"<span style='font-size: 16px; font-weight: bold;'>Inference Speed: {INFER_SPEED:.3f} s/Img</span>",
    )

def generate_multiple(
    model_choice: str,
    prompt: str,
    negative_prompt: str,
    style: str,
    use_negative_prompt: bool,
    num_imgs: int,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    num_inference_steps: int,
    enable_vae_tiling: bool,
    enable_vae_slicing: bool,
    enable_model_cpu_offload: bool,
    enable_sequential_cpu_offload: bool,
    tile_sample_min_size: int,
    randomize_seed: bool,
    use_multiline_prompt: bool,
    num_generations: int,
    progress=gr.Progress()
):
    global generation_cancel_status

    generation_cancel_status = False

    if use_multiline_prompt:
        prompts = [p.strip() for p in prompt.split('\n') if len(p.strip()) >= 3]
    else:
        prompts = [prompt.strip()] if len(prompt.strip()) >= 3 else []
    
    if not prompts:
        return [], [], "No valid prompts", "Error: All prompts were invalid (less than 3 characters)"

    total_generations = len(prompts) * num_generations
    
    all_images = []
    all_seeds = []
    
    for j in range(num_generations):
        for i, single_prompt in enumerate(prompts):
            if generation_cancel_status:
                yield all_images, all_seeds, "Generation cancelled", "Generation cancelled!"
                return

            images, current_seed, speed_info = generate(
                model_choice,
                single_prompt,
                negative_prompt,
                style,
                use_negative_prompt,
                num_imgs,
                seed,
                height,
                width,
                guidance_scale,
                num_inference_steps,
                enable_vae_tiling,
                enable_vae_slicing,
                enable_model_cpu_offload,
                enable_sequential_cpu_offload,
                tile_sample_min_size,
                randomize_seed,
            )
            
            all_images.extend(images)
            all_seeds.append(current_seed)
            
            progress_percentage = (j * len(prompts) + i + 1) / total_generations
            progress(progress_percentage, desc=f"Generating image {j * len(prompts) + i + 1}/{total_generations}")
            
            eta = INFER_SPEED * (total_generations - (j * len(prompts) + i + 1))
            status_info = f"Generated {j * len(prompts) + i + 1}/{total_generations} images. ETA: {eta:.2f}s"
            print(status_info)
            
            yield all_images, all_seeds, speed_info, status_info

    yield all_images, all_seeds, speed_info, "Generation complete!"

def cancel_generation():
    global generation_cancel_status
    generation_cancel_status = True
    return "Cancelling generation..."

def change_model(model_choice):
    success, message = load_model(model_choice)
    if success:
        ratios = ASPECT_RATIOS if model_choice == "Sana 1K (1024x1024)" else (ASPECT_RATIOS_2K if model_choice == "Sana 2K (2048x2048)" else ASPECT_RATIOS_4K)
        default_width, default_height = ratios["1:1"]
        return message, gr.update(choices=list(ratios.keys()), value="1:1"), gr.update(value=default_width), gr.update(value=default_height)
    else:
        return message, gr.update(), gr.update(), gr.update()

def update_dimensions(aspect_ratio_key, model_choice):
    ratios = ASPECT_RATIOS if model_choice == "Sana 1K (1024x1024)" else (ASPECT_RATIOS_2K if model_choice == "Sana 2K (2048x2048)" else ASPECT_RATIOS_4K)
    width, height = ratios[aspect_ratio_key]
    return gr.update(value=width), gr.update(value=height)

def open_folder():
    open_folder_path = os.path.abspath("output/online_demo_img")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')
    else:  # macOS and other Unix
        os.system(f'open "{open_folder_path}"')

title = "SANA APP V13 : Exclusive to SECourses : https://www.patreon.com/posts/116474081"

examples = [
    'a cyberpunk cat with a neon sign that says "Sana"',
    "A very detailed and realistic full body photo set of a tall, slim, and athletic Shiba Inu in a white oversized straight t-shirt, white shorts, and short white shoes.",
    "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
    "portrait photo of a girl, photograph, highly detailed face, depth of field",
    'make me a logo that says "So Fast"  with a really cool flying dragon shape with lightning sparks all over the sides and all of it contains Indonesian language',
    "🐶 Wearing 🕶 flying on the 🌈",
    "👧 with 🌹 in the ❄️",
    "an old rusted robot wearing pants and a jacket riding skis in a supermarket.",
    "professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.",
    "Astronaut in a jungle, cold color palette, muted colors, detailed",
    "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests",
]

css = """
/* Add your custom CSS here */
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(f"# {title}")

    with gr.Row():
        with gr.Column(elem_id="model-selection"):
            model_choice = gr.Radio(
                choices=["Sana 1K (1024x1024)", "Sana 2K (2048x2048)", "Sana 4K (4096x4096)"],
                label="Select Model",
                value="Sana 1K (1024x1024)",
            )
        with gr.Column():
            model_info_box = gr.Markdown(value="Using Sana 1K (1024x1024) model")

    with gr.Row():
        with gr.Column(elem_id="advanced-options"):
            with gr.Group():
                prompt = gr.Textbox(
                    label="Prompt",
                    show_label=False,
                    lines=5,
                    placeholder="Enter your prompt(s)",
                    elem_id="prompt-text"
                )
                use_multiline_prompt = gr.Checkbox(label="Use multi-line prompts", value=True)
                with gr.Row():
                    run_button = gr.Button("Run")
                    cancel_button = gr.Button("Cancel Generation")

            with gr.Accordion("Advanced options", open=True):
                with gr.Group():
                    with gr.Row():
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="1:1",
                        )
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=32,
                            value=1024,
                        )
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=MAX_IMAGE_SIZE,
                            step=32,
                            value=1024,
                        )
                    with gr.Row():
                        num_inference_steps = gr.Slider(
                            label="Sampling steps",
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=20,
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance scale",
                            minimum=0,
                            maximum=20,
                            step=0.1,
                            value=5.0,
                        )
                    with gr.Row():
                        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            placeholder="Enter a negative prompt",
                            visible=True,
                        )
                    style_selection = gr.Radio(
                        show_label=True,
                        container=True,
                        interactive=True,
                        choices=STYLE_NAMES,
                        value=DEFAULT_STYLE_NAME,
                        label="Image Style",
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    num_imgs = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                    )
                    num_generations = gr.Slider(
                        label="Number of Generations",
                        minimum=1,
                        maximum=10000,
                        step=1,
                        value=1,
                    )

            with gr.Accordion("VRAM Optimizations", open=False):
                enable_vae_tiling = gr.Checkbox(label="Enable VAE Tiling", value=True)
                enable_vae_slicing = gr.Checkbox(label="Enable VAE Slicing", value=False)
                enable_model_cpu_offload = gr.Checkbox(label="Enable Model CPU Offload", value=False)
                enable_sequential_cpu_offload = gr.Checkbox(label="Enable Sequential CPU Offload", value=False)
                tile_sample_min_size = gr.Slider(label="Tile Sample Min Size", minimum=448, maximum=2048, step=64, value=1024)

        with gr.Column():
            result = gr.Gallery(label="Result", show_label=False, elem_id="gallery")
            speed_box = gr.Markdown(value="Inference speed: N/A")
            status_box = gr.Markdown(value="")
            progress_bar = gr.Progress()
            btn_open_outputs = gr.Button("Open Outputs Folder")

    model_choice.change(
        fn=change_model,
        inputs=[model_choice],
        outputs=[model_info_box, aspect_ratio, width, height]
    )

    aspect_ratio.change(
        fn=update_dimensions,
        inputs=[aspect_ratio, model_choice],
        outputs=[width, height]
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
    )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
        cache_examples=False,
    )

    run_button.click(
        fn=generate_multiple,
        inputs=[
            model_choice,
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            num_imgs,
            seed,
            height,
            width,
            guidance_scale,
            num_inference_steps,
            enable_vae_tiling,
            enable_vae_slicing,
            enable_model_cpu_offload,
            enable_sequential_cpu_offload,
            tile_sample_min_size,
            randomize_seed,
            use_multiline_prompt,
            num_generations,
        ],
        outputs=[result, seed, speed_box, status_box],
    )

    cancel_button.click(
        fn=cancel_generation,
        inputs=[],
        outputs=[status_box]
    )

    btn_open_outputs.click(fn=open_folder)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(inbrowser=True, debug=True, share=args.share)