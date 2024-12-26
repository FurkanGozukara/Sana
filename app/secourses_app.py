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
import spaces
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image
import tqdm

from app.sana_pipeline import SanaPipeline

def open_folder():
    open_folder_path = os.path.abspath("output\online_demo_img")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

MAX_SEED = np.iinfo(np.int32).max

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
os.environ["ENABLE_CPU_OFFLOAD"] = "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"
DEMO_PORT = int(os.getenv("DEMO_PORT", "15432"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
SCHEDULE_NAME = ["Flow_DPM_Solver"]
DEFAULT_SCHEDULE_NAME = "Flow_DPM_Solver"
NUM_IMAGES_PER_PROMPT = 1
INFER_SPEED = 0

generation_cancel_status = False

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config", default="configs/sana_config/1024ms/Sana_1600M_img1024.yaml")
    parser.add_argument(
        "--model_path",
        nargs="?",
        default="models/checkpoints/Sana_1600M_1024px.pth",
        type=str,
        help="Path to the model file (positional)",
    )
    parser.add_argument("--output", default="./", type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--cfg_scale", default=5.0, type=float)
    parser.add_argument("--pag_scale", default=2.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--step", default=-1, type=int)
    parser.add_argument("--custom_image_size", default=None, type=int)
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    return parser.parse_known_args()[0]

# Global variables to track model and pipeline
current_model_choice = "Sana 1K (1024x1024)"  # Default model
pipe = None

args = get_args()

def save_image_sana(img, seed="", save_img=False, ready_image=False):
    save_path = os.path.join(f"output/online_demo_img/{datetime.now().date()}")
    os.umask(0o000)  # file permission: 666; dir permission: 777
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
        if ready_image:
            img.save(unique_name)
        else:
            save_image(img, unique_name, nrow=1, normalize=True, value_range=(-1, 1))

    return unique_name

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def tensor_to_pil(img_tensor):
    img_tensor = img_tensor.cpu()
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img_tensor = (img_tensor * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_tensor = img_tensor.permute(1, 2, 0)
    return Image.fromarray(img_tensor.numpy())

@torch.no_grad()
@torch.inference_mode()
@spaces.GPU(enable_queue=True)
def generate(
    prompt: str = None,
    negative_prompt: str = "",
    style: str = DEFAULT_STYLE_NAME,
    use_negative_prompt: bool = False,
    num_imgs: int = 1,
    seed: int = 0,
    height: int = 1024,
    width: int = 1024,
    flow_dpms_guidance_scale: float = 5.0,
    flow_dpms_pag_guidance_scale: float = 2.0,
    flow_dpms_inference_steps: int = 20,
    randomize_seed: bool = False,
):
    global INFER_SPEED
    global pipe

    # Load the model if it hasn't been loaded yet
    if pipe is None:
        if torch.cuda.is_available():
            pipe = SanaPipeline(args.config)
            pipe.from_pretrained(args.model_path)
            pipe.register_progress_bar(gr.Progress())

    seed = int(seed) if isinstance(seed, (int, float)) else 0
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"PORT: {DEMO_PORT}, model_path: {args.model_path}")

    print(prompt)

    num_inference_steps = flow_dpms_inference_steps
    guidance_scale = flow_dpms_guidance_scale
    pag_guidance_scale = flow_dpms_pag_guidance_scale

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    pipe.progress_fn(0, desc="Sana Start")

    time_start = time.time()
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        pag_guidance_scale=pag_guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_imgs,
        generator=generator,
    )
    pipe.progress_fn(1.0, desc="Sana End")
    INFER_SPEED = (time.time() - time_start) / num_imgs

    save_img = True
    if save_img:
        img = [save_image_sana(img, f"{seed}_{i}", save_img=save_img) for i, img in enumerate(images)]
        print(img)
    else:
        img = images

    torch.cuda.empty_cache()

    return (
        img,
        seed,
        f"<span style='font-size: 16px; font-weight: bold;'>Inference Speed: {INFER_SPEED:.3f} s/Img</span>",
    )

def generate_multiple(
    prompt: str,
    negative_prompt: str,
    style: str,
    use_negative_prompt: bool,
    num_imgs: int,
    seed: int,
    height: int,
    width: int,
    flow_dpms_guidance_scale: float,
    flow_dpms_pag_guidance_scale: float,
    flow_dpms_inference_steps: int,
    randomize_seed: bool,
    use_multiline_prompt: bool,
    num_generations: int,
    progress=gr.Progress()
):
    global INFER_SPEED
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
                single_prompt,
                negative_prompt,
                style,
                use_negative_prompt,
                num_imgs,
                seed[0] if isinstance(seed, list) else seed,
                height,
                width,
                flow_dpms_guidance_scale,
                flow_dpms_pag_guidance_scale,
                flow_dpms_inference_steps,
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
    global current_model_choice
    global args
    global pipe

    if current_model_choice != model_choice:
        current_model_choice = model_choice

        if model_choice == "Sana 1K (1024x1024)":
            config_path = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"
            model_path = "models/checkpoints/Sana_1600M_1024px.pth"
            default_width = 1024
            default_height = 1024
        elif model_choice == "Sana 2K (2048x2048)":
            config_path = "configs/sana_config/2048ms/Sana_1600M_img2048_bf16.yaml"
            model_path = "models/checkpoints/Sana_1600M_2Kpx_BF16.pth"
            default_width = 2048
            default_height = 2048
        else:
            return "Invalid model choice", default_width, default_height

        args.config = config_path
        args.model_path = model_path

        # Load the new model
        if torch.cuda.is_available():
            if pipe is not None:
                del pipe
                torch.cuda.empty_cache()
            pipe = SanaPipeline(args.config)
            pipe.from_pretrained(args.model_path)
            pipe.register_progress_bar(gr.Progress())

        return f"Using {model_choice} model", default_width, default_height
    else:
        return f"Using {model_choice} model", default_width, default_height

title = f"""
SANA APP V4 : Exclusive to SECourses : https://www.patreon.com/posts/116474081
"""

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
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(f"# {title}")
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )

    with gr.Row():
        with gr.Column(elem_id="model-selection"):
            model_choice = gr.Radio(
                choices=["Sana 1K (1024x1024)", "Sana 2K (2048x2048)"],
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
                    container=False,
                    elem_id="prompt-text"
                )
                use_multiline_prompt = gr.Checkbox(label="Use multi-line prompts", value=True)
                with gr.Row():
                    run_button = gr.Button("Run")
                    cancel_button = gr.Button("Cancel Generation")
            
            with gr.Accordion("Advanced options", open=True):
                with gr.Group():
                    with gr.Row():
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
                        flow_dpms_inference_steps = gr.Slider(
                            label="Sampling steps",
                            minimum=5,
                            maximum=40,
                            step=1,
                            value=18,
                        )
                        flow_dpms_guidance_scale = gr.Slider(
                            label="CFG Guidance scale",
                            minimum=1,
                            maximum=10,
                            step=0.1,
                            value=5.0,
                        )
                        flow_dpms_pag_guidance_scale = gr.Slider(
                            label="PAG Guidance scale",
                            minimum=1,
                            maximum=4,
                            step=0.5,
                            value=2.0,
                        )
                    with gr.Row():
                        use_negative_prompt = gr.Checkbox(label="Use negative prompt", value=False)
                    negative_prompt = gr.Text(
                        label="Negative prompt",
                        max_lines=1,
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
                    with gr.Row():
                        schedule = gr.Radio(
                            show_label=True,
                            container=True,
                            interactive=True,
                            choices=SCHEDULE_NAME,
                            value=DEFAULT_SCHEDULE_NAME,
                            label="Sampler Schedule",
                        )
                        num_imgs = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=6,
                            step=1,
                            value=1,
                        )
                        num_generations = gr.Slider(
                            label="Number of generations",
                            minimum=1,
                            maximum=10000,
                            step=1,
                            value=1,
                        )

        with gr.Column():
            result = gr.Gallery(label="Result", show_label=False, columns=NUM_IMAGES_PER_PROMPT, format="png")
            speed_box = gr.Markdown(
                value=f"<span style='font-size: 16px; font-weight: bold;'>Inference speed: {INFER_SPEED} s/Img</span>"
            )
            status_box = gr.Markdown(value="")
            progress_bar = gr.Progress()
            btn_open_outputs = gr.Button("Open Outputs Folder")
            btn_open_outputs.click(fn=open_folder)

    cancel_button.click(
        fn=cancel_generation,
        inputs=[],
        outputs=[status_box]
    )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=[result, seed],
        fn=generate,
    )

    use_negative_prompt.change(
        fn=lambda x: gr.update(visible=x),
        inputs=use_negative_prompt,
        outputs=negative_prompt,
        api_name=False,
    )

    model_choice.change(
        fn=change_model,
        inputs=[model_choice],
        outputs=[model_info_box, width, height]
    )

    gr.on(
        triggers=[
            prompt.submit,
            negative_prompt.submit,
            run_button.click,
        ],
        fn=generate_multiple,
        inputs=[
            prompt,
            negative_prompt,
            style_selection,
            use_negative_prompt,
            num_imgs,
            seed,
            height,
            width,
            flow_dpms_guidance_scale,
            flow_dpms_pag_guidance_scale,
            flow_dpms_inference_steps,
            randomize_seed,
            use_multiline_prompt,
            num_generations,
        ],
        outputs=[result, seed, speed_box, status_box],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch(inbrowser=True, debug=True, share=args.share)