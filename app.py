import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Load model (Use CPU-friendly settings for free tier)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

# If you have a GPU in your space (paid), change "cpu" to "cuda"
pipe = pipe.to("cpu")

def generate_image(prompt, negative_prompt):
    # Personalized Addition: Added negative prompting and adjustable steps
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=20 # Lower steps for faster CPU generation
    ).images[0]
    return image

# Define the Gradio Interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Enter your prompt", placeholder="A futuristic city..."),
        gr.Textbox(label="Negative Prompt", placeholder="blurry, bad quality...", value="low quality")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="My Custom Text-to-Image API",
    description="Generates images using Stable Diffusion. API accessible via the 'Use via API' link below."
)

iface.launch()
