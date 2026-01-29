import gradio as gr
from huggingface_hub import InferenceClient
import os

# You need to set HF_TOKEN in your Space settings for this to work reliably
# For local testing, replace os.environ.get with your actual string token
client = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=os.environ.get("HF_TOKEN"))

def generate_image(prompt, negative_prompt):
    try:
        image = client.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            width=1024,
            height=1024,
        )
        return image
    except Exception as e:
        return None

# Custom CSS for a dark, artistic look
css = """
body {background-color: #1a1a1a; color: white;}
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:
    gr.Markdown("# 🎨 Dream Painter API")
    gr.Markdown("Describe your dream, and the AI will paint it.")
    
    with gr.Row():
        with gr.Column():
            txt_input = gr.Textbox(label="What do you see?", placeholder="A cyberpunk city in the rain...")
            txt_negative = gr.Textbox(label="What to avoid?", placeholder="blur, low quality, distortion")
            btn_generate = gr.Button("Paint Dream", variant="primary")
        
        with gr.Column():
            img_output = gr.Image(label="Your Dream")
    
    # Event listener
    btn_generate.click(fn=generate_image, inputs=[txt_input, txt_negative], outputs=img_output)

if __name__ == "__main__":
    demo.launch()
