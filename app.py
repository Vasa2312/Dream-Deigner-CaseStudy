import gradio as gr
from huggingface_hub import InferenceClient

# Use a free generic model from HF Hub
model_id = "google/flan-t5-large"
client = InferenceClient(model=model_id)

def answer_query(message, history):
    # Prompt engineering to act as a medical assistant
    prompt = f"Context: You are a helpful medical assistant for agitation management.\nQuestion: {message}\nAnswer:"
    
    try:
        response = client.text_generation(prompt, max_new_tokens=100)
        return response
    except Exception as e:
        return f"Error connecting to API: {str(e)}"

# Create the Interface
demo = gr.ChatInterface(
    fn=answer_query,
    title="Agitation Care Advisor (API-Based)",
    description="Ask questions about patient care. Powered by FLAN-T5 via HF API."
)

if __name__ == "__main__":
    demo.launch()
