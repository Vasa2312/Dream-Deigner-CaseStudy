import gradio as gr
import torch
import numpy as np
from model_2 import AgitationHybridPro # Import your actual model structure

# 1. Load the Model
device = "cpu" # Spaces run on CPU by default
model = AgitationHybridPro(embed_dim=128, lstm_hidden=64).to(device)

# Load weights (Ensure 'best_pro_fold2.pt' is uploaded to the repo)
try:
    model.load_state_dict(torch.load("best_pro_fold2.pt", map_location=device))
    print("Model loaded successfully!")
except:
    print("Warning: Model weights not found, using random initialization for demo.")

model.eval()

def predict_agitation(noise_level):
    # Simulate data: [Batch=1, Time=120, Channels=6]
    # In a real app, this would be your uploaded CSV file
    dummy_input = torch.randn(1, 120, 6) * float(noise_level)
    
    with torch.no_grad():
        logits = model(dummy_input)
        prob = torch.sigmoid(logits).item()
    
    result = "AGITATION DETECTED ⚠️" if prob > 0.85 else "Normal Behavior ✅"
    return f"Confidence: {prob:.4f}\nResult: {result}"

# 2. Interface
iface = gr.Interface(
    fn=predict_agitation,
    inputs=gr.Slider(0.1, 5.0, label="Simulated Sensor Noise Level"),
    outputs="text",
    title="Real-Time Agitation Detection (Local)",
    description="Running custom HybridPro Transformer-LSTM model locally on CPU."
)

if __name__ == "__main__":
    iface.launch()
