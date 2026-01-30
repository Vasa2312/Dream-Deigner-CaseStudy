import torch
from model_2 import AgitationHybridPro

def test_model_architecture():
    # Verify the model outputs the correct shape
    model = AgitationHybridPro(embed_dim=128, lstm_hidden=64)
    dummy_input = torch.randn(2, 120, 6) # Batch size 2
    output = model(dummy_input)
    
    # Check 1: Output should have shape [2] (one score per sample)
    assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"
    
    # Check 2: Output should be finite (no NaNs)
    assert torch.isfinite(output).all(), "Model output contains NaNs"

if __name__ == "__main__":
    test_model_architecture()
    print("All tests passed!")
