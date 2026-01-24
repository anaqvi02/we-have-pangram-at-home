import torch
from src.model.detector import PangramDetector
from src.config import Config

def verify_mps_ops():
    if not torch.backends.mps.is_available():
        print("Skipping MPS verification: MPS not available.")
        return

    print("--- Starting MPS Verification ---")
    
    try:
        # 1. Load Model
        detector = PangramDetector()
        model = detector.model
        model.train() # Enable dropout etc
        
        # 2. Dummy Input
        input_ids = torch.randint(0, 1000, (4, 128)).to(Config.DEVICE)
        attention_mask = torch.ones((4, 128)).to(Config.DEVICE)
        labels = torch.tensor([0, 1, 0, 1]).to(Config.DEVICE)
        
        print("Forward Pass...")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"Loss: {loss.item()}")
        
        print("Backward Pass...")
        loss.backward()
        print("Gradients computed.")
        
        print("Optimization Step...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        optimizer.step()
        
        print("LayerNorm check...")
        # Often a point of failure in older MPS
        ln = torch.nn.LayerNorm(128).to(Config.DEVICE)
        ln(torch.randn(4, 128).to(Config.DEVICE))
        
        print("SUCCESS: DeBERTa-v3 seems stable on this MPS version.")
        
    except Exception as e:
        print(f"FAILURE: MPS Verification crashed with error:\n{e}")
        # Hint about fallback
        print("\nSUGGESTION: Try setting PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable.")

if __name__ == "__main__":
    verify_mps_ops()
