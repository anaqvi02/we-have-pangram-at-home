import torch
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import Config
from src.model.detector import PangramDetector

def interactive_test(model_path):
    print(f"Loading model from {model_path}...")
    try:
        detector = PangramDetector.load(model_path)
        detector.model.eval()
        detector.model.to(Config.DEVICE)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("\n--- 🤖 Pangram AI Detector Test ---")
    print("Enter text to analyze (Ctrl+C to exit).")
    print("---------------------------------------")

    while True:
        try:
            text = input("\n📝 Enter text: ")
            if not text.strip():
                continue
            
            # Tokenize
            encoding = detector.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=Config.MAX_LENGTH,
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].to(Config.DEVICE)
            attention_mask = encoding['attention_mask'].to(Config.DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = detector.model(input_ids, attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                ai_prob = probs[0][1].item()
            
            # Formatted Output
            bar_len = 20
            filled = int(ai_prob * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            label = "🤖 AI-GENERATED" if ai_prob > 0.5 else "👤 HUMAN-WRITTEN"
            color = "\033[91m" if ai_prob > 0.5 else "\033[92m" # Red for AI, Green for Human
            reset = "\033[0m"
            
            print(f"\nResult: {color}{label}{reset}")
            print(f"Confidence: {ai_prob:.4f} [{bar}]")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error during inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the 'final' checkpoint, but allow override
    default_path = Config.CHECKPOINT_DIR / "pangram_final"
    parser.add_argument("--model_path", type=str, default=str(default_path), help="Path to model checkpoint")
    args = parser.parse_args()
    
    interactive_test(args.model_path)
