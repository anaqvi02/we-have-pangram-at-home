import argparse
from src.config import Config
from src.train_lib import train_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default=str(Config.INDEX_PATH))
    parser.add_argument("--human_data_dir", type=str, default=str(Config.HUMAN_DATASET_PATH))
    parser.add_argument("--ai_data_dir", type=str, default=str(Config.AI_DATASET_PATH))
    parser.add_argument("--output_dir", type=str, default=str(Config.CHECKPOINT_DIR / "pangram_final"))
    parser.add_argument("--epochs", type=int, default=Config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--use_mock_data", action="store_true", help="Use generated dummy data")
    args = parser.parse_args()
    
    # User requested reduction to 3 epochs while script was running
    if args.epochs > 3:
        print(f"Overriding requested epochs {args.epochs} to 3 per user request.")
        args.epochs = 3
        
    train_pipeline(
        human_data_dir=args.human_data_dir,
        ai_data_dir=args.ai_data_dir,
        index_path=args.index_path,
        output_model_dir=args.output_dir,
        epochs=args.epochs,
        use_mock_data=args.use_mock_data
    )

if __name__ == "__main__":
    main()
