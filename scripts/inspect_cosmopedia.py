from datasets import load_dataset

def inspect():
    print("Inspecting Cosmopedia (Stanford)...")
    ds = load_dataset("HuggingFaceTB/cosmopedia", "stanford", split="train", streaming=True)
    for sample in ds:
        print("Keys:", sample.keys())
        print("Sample:", sample)
        break

if __name__ == "__main__":
    inspect()
