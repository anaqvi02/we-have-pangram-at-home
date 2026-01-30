import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
from pathlib import Path
from src.config import Config
import pandas as pd

class StreamingTextDataset(Dataset):
    """
    A unified Dataset wrapper that supports both in-memory lists (for debugging/mining)
    and disk-backed loading.
    """
    def __init__(self, data_source, tokenizer=None, max_length=Config.MAX_LENGTH):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Determine if data_source is a list or path
        self.is_memory = isinstance(data_source, list)
        self.is_hf_dataset = isinstance(data_source, HFDataset)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if self.is_memory:
            item = self.data_source[idx]
            text = item['text']
            label = item['label']
        elif self.is_hf_dataset:
            item = self.data_source[idx]
            text = item['text']
            label = item['label']
        else:
            # Placeholder for parquet loading logic
            raise NotImplementedError("Direct disk streaming not yet implemented in this snippet")

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long),
                'text': text # Keep text for mining
            }
        else:
            return {'text': text, 'label': label}

def create_dataloader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0 # Critical for MPS stability
    )

class PretokenizedDataset(Dataset):
    """
    Holds pre-tokenized tensors in memory.
    Optimized for fast Training Loop on MPS.
    """
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
    
    def extend(self, other_dataset):
        """Append another PretokenizedDataset to this one."""
        self.input_ids = torch.cat([self.input_ids, other_dataset.input_ids], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, other_dataset.attention_mask], dim=0)
        self.labels = torch.cat([self.labels, other_dataset.labels], dim=0)


class MemoryMappedDataset(Dataset):
    """Arrow-backed dataset wrapper that returns raw text + label.

    This keeps the underlying HuggingFace dataset memory-mapped and avoids any eager
    `map()` tokenization pass. Tokenization happens in the DataLoader collate function
    (batched + dynamic padding), which is both faster and lower-IO for large runs.
    """

    def __init__(self, hf_dataset, tokenizer, max_length=Config.MAX_LENGTH):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'text': item['text'],
            'labels': int(item['label']),
        }


class GrowableDataset(Dataset):
    """A dataset that can grow during training (for curriculum learning).

    Base dataset: Arrow-backed HF dataset + lazy tokenization.
    Mined samples: kept as raw texts + labels; tokenized by the same collator as the base.

    This keeps GPU fed (dynamic padding) and avoids storing huge fixed-512 tensors in RAM.
    """

    def __init__(self, base_hf_dataset, tokenizer, max_length=Config.MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.base_dataset = base_hf_dataset

        # Buffer for mined samples (raw text)
        self.buffer_texts = []
        self.buffer_labels = []

    def __len__(self):
        return len(self.base_dataset) + len(self.buffer_labels)

    def __getitem__(self, idx):
        base_len = len(self.base_dataset)

        if idx < base_len:
            item = self.base_dataset[idx]
            return {'text': item['text'], 'labels': int(item['label'])}

        buffer_idx = idx - base_len
        return {'text': self.buffer_texts[buffer_idx], 'labels': int(self.buffer_labels[buffer_idx])}

    def extend(self, new_pairs):
        """Add new samples from mining.

        Args:
            new_pairs: list[dict] with keys: 'text', 'label'
        """
        if not new_pairs:
            return

        self.buffer_texts.extend([p['text'] for p in new_pairs])
        self.buffer_labels.extend([int(p['label']) for p in new_pairs])

        print(f"  → Buffer grew to {len(self.buffer_labels)} mined samples")
    
    def get_buffer_size(self):
        """Return the number of mined samples in buffer."""
        return len(self.buffer_labels)
