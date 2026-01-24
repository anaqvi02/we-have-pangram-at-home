import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
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
