import json
import logging
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper


class FireRedAsrLlmDataset(Dataset):
    """
    Dataset for FireRedASR-LLM training

    Data format (JSON lines):
    {
        "wav": "/path/to/audio.wav",
        "text": "transcription text"
    }
    """

    def __init__(
        self,
        data_file: str,
        tokenizer,
        feat_extractor: ASRFeatExtractor,
        max_text_len: int = 128,
        sort_by_length: bool = True,
    ):
        """
        Args:
            data_file: Path to JSON lines file with wav and text
            tokenizer: LLM tokenizer
            feat_extractor: ASR feature extractor
            max_text_len: Maximum text length for tokenization
            sort_by_length: Whether to sort samples by audio length (useful for bucketing)
        """
        self.tokenizer = tokenizer
        self.feat_extractor = feat_extractor
        self.max_text_len = max_text_len

        # Load data
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Support multiple data formats
                # Format 1: {"wav": "...", "text": "..."}
                # Format 2: {"source": "...", "target": "..."} (your format)
                # Format 3: {"key": "...", "source": "...", "target": "..."}
                if 'wav' in item and 'text' in item:
                    self.data.append(item)
                elif 'source' in item and 'target' in item:
                    # Convert to standard format
                    # Remove spaces from target text
                    text = item['target'].replace(' ', '')
                    self.data.append({
                        'wav': item['source'],
                        'text': text
                    })

        if sort_by_length:
            # Sort by audio path to group similar lengths
            # In practice, you might want to compute actual lengths
            self.data = sorted(self.data, key=lambda x: x['wav'])

        logging.info(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with keys:
                - feat: (T, D) audio feature
                - feat_length: scalar
                - input_ids: (L,) input token ids
                - attention_mask: (L,) attention mask
                - labels: (L,) target labels
                - text: original text
        """
        item = self.data[idx]
        wav_path = item['wav']
        text = item['text']

        # Extract audio features
        feats, lengths, _ = self.feat_extractor([wav_path])
        feat = feats[0]  # (T, D)
        feat_length = lengths[0]  # scalar

        # Tokenize text
        input_ids, attention_mask, labels, clean_texts = \
            LlmTokenizerWrapper.preprocess_texts(
                origin_texts=[text],
                tokenizer=self.tokenizer,
                max_len=self.max_text_len,
                decode=False
            )

        return {
            'feat': feat,
            'feat_length': feat_length,
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'labels': labels[0],
            'text': text,
            'wav_path': wav_path,
        }


class LlmCollator:
    """
    Collate function for batching samples
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples

        Args:
            batch: List of samples from dataset

        Returns:
            Batched tensors
        """
        # Get batch size
        batch_size = len(batch)

        # Pad audio features
        feat_lengths = torch.stack([item['feat_length'] for item in batch])
        max_feat_len = feat_lengths.max().item()
        feat_dim = batch[0]['feat'].size(-1)

        padded_feats = torch.zeros(batch_size, max_feat_len, feat_dim)
        for i, item in enumerate(batch):
            feat_len = item['feat_length']
            padded_feats[i, :feat_len] = item['feat']

        # Pad text tokens
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]

        max_text_len = max([len(ids) for ids in input_ids])

        padded_input_ids = torch.full(
            (batch_size, max_text_len),
            self.pad_token_id,
            dtype=torch.long
        )
        padded_attention_mask = torch.zeros(
            (batch_size, max_text_len),
            dtype=torch.long
        )
        padded_labels = torch.full(
            (batch_size, max_text_len),
            -100,  # IGNORE_TOKEN_ID
            dtype=torch.long
        )

        for i in range(batch_size):
            seq_len = len(input_ids[i])
            padded_input_ids[i, :seq_len] = input_ids[i]
            padded_attention_mask[i, :seq_len] = attention_mask[i]
            padded_labels[i, :seq_len] = labels[i]

        return {
            'padded_feat': padded_feats,
            'feat_lengths': feat_lengths,
            'padded_input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels,
            'texts': [item['text'] for item in batch],
            'wav_paths': [item['wav_path'] for item in batch],
        }
    
class LlmCollatorV2:
    """
    Collate function for batching samples
    """

    def __init__(self, pad_token_id: int, dataset=None):
        self.pad_token_id = pad_token_id
        self.dataset = dataset 

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples

        Args:
            batch: List of samples from dataset

        Returns:
            Batched tensors
        """
        # Get batch size
        batch_size = len(batch)

        # Pad audio features
        feat_lengths = torch.stack([item['feat_length'] for item in batch])
        # max_feat_len = feat_lengths.max().item()
        # feat_dim = batch[0]['feat'].size(-1)

        # padded_feats = torch.zeros(batch_size, max_feat_len, feat_dim)
        # for i, item in enumerate(batch):
        #     feat_len = item['feat_length']
        #     padded_feats[i, :feat_len] = item['feat']
        feat_list = [item['feat'] for item in batch]
        padded_feats = self.dataset.feat_extractor.pad_feat(feat_list, 0.0) 
        text_list = [item['text'] for item in batch]
        padded_input_ids, padded_attention_mask, padded_labels, clean_texts = \
            LlmTokenizerWrapper.preprocess_texts(
                origin_texts=text_list,
                tokenizer=self.dataset.tokenizer,
                max_len=self.dataset.max_text_len,
                decode=False
            )

        # Pad text tokens
        # input_ids = [item['input_ids'] for item in batch]
        # attention_mask = [item['attention_mask'] for item in batch]
        # labels = [item['labels'] for item in batch]

        # max_text_len = max([len(ids) for ids in input_ids])

        # padded_input_ids = torch.full(
        #     (batch_size, max_text_len),
        #     self.pad_token_id,
        #     dtype=torch.long
        # )
        # padded_attention_mask = torch.zeros(
        #     (batch_size, max_text_len),
        #     dtype=torch.long
        # )
        # padded_labels = torch.full(
        #     (batch_size, max_text_len),
        #     -100,  # IGNORE_TOKEN_ID
        #     dtype=torch.long
        # )

        # for i in range(batch_size):
        #     seq_len = len(input_ids[i])
        #     padded_input_ids[i, :seq_len] = input_ids[i]
        #     padded_attention_mask[i, :seq_len] = attention_mask[i]
        #     padded_labels[i, :seq_len] = labels[i]


        return {
            'padded_feat': padded_feats,
            'feat_lengths': feat_lengths,
            'padded_input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels,
            'texts': [item['text'] for item in batch],
            'wav_paths': [item['wav_path'] for item in batch],
        }

def create_dataloader(
    data_file: str,
    tokenizer,
    feat_extractor: ASRFeatExtractor,
    batch_size: int,
    max_text_len: int = 128,
    num_workers: int = 4,
    shuffle: bool = True,
    sort_by_length: bool = True,
    is_debug: bool = False,
):
    """
    Create dataloader for training

    Args:
        data_file: Path to training data JSON file
        tokenizer: LLM tokenizer
        feat_extractor: Feature extractor
        batch_size: Batch size
        max_text_len: Max text length
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        sort_by_length: Whether to sort by length

    Returns:
        DataLoader instance
    """
    dataset = FireRedAsrLlmDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        feat_extractor=feat_extractor,
        max_text_len=max_text_len,
        sort_by_length=sort_by_length,
    )
    if is_debug:
        collator = LlmCollatorV2(pad_token_id=tokenizer.pad_token_id, dataset=dataset)
    else:
        collator = LlmCollator(pad_token_id=tokenizer.pad_token_id)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return dataloader
