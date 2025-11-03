import logging
import os
import random
import re
import yaml

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.module.adapter import Adapter
from fireredasr.tokenizer.llm_tokenizer import DEFAULT_SPEECH_TOKEN, IGNORE_TOKEN_ID
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper
from fireredasr.utils.param import count_model_parameters
import numpy as np 


class FireRedAsrLlm(nn.Module):
    @classmethod
    def load_encoder(cls, model_path):
        assert os.path.exists(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        # print(f'Loading encoder from: {model_path} package: {package}')
        print(f'fireredaed model args:', package["args"])

        # Save args to config.yaml file
        if not os.path.exists(os.path.join(os.path.dirname(model_path), "fireredasraed_config.yaml")):
            config_path = os.path.join(os.path.dirname(model_path), "fireredasraed_config.yaml")
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(package["args"], f, default_flow_style=False, allow_unicode=True)
            print(f'Model args saved to: {config_path}')

        model = FireRedAsrAed.from_args(package["args"])
        if "model_state_dict" in package:
            model.load_state_dict(package["model_state_dict"], strict=False)
        encoder = model.encoder
        encoder_dim = encoder.odim
        return encoder, encoder_dim

    @classmethod
    def from_args(cls, args):
        logging.info(args)
        logging.info("Build FireRedAsrLlm")
        # Build Speech Encoder
        encoder, encoder_dim = cls.load_encoder(args.encoder_path)
        count_model_parameters(encoder)
        if args.freeze_encoder:
            logging.info(f"Frezee encoder")
            for name, param in encoder.named_parameters():
                param.requires_grad = False
            encoder.eval()

        if args.use_flash_attn:
            attn_implementation = "flash_attention_2"
            # 注意：使用混合精度训练(AMP)时，模型应该保持FP32
            # autocast会自动处理精度转换，避免数值不稳定
            
            if args.use_fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            attn_implementation = "eager"
            # 注意：使用混合精度训练(AMP)时，模型应该保持FP32
            # autocast会自动处理精度转换，避免数值不稳定
            if args.use_fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        # Build LLM
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        count_model_parameters(llm)

        # LLM Freeze or LoRA
        llm_dim = llm.config.hidden_size
        if args.freeze_llm:
            logging.info(f"Frezee LLM")
            for name, param in llm.named_parameters():
                param.requires_grad = False
            llm.eval()
        else:
            if args.use_lora:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=64,
                    lora_alpha=16,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "up_proj",
                        "gate_proj",
                        "down_proj",
                    ],
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )
                llm = get_peft_model(llm, lora_config)
                llm.print_trainable_parameters()

        tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(args.llm_dir)
        assert tokenizer.pad_token_id == tokenizer.convert_tokens_to_ids("<|endoftext|>")
        llm.config.pad_token_id = tokenizer.pad_token_id
        llm.config.bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        llm.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        llm.config.default_speech_token_id = tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )
        logging.info(f'llm config pad token id: {llm.config.pad_token_id}, bos token id: {llm.config.bos_token_id}, eos token id: {llm.config.eos_token_id}, default speech token id: {llm.config.default_speech_token_id}')
        # Build projector
        encoder_projector = Adapter(
            encoder_dim, llm_dim, args.encoder_downsample_rate)
        print(f"build adapter: encoder_dim={encoder_dim}, llm_dim={llm_dim}, downsample_rate={args.encoder_downsample_rate}")
        count_model_parameters(encoder_projector)

        return cls(encoder, llm, encoder_projector,
                   args.freeze_encoder, args.freeze_llm)

    def __init__(self, encoder, llm, encoder_projector,
                 freeze_encoder, freeze_llm):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        # args
        self.freeze_encoder = freeze_encoder
        self.freeze_llm = freeze_llm
        self.llm_config = llm.config

    def forward(self, padded_feat, feat_lengths, padded_input_ids, attention_mask, labels=None):
        """
        Forward pass for training
        Args:
            padded_feat: (B, T, D) - padded speech features
            feat_lengths: (B,) - actual lengths of speech features
            padded_input_ids: (B, L) - padded input token ids
            attention_mask: (B, L) - attention mask for input tokens
            labels: (B, L) - target labels for training (optional)
        Returns:
            loss: scalar loss value (if labels provided)
            logits: (B, L, vocab_size) - model outputs
        """
        # Encode speech features
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
        else:
            encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)

        # 调试：检查 encoder 输出
        if torch.isnan(encoder_outs).any():
            logging.error(f"NaN in encoder_outs! shape={encoder_outs.shape}")
        if torch.isinf(encoder_outs).any():
            logging.error(f"Inf in encoder_outs! shape={encoder_outs.shape}")

        # Project encoder outputs to LLM dimension
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)

        # 调试：检查 projector 输出
        if torch.isnan(speech_features).any():
            logging.error(f"NaN in speech_features after projection! shape={speech_features.shape}")
            logging.error(f"encoder_outs range: [{encoder_outs.min():.6f}, {encoder_outs.max():.6f}]")
            # 检查 projector 权重
            for name, param in self.encoder_projector.named_parameters():
                logging.error(f"Projector {name}: has_nan={torch.isnan(param).any()}, range=[{param.min():.6f}, {param.max():.6f}]")
        if torch.isinf(speech_features).any():
            logging.error(f"Inf in speech_features after projection!")

        # Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)

        # 调试：检查 text embeddings
        if torch.isnan(inputs_embeds).any():
            logging.error(f"NaN in text inputs_embeds!")
        if torch.isinf(inputs_embeds).any():
            logging.error(f"Inf in text inputs_embeds!")

        # Merge speech features with text embeddings
        inputs_embeds, attention_mask, labels = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype),
                inputs_embeds,
                padded_input_ids,
                attention_mask,
                labels=labels,
                speech_lens=speech_lens
            )

        # 调试：检查 merged embeddings
        if torch.isnan(inputs_embeds).any():
            logging.error(f"NaN in merged inputs_embeds!")
        if torch.isinf(inputs_embeds).any():
            logging.error(f"Inf in merged inputs_embeds!")

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs

    def transcribe(self, padded_feat, feat_lengths, padded_input_ids, attention_mask,
                   beam_size=1, decode_max_len=0, decode_min_len=0,
                   repetition_penalty=1.0, llm_length_penalty=1.0, temperature=1.0):
        logging.info(f'padded_feat: {padded_feat.shape}, feat_lengths: {feat_lengths}')
        encoder_outs, enc_lengths, enc_mask = self.encoder(padded_feat, feat_lengths)
        # save for debug
        # output_dir = '/data/zhanglingling/AI/asr/FunASR/tests'
        # encoder_outs_npy = os.path.join(output_dir, f"encoder_outs_fireredasr.npy")
        # np.save(encoder_outs_npy, encoder_outs.cpu().numpy())
        # enc_lengths_npy = os.path.join(output_dir, f"enc_lengths_fireredasr.npy")
        # np.save(enc_lengths_npy, enc_lengths.cpu().numpy())
        speech_features, speech_lens = self.encoder_projector(encoder_outs, enc_lengths)
        # speech_features_npy = os.path.join(output_dir, f"speech_features_fireredasr.npy")
        # np.save(speech_features_npy, speech_features.cpu().numpy())
        # speech_lens_npy = os.path.join(output_dir, f"speech_lens_fireredasr.npy")
        # np.save(speech_lens_npy, speech_lens.cpu().numpy())
        inputs_embeds = self.llm.get_input_embeddings()(padded_input_ids)
        # inputs_embeds_npy = os.path.join(output_dir, f"inputs_embeds_fireredasr.npy")
        # np.save(inputs_embeds_npy, inputs_embeds.cpu().numpy())
        logging.info(f'speech_features: {speech_features.shape}, speech_lens: {speech_lens}')
        logging.info(f'inputs_embeds: {inputs_embeds.shape}, padded_input_ids: {padded_input_ids.shape} attention_mask: {attention_mask.shape}')
        inputs_embeds, attention_mask, _ = \
            self._merge_input_ids_with_speech_features(
                speech_features.to(inputs_embeds.dtype), inputs_embeds, padded_input_ids, attention_mask,
                speech_lens=speech_lens
            )

        max_new_tokens = speech_features.size(1) if decode_max_len < 1 else decode_max_len
        max_new_tokens = max(1, max_new_tokens)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            do_sample=False,
            min_length=decode_min_len,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            length_penalty=llm_length_penalty,
            temperature=temperature,
            bos_token_id=self.llm.config.bos_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            pad_token_id=self.llm.config.pad_token_id,
            attention_mask=attention_mask, # 推理加attention mask
        )
        # logging.info(f"generated_ids: {generated_ids.shape}")
        # logging.info(f'generated_ids: {generated_ids}')
        return generated_ids


    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
        speech_lens=None
    ):
        """
        Modified from: https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/whisper_llm_zh/model.py
        """
        #speech_lens = None
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.llm.config.pad_token_id)
        )
        # 修复batch推理问题 
        #left_padding = (input_ids[0, 0] == self.llm.config.pad_token_id).item()
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == self.llm.config.default_speech_token_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != self.llm.config.default_speech_token_id
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged speech-text sequence.
        # `special_speech_token_mask` identifies speech tokens. Each speech token will be replaced by `nb_text_tokens_per_speechs - 1` text tokens.
        # `torch.cumsum` computes how each speech token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )  # (N,U)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                IGNORE_TOKEN_ID,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_speech_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_speech_indices
            ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        if speech_lens is not None:
            speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        if speech_lens is not None:
            speech_to_overwrite &= speech_pad_position
        final_attention_mask |= speech_to_overwrite

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == self.llm.config.pad_token_id
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels #, position_ids
