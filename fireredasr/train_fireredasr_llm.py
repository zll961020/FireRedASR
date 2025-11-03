#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.data.llm_dataset import create_dataloader
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper


def setup_logging(log_dir, rank=0):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_rank{rank}.log')

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if rank == 0 else logging.NullHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def get_args():
    parser = argparse.ArgumentParser(description='Train FireRedASR-LLM')

    # Model args
    parser.add_argument('--model_dir', type=str, required=True, help='Path to pretrained model directory')
    parser.add_argument('--freeze_encoder', type=int, default=1,
                        help='Freeze encoder weights')
    parser.add_argument('--freeze_llm', type=int, default=1,
                        help='Freeze LLM weights')
    parser.add_argument('--use_lora', type=int, default=1,
                        help='Use LoRA for LLM')
    parser.add_argument('--use_amp', type=int, default=1,
                        help='Use FP16 mixed precision training')
    # parser.add_argument('--use_flash_attn', type=int, default=0,
    #                     help='Use flash attention')
    # parser.add_argument('--encoder_downsample_rate', type=int, default=8,
    #                     help='Encoder downsample rate')

    # Data args
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data JSON file')
    parser.add_argument('--valid_data', type=str, default='',
                        help='Path to validation data JSON file')
    # parser.add_argument('--cmvn_path', type=str, required=True,
    #                     help='Path to CMVN file')
    parser.add_argument('--max_text_len', type=int, default=128,
                        help='Maximum text length')

    # Training args
    parser.add_argument('--output_dir', type=str, default='exp/fireredasr_llm',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--keep_last_n', type=int, default=5,
                        help='Keep last N checkpoints')

    # Resume training
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


def save_checkpoint(model, optimizer, scheduler, step, epoch, output_dir, rank=0):
    """Save checkpoint"""
    if rank != 0:
        return

    checkpoint_dir = os.path.join(output_dir, f'checkpoint-{step}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model state
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'epoch': epoch,
    }, os.path.join(checkpoint_dir, 'model.pth.tar'))

    logging.info(f'Saved checkpoint to {checkpoint_dir}')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('step', 0), checkpoint.get('epoch', 0)


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, args,
                     writer=None, global_step=0, rank=0, world_size=1, scaler=None):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    optimizer.zero_grad()

    use_amp = bool(args.use_amp) and scaler is not None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}') if rank == 0 else dataloader

    for batch_idx, batch in enumerate(pbar):
        # Move batch to GPU
        padded_feat = batch['padded_feat'].cuda()
        feat_lengths = batch['feat_lengths'].cuda()
        padded_input_ids = batch['padded_input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        # 调试：检查输入数据
        if batch_idx == 0 and rank == 0:
            logging.info(f"Input shapes - feat: {padded_feat.shape}, input_ids: {padded_input_ids.shape}")
            logging.info(f"Input ranges - feat: [{padded_feat.min():.3f}, {padded_feat.max():.3f}]")
            logging.info(f"Labels range: [{labels.min()}, {labels.max()}]")
            logging.info(f"Labels contains ignore_index (-100): {(labels == -100).sum().item()} tokens")

            # 检查是否有 NaN/Inf 输入
            if torch.isnan(padded_feat).any():
                logging.error("NaN detected in input features!")
            if torch.isinf(padded_feat).any():
                logging.error("Inf detected in input features!")

        # Forward pass with automatic mixed precision
        with autocast(enabled=use_amp):
            outputs = model(
                padded_feat=padded_feat,
                feat_lengths=feat_lengths,
                padded_input_ids=padded_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            logging.info(f'output loss: {outputs.loss}')

            loss = outputs.loss / args.gradient_accumulation_steps

            # 调试：在第一个 batch 检查中间值
            if batch_idx == 0 and rank == 0:
                logging.info(f"Loss value: {loss.item() * args.gradient_accumulation_steps}")
                logging.info(f"Output logits shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")
                if hasattr(outputs, 'logits'):
                    logging.info(f"Logits range: [{outputs.logits.min():.3f}, {outputs.logits.max():.3f}]")
                    if torch.isnan(outputs.logits).any():
                        logging.error("NaN detected in output logits!")
                    if torch.isinf(outputs.logits).any():
                        logging.error("Inf detected in output logits!")

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 检查 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            if rank == 0:
                logging.warning(f'NaN or Inf loss detected at step {global_step}, skipping this batch')
            optimizer.zero_grad()
            continue

        total_loss += loss.item()

        # Update weights
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # Logging
            if global_step % args.log_interval == 0 and rank == 0:
                avg_loss = total_loss / args.log_interval
                lr = scheduler.get_last_lr()[0]

                logging.info(
                    f'Epoch: {epoch} | Step: {global_step} | '
                    f'Loss: {avg_loss:.4f} | LR: {lr:.2e}'
                )

                if writer is not None:
                    writer.add_scalar('train/loss', avg_loss, global_step)
                    writer.add_scalar('train/lr', lr, global_step)

                if rank == 0:
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})

                total_loss = 0.0

            # Save checkpoint
            # if global_step % args.save_interval == 0:
            #     save_checkpoint(model, optimizer, scheduler, global_step, epoch,
            #                     args.output_dir, rank)

    return global_step


def validate(model, dataloader, rank=0):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation') if rank == 0 else dataloader
        for batch in pbar:
            padded_feat = batch['padded_feat'].cuda()
            feat_lengths = batch['feat_lengths'].cuda()
            padded_input_ids = batch['padded_input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()

            outputs = model(
                padded_feat=padded_feat,
                feat_lengths=feat_lengths,
                padded_input_ids=padded_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if rank == 0:
        logging.info(f'Validation Loss: {avg_loss:.4f}')

    return avg_loss


def main():
    args = get_args()

    # Enable PyTorch memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Setup logging
    logger = setup_logging(args.output_dir, rank)

    if rank == 0:
        logger.info(f'Arguments: {args}')
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
    else:
        writer = None

    # Build model
    logger.info('Building model...')

    # Create args for model
    from argparse import Namespace
    model_dir = args.model_dir 
    model_path = os.path.join(model_dir, "model.pth.tar")
    encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
    llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
    cmvn_path = os.path.join(model_dir, "cmvn.ark")
    torch.serialization.add_safe_globals([argparse.Namespace])
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    package["args"].freeze_encoder = args.freeze_encoder
    package["args"].freeze_llm = args.freeze_llm
    package["args"].use_lora = args.use_lora
    model = FireRedAsrLlm.from_args(package["args"])
    #logging.info(f'fireredasr llm model path: {model_path}, package: {package}')

    # 加载模型权重并检查
    missing_keys, unexpected_keys = model.load_state_dict(package["model_state_dict"], strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading model: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

    # 检查 encoder_projector 是否被正确初始化
    logger.info("Checking encoder_projector weights...")
    projector_has_nan = False
    for name, param in model.encoder_projector.named_parameters():
        if torch.isnan(param).any():
            logger.error(f"NaN detected in encoder_projector.{name}!")
            projector_has_nan = True
        if torch.isinf(param).any():
            logger.error(f"Inf detected in encoder_projector.{name}!")
            projector_has_nan = True
        logger.info(f"encoder_projector.{name}: shape={param.shape}, range=[{param.min():.6f}, {param.max():.6f}]")

    # 如果 encoder_projector 有 NaN，重新初始化
    if projector_has_nan or (missing_keys and any('encoder_projector' in k for k in missing_keys)):
        logger.warning("Encoder_projector has NaN or missing keys, reinitializing...")
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.encoder_projector.apply(init_weights)
        logger.info("Encoder_projector reinitialized successfully")

        # 再次检查
        for name, param in model.encoder_projector.named_parameters():
            logger.info(f"After reinit - encoder_projector.{name}: range=[{param.min():.6f}, {param.max():.6f}]")

    # 启用梯度检查点以节省显存
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()
        logger.info('Enabled gradient checkpointing for LLM')

    model.cuda()
    # print trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✓ Trainable: {name}, shape={param.shape}")
        else:
            print(f"✗ Frozen: {name}, shape={param.shape}")

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Build tokenizer and feature extractor
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(
        llm_dir,
        use_flash_attn=bool(package["args"].use_flash_attn) if hasattr(package["args"], 'use_flash_attn') else False
    )
    feat_extractor = ASRFeatExtractor(cmvn_path)

    # Build dataloaders
    logger.info('Building dataloaders...')
    train_dataloader = create_dataloader(
        data_file=args.train_data,
        tokenizer=tokenizer,
        feat_extractor=feat_extractor,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        num_workers=args.num_workers,
        shuffle=True,
        sort_by_length=True,
    )

    valid_dataloader = None
    if args.valid_data:
        valid_dataloader = create_dataloader(
            data_file=args.valid_data,
            tokenizer=tokenizer,
            feat_extractor=feat_extractor,
            batch_size=args.batch_size,
            max_text_len=args.max_text_len,
            num_workers=args.num_workers,
            shuffle=False,
            sort_by_length=False,
        )

    # Setup optimizer
    logger.info('Setting up optimizer...')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Setup GradScaler for mixed precision training
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        logger.info('Using FP16 mixed precision training with GradScaler')

    # Setup scheduler
    total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=min(args.warmup_steps / total_steps, 0.3),  # Limit to 30% of training
        anneal_strategy='cos'
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        global_step, start_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )

    # Training loop
    logger.info('Starting training...')
    best_valid_loss = 1e3 
    for epoch in range(start_epoch, args.num_epochs):
        global_step = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            writer=writer,
            global_step=global_step,
            rank=rank,
            world_size=world_size,
            scaler=scaler
        )

        # Validation
        if valid_dataloader is not None:
            valid_loss = validate(model, valid_dataloader, rank)
            if rank == 0 and writer is not None:
                writer.add_scalar('valid/loss', valid_loss, global_step)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    # Save best model
                    save_checkpoint(model, optimizer, scheduler, global_step, epoch,
                                    os.path.join(args.output_dir, 'best_model'), rank)

        # Save epoch checkpoint
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, epoch + 1,
                            args.output_dir, rank)

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        writer.close()
        logger.info('Training completed!')


if __name__ == '__main__':
    main()
