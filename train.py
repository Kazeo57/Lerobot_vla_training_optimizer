#!/usr/bin/env python3
"""
Script d'entraînement VLA optimisé pour réduire l'usage GPU
Optimisations principales :
- Gradient accumulation pour réduire la batch size
- Mixed precision training (FP16/BF16)
- Gradient checkpointing
- DataLoader optimisé
- Memory efficient attention
- Dynamic padding
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
from typing import Dict, Any, Optional
import gc
import psutil
from contextlib import contextmanager

# Configuration optimisée pour les ressources limitées
class OptimizedTrainingConfig:
    def __init__(self):
        # Paramètres de batch optimisés
        self.micro_batch_size = 1  # Très petit batch par forward pass
        self.gradient_accumulation_steps = 32  # Simule un batch de 32
        self.effective_batch_size = self.micro_batch_size * self.gradient_accumulation_steps
        
        # Optimisations mémoire
        self.use_mixed_precision = True
        self.precision_type = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
        self.gradient_checkpointing = True
        self.pin_memory = False  # Désactiver si RAM limitée
        self.num_workers = min(2, os.cpu_count())  # Limiter les workers
        
        # Optimisations attention
        self.use_flash_attention = True
        self.use_memory_efficient_attention = True
        
        # Scheduling et monitoring
        self.max_grad_norm = 1.0
        self.warmup_ratio = 0.1
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        
        # Checkpointing intelligent
        self.save_steps = 1000
        self.eval_steps = 500
        self.logging_steps = 50
        self.max_checkpoints_to_keep = 2

class MemoryEfficientDataLoader:
    """DataLoader avec optimisations mémoire"""
    
    def __init__(self, dataset, config: OptimizedTrainingConfig):
        self.dataset = dataset
        self.config = config
        
    def create_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            persistent_workers=self.config.num_workers > 0,
            collate_fn=self._memory_efficient_collate
        )
    
    def _memory_efficient_collate(self, batch):
        """Collate function optimisée pour la mémoire"""
        # Dynamic padding instead of max padding
        max_seq_len = max(len(item['input_ids']) for item in batch)
        
        collated = {}
        for key in batch[0].keys():
            if key in ['input_ids', 'attention_mask']:
                # Pad only to the maximum length in this batch
                collated[key] = torch.stack([
                    torch.cat([
                        item[key], 
                        torch.zeros(max_seq_len - len(item[key]), dtype=item[key].dtype)
                    ]) for item in batch
                ])
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        
        return collated

class OptimizedVLATrainer:
    def __init__(self, model, config: OptimizedTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Optimiseur avec paramètres optimisés
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),  # Betas optimisés pour les transformers
            eps=1e-8
        )
        
        # Gradient checkpointing
        if config.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        self.global_step = 0
        self.accumulated_loss = 0.0
        
    @contextmanager
    def memory_cleanup(self):
        """Context manager pour nettoyer la mémoire"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_usage(self):
        """Monitorer l'usage mémoire"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
        else:
            gpu_memory = gpu_cached = 0.0
        
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        
        return {
            'gpu_allocated': gpu_memory,
            'gpu_cached': gpu_cached,
            'cpu_used': cpu_memory
        }
    
    def forward_step(self, batch):
        """Forward pass optimisé"""
        with autocast(enabled=self.config.use_mixed_precision):
            # Déplacer le batch sur GPU de manière efficace
            batch = {k: v.to(self.device, non_blocking=True) 
                    for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            outputs = self.model(**batch)
            loss = outputs.loss / self.config.gradient_accumulation_steps
            
        return loss, outputs
    
    def backward_step(self, loss):
        """Backward pass optimisé"""
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def optimizer_step(self):
        """Optimizer step avec gradient clipping"""
        if self.scaler:
            # Gradient clipping avec mixed precision
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)  # Plus efficace que zero_grad()
    
    def train_epoch(self, dataloader):
        """Boucle d'entraînement optimisée"""
        self.model.train()
        
        for step, batch in enumerate(dataloader):
            with self.memory_cleanup():
                # Forward pass
                loss, outputs = self.forward_step(batch)
                self.accumulated_loss += loss.item()
                
                # Backward pass
                self.backward_step(loss)
                
                # Optimizer step (gradient accumulation)
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = self.accumulated_loss / self.config.gradient_accumulation_steps
                        memory_stats = self.get_memory_usage()
                        
                        logging.info(
                            f"Step {self.global_step}: "
                            f"Loss={avg_loss:.4f}, "
                            f"GPU={memory_stats['gpu_allocated']:.1f}GB, "
                            f"CPU={memory_stats['cpu_used']:.1f}GB"
                        )
                    
                    self.accumulated_loss = 0.0
                    self.global_step += 1
                    
                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()
    
    def save_checkpoint(self):
        """Sauvegarde optimisée des checkpoints"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, f'checkpoint_step_{self.global_step}.pt')
        
        # Nettoyer les anciens checkpoints
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        """Supprimer les anciens checkpoints pour économiser l'espace"""
        checkpoints = [f for f in os.listdir('.') if f.startswith('checkpoint_step_')]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        while len(checkpoints) > self.config.max_checkpoints_to_keep:
            os.remove(checkpoints.pop(0))

def main():
    """Fonction principale d'entraînement"""
    # Configuration
    config = OptimizedTrainingConfig()
    
    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Model loading avec optimisations
    logger.info("Loading model...")
    model = load_model()  # Votre fonction de chargement
    
    # Optimisations du modèle
    if hasattr(model, 'config'):
        # Activer les optimisations d'attention si disponibles
        if hasattr(model.config, 'use_flash_attention_2'):
            model.config.use_flash_attention_2 = config.use_flash_attention
    
    # Dataset et DataLoader
    logger.info("Setting up data...")
    dataset = load_dataset()  # Votre fonction de chargement
    data_loader_manager = MemoryEfficientDataLoader(dataset, config)
    dataloader = data_loader_manager.create_dataloader()
    
    # Trainer
    trainer = OptimizedVLATrainer(model, config)
    
    # Training loop
    logger.info("Starting training...")
    try:
        for epoch in range(10):  # Ajuster selon vos besoins
            logger.info(f"Epoch {epoch + 1}/10")
            trainer.train_epoch(dataloader)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        trainer.save_checkpoint()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()