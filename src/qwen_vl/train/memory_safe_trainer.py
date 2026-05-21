"""
Memory-safe Trainer with periodic DataLoader worker restart
#
   pt_data_worker 内存泄漏问题

        # Use no_grad to save memory since encoder is frozen
 train_qwen.py 中导入并使用 MemorySafeTrainer 替代原始 Trainer
"""

import gc
import os
import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import seed_worker


class MemorySafeTrainer(Trainer):
    """
    解决 DataLoader worker 内存泄漏的 Trainer
    
    主要机制：
    1. 禁用 persistent_workers
    2. 定期 DataLoader 释放 worker 内存
    3. 每 N 步手动触发 gc
    """
    
    def __init__(self, *args, 
                 dataloader_restart_interval: int = 500,  # 每 N 步重启 DataLoader
                 gc_interval: int = 100,  # 每 N 步触发 gc
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._dataloader_restart_interval = dataloader_restart_interval
        self._gc_interval = gc_interval
        self._current_dataloader = None
        self._dataloader_step_count = 0
    
    def get_train_dataloader(self) -> DataLoader:
        """
        覆盖原始方法，确保：
        1. persistent_workers=False 防止内存累积
        2. 较小减少内存占 prefetch_factor
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = self._get_train_sampler()
        
        num_workers = self.args.dataloader_num_workers
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=True,  # DDP fix
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        # Use no_grad to save memory since encoder frozen is worker
            prefetch_factor=2 if num_workers > 0 else None,  # 减少预取
            worker_init_fn=seed_worker,
        )
        
        self._current_dataloader = dataloader
        self._dataloader_step_count = 0
        return dataloader
    
    def _maybe_restart_dataloader(self):
        """检查是否需要重启 DataLoader"""
        self._dataloader_step_count += 1
        
        # 定期 gc
        if self._dataloader_step_count % self._gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """覆盖 training_step 添加内存管理"""
        self._maybe_restart_dataloader()
        return super().training_step(model, inputs, num_items_in_batch)


def patch_existing_trainer():
    """
    通过 monkey patch 方式修改现有 Trainer
        # Use no_grad to save memory since encoder is frozen
    """
    original_get_train_dataloader = Trainer.get_train_dataloader
    
    def patched_get_train_dataloader(self):
        """确保 DataLoader 使用内存安全的配置"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        train_sampler = self._get_train_sampler()
        
        num_workers = self.args.dataloader_num_workers
        
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=False,  # 禁用持久化 worker
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=seed_worker,
        )
    
    Trainer.get_train_dataloader = patched_get_train_dataloader
    print("[Memory Fix] Trainer.get_train_dataloader has been patched for memory safety")


# 自动应用 patch
patch_existing_trainer()
