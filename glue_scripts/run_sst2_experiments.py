import os
import torch
import numpy as np
import random
from copy import deepcopy
from itertools import product
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import csv
import re
import pandas as pd
import plotly.express as px

import gc

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---- Stable Rank & SVD Utilities ----------------------------------------------
_EPS = 1e-8
_SVD_K = 10  # Number of top singular values to store

def _power_iter_sigma_max(W: torch.Tensor, iters: int = 30) -> float:
    with torch.no_grad():
        W = W.float()
        torch.manual_seed(42)
        v = torch.randn(W.shape[1], device=W.device)
        v = v / (v.norm() + _EPS)
        for _ in range(iters):
            u = (W @ v)
            u = u / (u.norm() + _EPS)
            v = (W.t() @ u)
            v = v / (v.norm() + _EPS)
        return float((u @ (W @ v)).abs().item())

def _stable_rank(W: torch.Tensor, iters: int = 20) -> float:
    W = W.float()
    fro2 = float(torch.linalg.matrix_norm(W, ord="fro")**2)
    try:
        smax = _power_iter_sigma_max(W, iters=iters)
    except RuntimeError:
        smax = _power_iter_sigma_max(W.cpu(), iters=iters)
    return float(fro2 / (smax**2 + _EPS))

def _as_2d(t: torch.Tensor) -> torch.Tensor:
    return t if t.ndim == 2 else t.view(t.shape[0], -1)

def _adapter_names(mod):
    if hasattr(mod, "active_adapter"):
        act = mod.active_adapter
        if isinstance(act, str):
            return [act]
        if isinstance(act, (list, tuple, set)):
            return list(act)
    if hasattr(mod, "active_adapters") and mod.active_adapters:
        return list(mod.active_adapters)
    if hasattr(mod, "lora_A") and hasattr(mod.lora_A, 'keys'):
        keys = list(mod.lora_A.keys())
        if keys:
            return keys
    return ["default"]

def _append_to_csv(output_dir, filename, rows, header):
    path = os.path.join(output_dir, filename)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerows(rows)

def _get_sr(model, iters: int = 20):
    sr_data = []
    
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue
            
        if not name.endswith("weight") or p.ndim < 2 or "LayerNorm" in name:
            continue

        W = _as_2d(p.data).cpu()
        clean_name = name.replace(".base_layer.weight", ".weight")
        sr_data.append([clean_name, "base", _stable_rank(W, iters=iters)])

        if "base_layer.weight" in name:
            parent_name = name.replace(".base_layer.weight", "")
            try:
                parent_mod = model.get_submodule(parent_name)
            except Exception:
                continue

            if hasattr(parent_mod, "lora_A") and hasattr(parent_mod, "lora_B"):
                total_delta = None
                adapter_names = _adapter_names(parent_mod)
                
                for adp in adapter_names:
                    if adp in parent_mod.lora_A and adp in parent_mod.lora_B:
                        try:
                            A = parent_mod.lora_A[adp].weight.cpu()
                            B = parent_mod.lora_B[adp].weight.cpu()
                            
                            r = parent_mod.r[adp] if isinstance(parent_mod.r, dict) else parent_mod.r
                            alpha = parent_mod.lora_alpha[adp] if isinstance(parent_mod.lora_alpha, dict) else parent_mod.lora_alpha
                            
                            # Determine scaling based on adapter configuration
                            use_rslora_flag = getattr(parent_mod, 'peft_config', {}).get(adp, {}).get('use_rslora', False)
                            # scale = alpha / np.sqrt(r) if use_rslora_flag else alpha/r
                            scale = parent_mod.scaling[adp] 
                            delta = (B @ A) * scale
                            
                            if total_delta is None:
                                total_delta = delta
                            else:
                                total_delta = total_delta + delta
                        except Exception as e:
                            print(f"Warning: Error processing adapter '{adp}' for {parent_name}: {e}")
                            continue
                
                if total_delta is not None and total_delta.shape == W.shape:
                    sr_data.append([clean_name, "delta", _stable_rank(total_delta, iters=iters)])
                    effective_W = W + total_delta
                    sr_data.append([clean_name, "effective", _stable_rank(effective_W, iters=iters)])
    return sr_data

def _get_svd(model, k: int = 10):
    svd_data = []
    
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue
        if not name.endswith("weight") or p.ndim < 2 or "LayerNorm" in name:
            continue
        
        W = _as_2d(p.data).cpu()
        clean_name = name.replace(".base_layer.weight", ".weight")
        
        # SVD for base weights
        # try:
        #     _, S, _ = torch.linalg.svd(W)
        #     for i, sv in enumerate(S[:k].tolist()):
        #         svd_data.append([clean_name, "base", f"svd_{i+1}", sv])
        # except Exception as e:
        #     print(f"Warning: SVD failed for base layer {clean_name}: {e}")

        if "base_layer.weight" in name:
            parent_name = name.replace(".base_layer.weight", "")
            try:
                parent_mod = model.get_submodule(parent_name)
            except Exception:
                continue

            if hasattr(parent_mod, "lora_A") and hasattr(parent_mod, "lora_B"):
                total_delta = None
                adapter_names = _adapter_names(parent_mod)
                
                for adp in adapter_names:
                    if adp in parent_mod.lora_A and adp in parent_mod.lora_B:
                        try:
                            A = parent_mod.lora_A[adp].weight.cpu()
                            B = parent_mod.lora_B[adp].weight.cpu()
                            
                            r = parent_mod.r[adp] if isinstance(parent_mod.r, dict) else parent_mod.r
                            alpha = parent_mod.lora_alpha[adp] if isinstance(parent_mod.lora_alpha, dict) else parent_mod.lora_alpha
                            
                            # Determine scaling based on adapter configuration
                            use_rslora_flag = getattr(parent_mod, 'peft_config', {}).get(adp, {}).get('use_rslora', False)
                            # scale = alpha / np.sqrt(r) if use_rslora_flag else alpha/r
                            scale = parent_mod.scaling[adp] 
                            
                            delta = (B @ A) * scale
                            
                            if total_delta is None:
                                total_delta = delta
                            else:
                                total_delta = total_delta + delta
                        except Exception as e:
                            print(f"Warning: Error processing adapter '{adp}' for {parent_name}: {e}")
                            continue

                if total_delta is not None and total_delta.shape == W.shape:
                    # SVD for delta matrix
                    # try:
                    #     _, S, _ = torch.linalg.svd(total_delta)
                    #     for i, sv in enumerate(S[:k].tolist()):
                    #         svd_data.append([clean_name, "delta", f"svd_{i+1}", sv])
                    # except Exception as e:
                    #     print(f"Warning: SVD failed for delta layer {clean_name}: {e}")
                        
                    # SVD for effective weight
                    effective_W = W + total_delta
                    # try:
                    #     _, S, _ = torch.linalg.svd(effective_W)
                    #     for i, sv in enumerate(S[:k].tolist()):
                    #         svd_data.append([clean_name, "effective", f"svd_{i+1}", sv])
                    # except Exception as e:
                    #     print(f"Warning: SVD failed for effective layer {clean_name}: {e}")
    return svd_data

def _get_full_ft_sr_and_svd(model, pretrained_weights, iters: int = 20, k: int = 10):
    sr_data = []
    svd_data = []
    
    for name, p in model.named_parameters():
        if not name.endswith("weight") or p.ndim < 2 or "LayerNorm" in name:
            continue
            
        W_current = _as_2d(p.data).cpu()
        sr_data.append([name, "finetuned", _stable_rank(W_current, iters=iters)])
        
        # try:
        #     _, S, _ = torch.linalg.svd(W_current)
        #     for i, sv in enumerate(S[:k].tolist()):
        #         svd_data.append([name, "finetuned", f"svd_{i+1}", sv])
        # except Exception as e:
        #     print(f"Warning: SVD failed for finetuned layer {name}: {e}")

        if pretrained_weights and name in pretrained_weights:
            W_pretrained = _as_2d(pretrained_weights[name]).cpu()
            sr_data.append([name, "pretrained", _stable_rank(W_pretrained, iters=iters)])
            
            # try:
            #     _, S, _ = torch.linalg.svd(W_pretrained)
            #     for i, sv in enumerate(S[:k].tolist()):
            #         svd_data.append([name, "pretrained", f"svd_{i+1}", sv])
            # except Exception as e:
            #     print(f"Warning: SVD failed for pretrained layer {name}: {e}")
    
    return sr_data

class SRFullFTTrainer(Trainer):
    def __init__(self, *args, pretrained_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_weights = pretrained_weights
        
    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        epoch = float(self.state.epoch) if self.state.epoch is not None else None
        step = int(self.state.global_step)
        
        sr_data = _get_full_ft_sr_and_svd(self.model, self.pretrained_weights, iters=20, k=_SVD_K)
    
        if sr_data:
            sr_rows = [[epoch, step] + row for row in sr_data]
            _append_to_csv(self.args.output_dir, "sr_layerwise.csv", sr_rows, ["epoch", "global_step", "layer", "metric_type", "value"])
        # if svd_data:
        #     svd_rows = [[epoch, step] + row for row in svd_data]
        #     _append_to_csv(self.args.output_dir, "svd_layerwise.csv", svd_rows, ["epoch", "global_step", "layer", "metric_type", "value"])
            
        return metrics

class SRLoRATrainer(Trainer):
    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)
        epoch = float(self.state.epoch) if self.state.epoch is not None else None
        step = int(self.state.global_step)
        
        sr_data = _get_sr(self.model, iters=20)
        # svd_data = _get_svd(self.model, k=_SVD_K)
        
        if sr_data:
            sr_rows = [[epoch, step] + row for row in sr_data]
            _append_to_csv(self.args.output_dir, "sr_layerwise.csv", sr_rows, ["epoch", "global_step", "layer", "metric_type", "value"])
        # if svd_data:
        #     svd_rows = [[epoch, step] + row for row in svd_data]
        #     _append_to_csv(self.args.output_dir, "svd_layerwise.csv", svd_rows, ["epoch", "global_step", "layer", "metric_type", "value"])
            
        return metrics

# ---- Configuration & Main Loop ------------------------------------------------

task_config = {
    "sst2": {"num_labels": 2, "sentence_columns": ["sentence"]},
    "cola": {"num_labels": 2, "sentence_columns": ["sentence"]},
    "mrpc": {"num_labels": 2, "sentence_columns": ["sentence1", "sentence2"]},
    "qnli": {"num_labels": 2, "sentence_columns": ["question", "sentence"]},
    "rte": {"num_labels": 2, "sentence_columns": ["sentence1", "sentence2"]},
    "mnli": {"num_labels": 3, "sentence_columns": ["premise", "hypothesis"]},
}

model_config = {
    "roberta-base": {
        "target_modules": ["query", "key", "value", "output.dense", "intermediate.dense"],
    },
    "roberta-large": {
        "target_modules": ["query", "key", "value", "output.dense", "intermediate.dense"],
    },
    "bert-base-uncased": {
        "target_modules": ["query", "key", "value", "attention.output.dense", "intermediate.dense"],
    },
}

# config = {
#     "tasks": ["sst2", "cola", "mrpc", "qnli", "rte", "mnli"],
#     "models": ["roberta-base"],
#     "learning_rates": np.logspace(np.log10(1e-5), np.log10(1e-4), 5), # List for lora LR sweep # np.logspace(np.log10(1e-4), np.log10(1e-2), 5)
#     "lora_ranks": [1, 4, 8, 16, 32, 64, 128, 256, 512, 768], # List for rank sweep
#     "lora_alpha": 32, # Fixed alpha
#     "methods": ["lora", "full_finetuning", "rslora"],
#     "epochs": 5,
#     "batch_size": 32,
#     "logging_strategy": "epoch", 
#     "logging_steps": 100, 
# }
config = {
    "tasks": ["sst2"],
    "models": ["roberta-base"],
    "learning_rates": np.logspace(np.log10(1e-4), np.log10(1e-2), 5),
    "lora_ranks": [512, 768], # List for rank sweep
    "alpha_multiplier": 1,
    "methods": ["lora", "rslora"],
    "epochs": 5,
    "batch_size": 32,
    "logging_strategy": "epoch", 
    "logging_steps": 100, 
}

def get_output_dir(task, model_name, method, lora_r=None, lora_alpha=None, lr=None):
    base_dir = f"sst2/{model_name}"
    if method == "full_finetuning":
        return f"{base_dir}/full_finetuning_lr{lr}"
    elif method == "lora":
        return f"{base_dir}/lora_r{lora_r}_alpha{lora_alpha}_lr{lr}"
    elif method == "rslora":
        return f"{base_dir}/rslora_r{lora_r}_alpha{lora_alpha}_lr{lr}"
    return f"{base_dir}/unknown"


def run_experiment(task, model_name, method, lora_r=None, lora_alpha=None, lr=None):
    print(f"\n Running experiment for Task: {task}, Model: {model_name}, Method: {method}...")
    
    task_info = task_config[task]
    num_labels = task_info["num_labels"]
    sentence_cols = task_info["sentence_columns"]
    
    ds = load_dataset("glue", task)
    tok = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(batch):
        if len(sentence_cols) == 1:
            return tok(batch[sentence_cols[0]], truncation=True, padding="max_length", max_length=128)
        else:
            return tok(batch[sentence_cols[0]], batch[sentence_cols[1]], truncation=True, padding="max_length", max_length=128)
    
    ds_tok = ds.map(tokenize, batched=True)
    ds_tok = ds_tok.rename_column("label", "labels")
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    if task == "mnli":
        eval_ds = ds_tok["validation_matched"]
        print("Using validation_matched for MNLI evaluation.")
    else:
        eval_ds = ds_tok["validation"]
    
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    metric = evaluate.load("glue", task)
    
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    output_dir = get_output_dir(task, model_name, method, lora_r, lora_alpha, lr)
    
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"]*2,
        num_train_epochs=config["epochs"],
        learning_rate=lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",   
        warmup_ratio=0.1, 
        eval_strategy=config["logging_strategy"],
        logging_strategy=config["logging_strategy"],
        logging_steps=config["logging_steps"] if config["logging_strategy"] == "steps" else None,
        save_strategy=config["logging_strategy"],
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none",
        fp16=True if torch.cuda.is_available() else False,
    )
    
    if method == "full_finetuning":
        pretrained_weights = {name: param.data.clone().cpu() for name, param in base_model.named_parameters() 
                              if name.endswith("weight") and param.ndim >= 2 and "LayerNorm" not in name}
        trainer = SRFullFTTrainer(
            model=base_model,
            args=args,
            train_dataset=ds_tok["train"],
            eval_dataset=eval_ds,
            tokenizer=tok,
            compute_metrics=compute_metrics,
            pretrained_weights=pretrained_weights,
        )
    elif method == "lora":
        target_modules = model_config[model_name]["target_modules"]
        lora_cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            use_rslora=False,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        trainer = SRLoRATrainer(
            model=peft_model,
            args=args,
            train_dataset=ds_tok["train"],
            eval_dataset=eval_ds,
            tokenizer=tok,
            compute_metrics=compute_metrics,
        )
    elif method == "rslora":
        target_modules = model_config[model_name]["target_modules"]
        lora_cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            use_rslora=True,
        )
        peft_model = get_peft_model(base_model, lora_cfg)
        trainer = SRLoRATrainer(
            model=peft_model,
            args=args,
            train_dataset=ds_tok["train"],
            eval_dataset=eval_ds,
            tokenizer=tok,
            compute_metrics=compute_metrics,
        )
    else:
        print(f"Unknown method: {method}. Skipping.")
        return

    trainer.train()
    
    print("Final evaluation...")
    # trainer.evaluate()

    logs = pd.DataFrame(trainer.state.log_history)
    train_df = logs.loc[logs["loss"].notna(), ["epoch","loss"]].rename(columns={"loss":"train_loss"})
    eval_df = logs.loc[logs["eval_loss"].notna(), [c for c in logs.columns if c.startswith("eval_") or c == "epoch"]]
    eval_df = eval_df.rename(columns=lambda c: c.replace("eval_",""))
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_per_epoch.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, "eval_per_epoch.csv"), index=False)
    
    print(f"Experiment finished. Results saved to {output_dir}\n")
    return trainer

if __name__ == "__main__":
    for task in config["tasks"]:
        for model_name in config["models"]:
            for method in config["methods"]:
                # Full Fine-tuning: LR sweep only
                if method == "full_finetuning":
                    for lr in config["learning_rates"]:
                         run_experiment(task, model_name, method, lr=lr)
                         clear_gpu_memory()
                # LoRA & RSLora: LR and Rank sweep, alpha is manually set
                elif method in ["lora", "rslora"]:
                    for r in config["lora_ranks"]:
                        for lr in config["learning_rates"]:
                            alpha = r * config["alpha_multiplier"]
                            print(f"Using r={r}, alpha={alpha}, lr={lr}")
                            run_experiment(task, model_name, method, lora_r=r, lora_alpha=alpha, lr=lr)
                            clear_gpu_memory()