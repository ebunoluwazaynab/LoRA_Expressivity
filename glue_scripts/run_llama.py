import os
import torch
import numpy as np
import random
import re
import json
import csv
import pandas as pd
from itertools import product
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import gc
from typing import Dict, Any, List, Tuple, Optional

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

_EPS = 1e-8
_SVD_K = 10 

class SRFullFTTrainer(Trainer):
    """A custom Trainer for Full Fine-Tuning that logs stable rank and SVD."""
    def __init__(self, *args, pretrained_weights: Dict[str, torch.Tensor] = None, **kwargs):
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

        # Call the unified function to get all data
        sr_data = _get_lora_sr_and_svd(self.model, iters=20, k=_SVD_K)

        if sr_data:
            sr_rows = [[epoch, step] + row for row in sr_data]
            _append_to_csv(self.args.output_dir, "sr_layerwise.csv", sr_rows, ["epoch", "global_step", "layer", "metric_type", "value"])
        # if svd_data:
        #     svd_rows = [[epoch, step] + row for row in svd_data]
        #     _append_to_csv(self.args.output_dir, "svd_layerwise.csv", svd_rows, ["epoch", "global_step", "layer", "metric_type", "value"])

        return metrics


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


def _append_to_csv(output_dir: str, filename: str, rows: List[List[Any]], header: List[str]):
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    file_exists = os.path.exists(path)
    
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerows(rows)

def _get_lora_sr_and_svd(model, iters: int = 20, k: int = 10):
    sr_data = []
    svd_data = []

    for name, p in model.named_parameters():
        if not name.endswith("weight") or p.ndim < 2 or "LayerNorm" in name:
            continue

        W_base = _as_2d(p.data).cpu()
        clean_name = name.replace(".base_layer.weight", ".weight")

        # Calculate metrics for the base layer
        sr_data.append([clean_name, "base", _stable_rank(W_base, iters=iters)])
        # try:
        #     _, S, _ = torch.linalg.svd(W_base)
        #     for i, sv in enumerate(S[:k].tolist()):
        #         svd_data.append([clean_name, "base", f"svd_{i+1}", sv])
        # except Exception as e:
        #     print(f"Warning: SVD failed for base layer {clean_name}: {e}")

        # Check for LoRA adapter and calculate metrics for delta and effective weights
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
                            scale = parent_mod.scaling[adp]
                            delta = (B @ A) * scale

                            if total_delta is None:
                                total_delta = delta
                            else:
                                total_delta += delta
                        except Exception as e:
                            print(f"Warning: Error processing adapter '{adp}' for {parent_name}: {e}")

                if total_delta is not None and total_delta.shape == W_base.shape:
                    # Calculate metrics for delta weights
                    sr_data.append([clean_name, "delta", _stable_rank(total_delta, iters=iters)])
                    # try:
                    #     _, S_delta, _ = torch.linalg.svd(total_delta)
                    #     for i, sv in enumerate(S_delta[:k].tolist()):
                    #         svd_data.append([clean_name, "delta", f"svd_{i+1}", sv])
                    # except Exception as e:
                    #     print(f"Warning: SVD failed for delta layer {clean_name}: {e}")

                    # Calculate metrics for effective weights
                    effective_W = W_base + total_delta
                    sr_data.append([clean_name, "effective", _stable_rank(effective_W, iters=iters)])
                    # try:
                    #     _, S_eff, _ = torch.linalg.svd(effective_W)
                    #     for i, sv in enumerate(S_eff[:k].tolist()):
                    #         svd_data.append([clean_name, "effective", f"svd_{i+1}", sv])
                    # except Exception as e:
                    #     print(f"Warning: SVD failed for effective layer {clean_name}: {e}")
    return sr_data
    
   

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

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def compute_metrics_for_gsm8k(eval_pred, tokenizer):
    predictions, labels = eval_pred

    # decode generated sequences
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # decode labels (replace -100 with pad so we can decode)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def _extract_number(text: str):
        """
        Robust numeric extractor:
        1) Prefer '#### number' (GSM8K convention)
        2) Try common math formatting: \boxed{...}, $...$
        3) Fallback: last number in the string
        Returns float or None.
        """
        if not text:
            return None
        t = text.replace(",", " ").strip()
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", t)
        if m:
            return float(m.group(1))
        m = re.search(r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}", t)
        if m:
            return float(m.group(1))
        m = re.search(r"\$\s*(-?\d+(?:\.\d+)?)\s*\$", t)
        if m:
            return float(m.group(1))
        nums = re.findall(r"-?\d+(?:\.\d+)?", t)
        if nums:
            return float(nums[-1])

        return None

    correct = 0
    total = len(decoded_preds)

    for pred, label in zip(decoded_preds, decoded_labels):
        pa = _extract_number(pred)
        la = _extract_number(label)
        if pa is not None and la is not None:
            if abs(pa - la) < 1e-6:
                correct += 1

    acc = correct / total if total else 0.0
    return {"accuracy": acc}

def get_output_dir(model_name, method, lora_r=None, lora_alpha=None, lr=None):
    clean_model_name = model_name.split('/')[-1]
    base_dir = f"experiments/{clean_model_name}"
    
    if method == "full_ft":
        return f"{base_dir}/full_ft_lr{lr:.2e}"
    elif method == "lora":
        return f"{base_dir}/lora_r{lora_r}_alpha{lora_alpha}_lr{lr:.2e}"
    elif method == "rslora":
        return f"{base_dir}/rslora_r{lora_r}_alpha{lora_alpha}_lr{lr:.2e}"
    return f"{base_dir}/unknown_method"


def run_experiment(model_config: Dict[str, Any], lora_config: Dict[str, Any], ft_config: Dict[str, Any], experiment_config: Dict[str, Any]):
    model_name_or_path = model_config["model_name"]
    dataset_name = model_config["dataset"]
    experiment_type = experiment_config["type"]
    batch_size = experiment_config["per_device_train_batch_size"]
    epochs = experiment_config["num_train_epochs"]

    print(f"Loading model, tokenizer, and dataset for {experiment_type}...")
    if experiment_type == "full_ft":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.config.use_cache = False
        trainer_class = SRFullFTTrainer

        pretrained_weights = {name: param.data.clone().cpu() for name, param in model.named_parameters() 
                              if name.endswith("weight") and param.ndim >= 2 and "LayerNorm" not in name}

    elif experiment_type in ["lora", "rslora"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        trainer_class = SRLoRATrainer
        pretrained_weights = None
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Define tokenize function
    def tokenize_function(examples):
        prompts = [f"Question: {q}\nAnswer: " for q in examples["question"]]
        full_texts = [f"{q}\nAnswer: {a}{tokenizer.eos_token}" for q, a in zip(examples["question"], examples["answer"])]
        
        # Tokenize full sequences first
        full_tokens = tokenizer(full_texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        
        # Now, tokenize just the prompts to find their lengths for masking
        prompt_tokens = tokenizer(prompts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        
        labels = full_tokens["input_ids"].clone()
        
        # Correctly mask the prompt part of the labels
        for i in range(len(prompts)):
            # Find the length of the prompt for this specific example
            prompt_len = prompt_tokens["attention_mask"][i].sum().item()
            labels[i, :prompt_len] = -100
            
        full_tokens["labels"] = labels
        return full_tokens

    # Load and process the datasets
    if dataset_name == "gsm8k":
        train_dataset_raw = load_dataset("gsm8k", "main", split="train")
        eval_dataset_raw = load_dataset("gsm8k", "main", split="test[:1000]")
    else:
        full_dataset = load_dataset(dataset_name, split="train")
        train_size = int(0.9 * len(full_dataset))
        train_dataset_raw = full_dataset.select(range(train_size))
        eval_dataset_raw = full_dataset.select(range(train_size, len(full_dataset)))
    
    train_dataset = train_dataset_raw.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset_raw.column_names
    )
    eval_dataset = eval_dataset_raw.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset_raw.column_names
    )
    
    peft_config = None
    if experiment_type == "lora":
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            **lora_config
        )
        model = get_peft_model(model, peft_config)
    elif experiment_type == "rslora":
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_rslora=True,
            **lora_config
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    output_dir = experiment_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({"model_config": model_config, "lora_config": lora_config, "ft_config": ft_config, "experiment_config": experiment_config}, f, indent=4)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        save_total_limit=1,
        logging_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=256,
        push_to_hub=False,
        **ft_config
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "compute_metrics": lambda p: compute_metrics_for_gsm8k(p, tokenizer),
    }
    if experiment_type == "full_ft":
        trainer_kwargs["pretrained_weights"] = pretrained_weights
    
    trainer = trainer_class(**trainer_kwargs)

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    trainer.save_model(os.path.join(output_dir, "final_model"))
    print("Final model saved.")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    config = {
        "tasks": ["gsm8k"],
        "models": ["meta-llama/Llama-2-7b-hf"],
        "learning_rates": np.logspace(np.log10(1e-5), np.log10(5e-4), 5).tolist(),
        "lora_ranks": [4, 8, 16, 32],
        "alpha_multiplier": 16,
        "methods": ["lora", "rslora", "full_ft"],
        "epochs": 5,
        "batch_size": 16,
    }

    model_name = config["models"][0]
    task_name = config["tasks"][0]

    for method in config["methods"]:
        if method == "full_ft":
            for lr in config["learning_rates"]:
                print(f"Running Full FT experiment with LR={lr:.2e}")
                output_dir = get_output_dir(model_name, method, lr=lr)
                
                model_config = {"model_name": model_name, "dataset": task_name}
                lora_config = {}
                ft_config = {"learning_rate": lr}
                exp_config = {
                    "type": "full_ft",
                    "output_dir": output_dir,
                    "per_device_train_batch_size": config["batch_size"],
                    "num_train_epochs": config["epochs"],
                }
                
                run_experiment(model_config, lora_config, ft_config, exp_config)
                clear_gpu_memory()

        elif method in ["lora", "rslora"]:
            for r, lr in product(config["lora_ranks"], config["learning_rates"]):
                alpha = r * config["alpha_multiplier"]
                print(f"Running {method} experiment with r={r}, alpha={alpha}, LR={lr:.2e}")
                
                output_dir = get_output_dir(model_name, method, lora_r=r, lora_alpha=alpha, lr=lr)
                
                model_config = {"model_name": model_name, "dataset": task_name}
                lora_config = {"r": r, "lora_alpha": alpha, "lora_dropout": 0.05}
                ft_config = {"learning_rate": lr}
                exp_config = {
                    "type": method,
                    "output_dir": output_dir,
                    "per_device_train_batch_size": config["batch_size"],
                    "num_train_epochs": config["epochs"],
                }
                
                run_experiment(model_config, lora_config, ft_config, exp_config)
                clear_gpu_memo