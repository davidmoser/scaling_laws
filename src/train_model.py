import json
import os
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

from src.model_config import ModelConfig, TrainingResults

token = os.environ["HF_TOKEN"]
has_cuda = torch.cuda.is_available()


class ModelTrainer:
    def __init__(self):
        self.tokenizer = None
        self.tokenized = None

    def load_data(self, n_positions):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        def tokenize_fn(examples):
            return self.tokenizer(examples["text"], add_special_tokens=False,
                                  return_attention_mask=False, return_special_tokens_mask=False)

        def group_texts(examples):
            concat = list(chain(*examples["input_ids"]))
            usable = (len(concat) // n_positions) * n_positions
            blocks = [concat[i: i + n_positions]
                      for i in range(0, usable, n_positions)]
            return {"input_ids": blocks, "labels": blocks}

        raw = load_dataset("allenai/c4", "en", streaming=True, token=token)
        cols = raw["train"].column_names
        raw = raw.map(tokenize_fn, batched=True)
        raw = raw.remove_columns(cols)
        raw = raw.map(group_texts, batched=True)
        self.tokenized = raw

    def train(self, config: ModelConfig) -> TrainingResults:
        cfg = GPT2Config(
            n_embd=config.d_model,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            vocab_size=config.n_vocab,
            n_positions=config.n_positions,
            use_cache=False,
            use_bfloat16=True,
            attn_implementation="flash_attention_2" if has_cuda else "eager",
        )
        model = GPT2LMHeadModel(cfg)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir=f"../results/{config.model_name}",
            overwrite_output_dir=True,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            eval_strategy="steps",
            eval_steps=100,
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="no",
            max_steps=config.max_steps,
            report_to="none",
            gradient_checkpointing=True,
            learning_rate=0.001,
            bf16=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.tokenized["train"],
            eval_dataset=self.tokenized["validation"].take(10),
            data_collator=data_collator
        )

        trainer.train()
        history = trainer.state.log_history

        train_loss = [(h["step"], h["loss"]) for h in history if "loss" in h]
        eval_loss = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

        return TrainingResults(config=config, train_loss=train_loss, eval_loss=eval_loss)

    def train_and_save(self, config: ModelConfig):
        results = self.train(config)
        fname = f"../results/{config.model_name}_results.json"
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {fname}")

    def run_all(self):
        configs = [
            ModelConfig(model_name="gpt2_small", n_layers=12, d_model=512, n_heads=8, max_steps=30000),
        ]
        for cfg in configs:
            print(f"Model size: {cfg.num_parameters()}, RAM usage: {cfg.gpu_memory_gb()} GB")
            self.load_data(cfg.n_positions)
            self.train_and_save(cfg)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_all()
