import json
import os
from dataclasses import dataclass, asdict
from itertools import chain

from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

token = os.environ["HF_TOKEN"]

N_POSITIONS = 1024


@dataclass
class ModelConfig:
    model_name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_vocab: int = 50257
    max_steps: int = 2000
    batch_size: int = 512

    def num_parameters(self, include_embedding=False) -> int:
        d_model = self.d_model
        params = 0
        if include_embedding:
            params += self.n_vocab * d_model  # embedding
            params += N_POSITIONS * d_model  # positional (not trainable)

        layer_params = 0
        d_attn = self.d_model
        layer_params += 3 * (d_model * d_attn + d_attn)  # QKV matrices
        layer_params += d_attn * d_model + d_model  # project back to model space
        d_ff = 4 * d_model
        layer_params += 2 * d_model * d_ff + d_ff + d_model  # feed forward
        layer_params += 4 * d_model  # two layer norms
        params += self.n_layers * layer_params

        params += 2 * d_model  # final layer norm
        return params

    def num_activations(self, seq_len: int = N_POSITIONS) -> int:
        return 2 * self.batch_size * seq_len * self.d_model * (self.n_layers + 1)

    def gpu_memory_gb(self, seq_len: int = N_POSITIONS, *, fp16: bool = False) -> float:
        bytes_per_el = 2 if fp16 else 4
        total_bytes = bytes_per_el * (3 * self.num_parameters() + self.num_activations(seq_len))
        return total_bytes / (1024 ** 3)  # GiB


@dataclass
class TrainingResults:
    train_loss: list[tuple[int, float]]  # list of (step, loss)
    eval_loss: list[tuple[int, float]]  # list of (step, eval_loss)


class ModelTrainer:
    def __init__(self):
        self.tokenizer = None
        self.tokenized = None

    def load_data(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        def tokenize_fn(examples):
            return self.tokenizer(examples["text"], add_special_tokens=False)

        def group_texts(examples):
            concat = list(chain(*examples["input_ids"]))
            usable = (len(concat) // N_POSITIONS) * N_POSITIONS
            blocks = [concat[i: i + N_POSITIONS]
                      for i in range(0, usable, N_POSITIONS)]
            return {"input_ids": blocks, "labels": blocks}

        splits = ["train", "validation"]
        self.tokenized = {
            split: (
                load_dataset(
                    "wikitext", "wikitext-2-raw-v1",
                    split=split, streaming=True
                )
                .map(tokenize_fn, batched=True, remove_columns=["text"])
                .map(group_texts, batched=True, remove_columns=["attention_mask"])
                .filter(lambda ex: len(ex["input_ids"]) > 0)
            )
            for split in splits
        }

    def train(self, config: ModelConfig) -> TrainingResults:
        # build model
        cfg = GPT2Config(
            n_embd=config.d_model,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            vocab_size=config.n_vocab,
            n_positions=N_POSITIONS,
        )
        model = GPT2LMHeadModel(cfg)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # training arguments
        args = TrainingArguments(
            output_dir=f"../results/{config.model_name}",
            overwrite_output_dir=True,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            eval_strategy="steps",
            eval_steps=1000,
            logging_strategy="steps",
            logging_steps=1000,
            save_strategy="no",
            max_steps=config.max_steps,
            report_to="none",
            gradient_checkpointing=True,
            # fp16=True,
            # optim="adamw_bnb_8bit",        # switch optimiser
            learning_rate=0.001,
        )

        eval_dataset = self.tokenized["train"].take(1 * config.batch_size)
        train_dataset = self.tokenized["train"].skip(1 * config.batch_size)
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        trainer.train()
        history = trainer.state.log_history

        train_loss = [(h["step"], h["train_loss"]) for h in history if "train_loss" in h]
        eval_loss = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

        return TrainingResults(train_loss=train_loss,
                               eval_loss=eval_loss)

    def train_and_save(self, config: ModelConfig):
        results = self.train(config)
        record = {
            "config": asdict(config),
            "results": {
                "train_loss": results.train_loss,
                "eval_loss": results.eval_loss,
            }
        }
        fname = f"../results/{config.model_name}_results.json"
        with open(fname, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Saved results to {fname}")

    def run_all(self):
        configs = [
            ModelConfig(model_name="gpt2_small", n_layers=12, d_model=768, n_heads=12, max_steps=10),
        ]
        for cfg in configs:
            print(f"Model size: {cfg.num_parameters()}, RAM usage: {cfg.gpu_memory_gb()} GB")
            self.load_data()
            self.train_and_save(cfg)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_all()
