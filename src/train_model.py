import json
import os
from dataclasses import dataclass, asdict

from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

token=os.environ["HF_TOKEN"]

N_POSITIONS = 1024


@dataclass
class ModelConfig:
    model_name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_vocab: int = 50257
    max_steps: int = 2000
    batch_size: int = 2


@dataclass
class TrainingResults:
    train_loss: list[tuple[int, float]]  # list of (step, loss)
    eval_loss: list[tuple[int, float]]  # list of (step, eval_loss)


class ModelTrainer:
    def __init__(self):
        # load dataset & tokenizer once
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", token=token)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        def tokenize_fn(examples):
            return self.tokenizer(examples["text"], return_special_tokens_mask=True, max_length=N_POSITIONS)

        tokenized = self.dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        # filter empty
        for split in tokenized:
            tokenized[split] = tokenized[split].filter(lambda ex: len(ex["input_ids"]) > 0)
        self.tokenized = tokenized

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
            eval_steps=100,
            logging_strategy="steps",
            logging_steps=200,
            save_strategy="no",
            max_steps=config.max_steps,
            report_to="none",
        )

        eval_dataset = self.tokenized["validation"].select(range(100 * config.batch_size))
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=self.tokenized["train"],
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
            ModelConfig(model_name="gpt2_small", n_layers=6, d_model=128, n_heads=4, max_steps=100),
            ModelConfig(model_name="gpt2_medium", n_layers=12, d_model=256, n_heads=8, max_steps=100),
            ModelConfig(model_name="gpt2_large", n_layers=24, d_model=512, n_heads=16, max_steps=100),
        ]
        for cfg in configs:
            self.train_and_save(cfg)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_all()
