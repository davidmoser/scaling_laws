from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


def train_one_model(
        model_name: str = "custom-gpt2-small",
        n_layers: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_vocab: int = 50257,  # typical GPT-2 vocab size
        max_steps: int = 2000
):
    """
    Train a single GPT-2–style model of a given size on WikiText-2.
    """

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # For language modeling, we tokenize on-the-fly (or preprocess once):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Filter out any items that have zero tokens:
    def is_not_empty(example):
        return len(example["input_ids"]) > 0

    for split in tokenized.keys():
        tokenized[split] = tokenized[split].filter(is_not_empty)


    # The parameter count roughly depends on (num_layers, hidden_size, num_heads).
    config = GPT2Config(
        n_embd=d_model,
        n_layer=n_layers,
        n_head=n_heads,
        vocab_size=n_vocab,
        # If you like, reduce the context window to e.g. 256 or 512 if you want smaller models
        n_ctx=512,
        # tweak other settings if desired, e.g. initializer_range, etc.
    )

    model = GPT2LMHeadModel(config)

    # Confirm the total trainable parameters:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created model with ~{total_params:,} trainable parameters.")

    # This pads sequences and creates random “attention mask” for masked language modeling if you want it.
    # For vanilla LM, just set mlm=False
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=f"./{model_name}",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=200,
        save_strategy="no",
        max_steps=max_steps,  # Stop after max_steps
        # Alternatively, use num_train_epochs with dataset length
        # num_train_epochs=1.0,
        report_to="none",  # disable Weights & Biases or other loggers
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Final eval results: {eval_results}")


if __name__ == "__main__":
    train_one_model(
        model_name="custom-gpt2-small",
        n_layers=6,
        d_model=128,
        n_heads=4,
        n_vocab=50257,
        max_steps=2000
    )
