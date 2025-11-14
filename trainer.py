from transformers import DataCollatorForLanguageModeling,Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from constants import OUTPUT_DIR
from evaluation import compute_metrics


def create_training_arguments() -> TrainingArguments:
    """
    Create and return the training arguments for the model.

    Returns:
        Training arguments for the model.

    NOTE: You can change the training arguments as needed.
    # Below is an example of how to create training arguments. You are free to change this.
    # ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
    """
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=OUTPUT_DIR,
    #     num_train_epochs=1,
    #     per_device_train_batch_size=256,
    #     per_device_eval_batch_size=256,
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     warmup_steps=0,
    #     logging_steps=100,
    #     save_steps=1000,
    #     eval_strategy="steps",
    #     eval_steps=500,
    #     save_total_limit=3,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="bleu",
    #     greater_is_better=True,
    #     max_grad_norm=1.0,
    #     predict_with_generate=True,
    #     bf16=True,
    #     gradient_accumulation_steps=2,
    #     dataloader_num_workers=4,
    # )
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=50,
        eval_strategy="steps",
        save_steps=500,
        bf16=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        prediction_loss_only=True
    )

    return training_args


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.

    NOTE: You are free to change this. But make sure the data collator is the same as the model.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def build_trainer(model, tokenizer, tokenized_datasets) -> Trainer:
    """
    Build and return the trainer object for training and evaluation.

    Args:
        model: Model for sequence-to-sequence tasks.
        tokenizer: Tokenizer object.
        tokenized_datasets: Tokenized datasets.

    Returns:
        Trainer object for training and evaluation.

    NOTE: You are free to change this. But make sure the trainer is the same as the model.
    """
    # data_collator = create_data_collator(tokenizer, model)
    training_args: TrainingArguments = create_training_arguments()
    # Make sure tokenizer has a pad token for batching (common for GPT-like models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    return Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer)
    )


    # return Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_datasets["train"],
    #     eval_dataset=tokenized_datasets["validation"],
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    # )
