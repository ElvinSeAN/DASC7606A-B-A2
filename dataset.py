from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from transformers import DataCollatorForSeq2Seq

from constants import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


def build_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Build the dataset.

    Returns:
        The dataset.

    NOTE: You can replace this with your own dataset. Make sure to include
    the `validation` split and ensure that it is the same as the test split from the WMT19 dataset,
    Which means that:
        raw_datasets["validation"] = load_dataset('wmt19', 'zh-en', split="validation")
    """
    dataset = load_dataset("wmt19", "zh-en")
    train_dataset = dataset["train"].select(range(1300000))
    validation_dataset = dataset["train"].select(range(1300000, 1302000))

    # NOTE: You should not change the test dataset
    test_dataset = dataset["validation"]
    return DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset
    })


def create_data_collator(tokenizer, model):
    """
    Create data collator for sequence-to-sequence tasks.

    Args:
        tokenizer: Tokenizer object.
        model: Model object.

    Returns:
        DataCollatorForSeq2Seq instance.
    """
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def preprocess_function(examples, prefix, tokenizer, max_input_length, max_target_length):
    """
    Preprocess the data.

    Args:
        examples: Examples.
        prefix: Prefix.
        tokenizer: Tokenizer object.
        max_input_length: Maximum input length.
        max_target_length: Maximum target length.

    Returns:
        Model inputs.
    """
    # inputs = [prefix + ex["zh"] for ex in examples["translation"]]
    # targets = [ex["en"] for ex in examples["translation"]]
    #
    # model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    #
    # model_inputs["labels"] = labels["input_ids"]
    # return model_inputs

    sources = [prefix + ex["zh"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]

    # e.g. "<zh> -> <en>" format
    texts = [f"{src}\nEnglish: {tgt}" for src, tgt in zip(sources, targets)]

    tokenized = tokenizer(
        texts,
        max_length=max_input_length,  # or some total length
        truncation=True,
    )

    # For basic supervised fine-tuning: labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def preprocess_data(raw_datasets: DatasetDict, tokenizer) -> DatasetDict:
    """
    Preprocess the data.

    Args:
        raw_datasets: Raw datasets.
        tokenizer: Tokenizer object.

    Returns:
        Tokenized datasets.
    """
    # tokenized_datasets: DatasetDict = raw_datasets.map(
    #     function=lambda examples: preprocess_function(
    #         examples=examples,
    #         prefix="",
    #         tokenizer=tokenizer,
    #         max_input_length=MAX_INPUT_LENGTH,
    #         max_target_length=MAX_TARGET_LENGTH,
    #     ),
    #     batched=True,
    # )
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    return tokenized_datasets
