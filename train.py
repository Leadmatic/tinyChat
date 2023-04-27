from dataclasses import field, dataclass
from typing import Optional, List

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
import torch
import transformers
from transformers import Trainer

from data import TextDataset
from collator import TextCollator


@dataclass
class TrainingConfig(transformers.TrainingArguments):
    """
    This dataclass is used to store the configuration parameters for training.
    It inherits from transformers.TrainingArguments.
    """

    # Optimizer to use for training
    optim: str = field(default="adamw_torch")
    # Maximum sequence length for the model
    model_max_length: int = field(default=512)
    # Name or path of the pre-trained model
    model_name_or_path: str = "google/flan-t5-base"

    # Path to the training data
    data_path: str = "./databricks.json"

    # LoRA arguments
    # parameters based on alpaca-lora
    # Source: https://github.com/tloen/alpaca-lora/blob/main/finetune.py
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v"])
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Directory to cache data
    cache_dir: Optional[str] = field(default=None)


def train():
    """
    This function is used to train the model.
    """
    # Parse arguments into the TrainingConfig dataclass
    training_config = transformers.HfArgumentParser(
        TrainingConfig
    ).parse_args_into_dataclasses()[0]

    # Initialize the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_config.model_name_or_path,
        cache_dir=training_config.cache_dir,
        model_max_length=training_config.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    device_map = "auto"

    # Determine whether to load the model in 8-bit
    load_in_8bit = training_config.model_name_or_path == "google/flan-t5-large"

    # Load the model
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        training_config.model_name_or_path,
        load_in_8bit=load_in_8bit,
        use_cache=False,
        torch_dtype=torch.float16,
        cache_dir=training_config.cache_dir,
        device_map=device_map,
    )

    # Prepare the model for 8-bit training if needed
    if load_in_8bit:
        model = prepare_model_for_int8_training(model)

    # Set up the LoRA configuration
    config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        target_modules=training_config.lora_target_modules,
        lora_dropout=training_config.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    # Get the model with the LoRA configuration
    model = get_peft_model(model, config)

    # Load the dataset and initialize the data collator
    dataset = TextDataset(json_file_path=training_config.data_path)
    collator = TextCollator(text_tokenizer=tokenizer)

    # Initialize the trainer
    trainer = Trainer(
        model,
        args=training_config,
        data_collator=collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained(training_config.output_dir)


if __name__ == "__main__":
    train()
