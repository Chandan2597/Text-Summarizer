from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from TextSummarizer.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # ✅ Limit CPU usage (prevents freezing)
        torch.set_num_threads(2)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ USE LIGHT MODEL (CRITICAL)
        model_ckpt = self.config.model_ckpt

        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

        # ✅ Reduce memory usage
        model.gradient_checkpointing_enable()

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # ✅ LIGHT TRAINING CONFIG
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=100,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=500,
            gradient_accumulation_steps=1,  # IMPORTANT
            fp16=False
        )

        trainer = Trainer(
            model=model,
            args=trainer_args,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        # Save model
        model.save_pretrained(os.path.join(self.config.root_dir, "model"))

        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))