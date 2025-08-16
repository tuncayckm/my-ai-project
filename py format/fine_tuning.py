import re
import math
import torch.nn.functional as F
from typing import List, Optional
from config import app_logger as logger
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


def detect_lora_target_modules(model, model_type: Optional[str] = None) -> List[str]:
    """
    Model mimarisi ve versiyonuna göre dinamik LoRA target modüllerini tespit eder.
    Regex ile tam eşleşme yapılır.
    """
    if model_type == "special_model_x":
        return ["special_q_proj", "special_v_proj"]

    candidate_modules = ["q_proj", "v_proj", "query_key_value", "k_proj", "o_proj"]
    found_modules = set()

    for name, _ in model.named_modules():
        for candidate in candidate_modules:
            pattern = rf"\b{re.escape(candidate)}\b"
            if re.search(pattern, name):
                found_modules.add(candidate)

    return sorted(found_modules) if found_modules else ["q_proj", "v_proj"]

def preprocess_dataset(raw_dataset, tokenizer, text_column="text"):
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
    return raw_dataset.map(tokenize_function, batched=True, remove_columns=[col for col in raw_dataset.column_names if col != text_column])


def prepare_fine_tuning(
    model_name: str = "mosaicml/phi-3-mini",
    seed: int = 42,
    target_modules: Optional[List[str]] = None,
    model_type: Optional[str] = None
):
    logging.info("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.model_max_length = getattr(tokenizer, "model_max_length", 2048)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_in_8bit = device == "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=load_in_8bit,
        device_map="auto" if device == "cuda" else None
    )

    if load_in_8bit:
        model = prepare_model_for_int8_training(model)
    else:
        logging.info("8-bit load unavailable, running in full precision.")

    # LoRA target modüllerini dışarıdan alma, yoksa dinamik tespit (model_type parametreli)
    if target_modules is None:
        target_modules = detect_lora_target_modules(model, model_type=model_type)
    logging.info(f"Using LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    logging.info("LoRA fine-tuning ready.")
    return model, tokenizer

def compute_perplexity(eval_pred):
    logits, labels = eval_pred
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # Compute cross entropy loss ignoring -100 labels
    logits = logits if isinstance(logits, torch.Tensor) else torch.tensor(logits)
    labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())
    # Mask padding tokens
    mask = shift_labels != -100
    loss = loss * mask
    # Average loss per token
    avg_loss = loss.sum() / mask.sum()
    perplexity = math.exp(avg_loss.item())
    return {"perplexity": perplexity}

def train_model_lora(
    model,
    tokenizer,
    dataset,
    output_dir: str = "./lora_finetuned_model",
    per_device_train_batch_size: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-4,
    seed: int = 42,
    eval_dataset = None,
    do_preprocess: bool = False,
    text_column: str = "text"     
):
    set_seed(seed)

    if do_preprocess:
        dataset = preprocess_dataset(dataset, tokenizer, text_column=text_column)
        if eval_dataset is not None:
            eval_dataset = preprocess_dataset(eval_dataset, tokenizer, text_column=text_column)

    # DataCollatorForLanguageModeling, causal LM için mlm=False
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # fp16 kontrolü: CUDA varsa ve destekliyorsa
    fp16 = False
    if torch.cuda.is_available():
        try:
            fp16 = torch.cuda.get_device_capability()[0] >= 7  # SM >= 7.0 genelde destekler
        except Exception:
            fp16 = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
        learning_rate=learning_rate,
        fp16=fp16,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=500 if eval_dataset is not None else None,
        save_strategy="steps",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        report_to="none"
    )

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    # Hata yönetimi: dataset uyumsuzluğu ve tokenizer kontrolü için
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_perplexity if eval_dataset is not None else None
        )
    except Exception as e:
        logging.error(f"Trainer initialization failed: {e}")
        raise e

    logging.info("Starting training...")
    try:
        trainer.train()
        if eval_dataset is not None:
            eval_results = trainer.evaluate()
            logging.info(f"Evaluation results: {eval_results}")
        trainer.save_model(output_dir)
        logging.info(f"Model saved to: {output_dir}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise e
