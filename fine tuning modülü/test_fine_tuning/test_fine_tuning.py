import pytest
import torch
import math
from unittest.mock import MagicMock, patch, AsyncMock
from fine_tuning import (
    detect_lora_target_modules,
    preprocess_dataset,
    prepare_fine_tuning,
    compute_perplexity,
    train_model_lora
)


class DummyModule:
    def named_modules(self):
        return [
            ("layer1.q_proj", None),
            ("layer2.v_proj", None),
            ("layer3.other", None)
        ]


def test_detect_lora_target_modules_default():
    modules = detect_lora_target_modules(DummyModule())
    assert modules == ["q_proj", "v_proj"]


def test_detect_lora_target_modules_special_type():
    modules = detect_lora_target_modules(DummyModule(), model_type="special_model_x")
    assert modules == ["special_q_proj", "special_v_proj"]


def test_detect_lora_target_modules_no_match():
    class EmptyModule:
        def named_modules(self): return [("something_else", None)]
    modules = detect_lora_target_modules(EmptyModule())
    assert modules == ["q_proj", "v_proj"]


def test_preprocess_dataset_removes_columns():
    tokenizer = MagicMock()
    tokenizer.model_max_length = 10
    raw_dataset = MagicMock()
    raw_dataset.column_names = ["text", "extra"]
    raw_dataset.map.return_value = "processed"
    result = preprocess_dataset(raw_dataset, tokenizer, text_column="text")
    assert result == "processed"
    raw_dataset.map.assert_called()


@patch("fine_tuning.prepare_model_for_int8_training", side_effect=lambda m: m)
@patch("fine_tuning.AutoModelForCausalLM.from_pretrained")
@patch("fine_tuning.AutoTokenizer.from_pretrained")
@patch("fine_tuning.get_peft_model", side_effect=lambda m, c: m)
def test_prepare_fine_tuning_auto_targets(mock_get_peft, mock_tokenizer, mock_model, mock_int8):
    mock_tokenizer.return_value = MagicMock(pad_token=None, eos_token="<eos>")
    mock_model.return_value = DummyModule()
    model, tokenizer = prepare_fine_tuning(model_name="test-model", seed=123)
    assert model is not None
    assert tokenizer.pad_token == "<eos>"
    mock_get_peft.assert_called()


@patch("fine_tuning.prepare_model_for_int8_training", side_effect=lambda m: m)
@patch("fine_tuning.AutoModelForCausalLM.from_pretrained")
@patch("fine_tuning.AutoTokenizer.from_pretrained")
@patch("fine_tuning.get_peft_model", side_effect=lambda m, c: m)
def test_prepare_fine_tuning_with_targets(mock_get_peft, mock_tokenizer, mock_model, mock_int8):
    mock_tokenizer.return_value = MagicMock(pad_token="PAD", eos_token="<eos>")
    mock_model.return_value = DummyModule()
    model, tokenizer = prepare_fine_tuning(model_name="test-model", seed=123, target_modules=["x_proj"])
    assert tokenizer.pad_token == "PAD"
    mock_get_peft.assert_called()


def test_compute_perplexity_with_tensor():
    logits = torch.randn(2, 3, 5)
    labels = torch.randint(0, 5, (2, 3))
    labels[0, 0] = -100
    result = compute_perplexity((logits, labels))
    assert "perplexity" in result
    assert isinstance(result["perplexity"], float)


def test_compute_perplexity_with_list():
    logits = torch.randn(2, 3, 5).tolist()
    labels = torch.randint(0, 5, (2, 3)).tolist()
    result = compute_perplexity((logits, labels))
    assert "perplexity" in result


@patch("fine_tuning.DataCollatorForLanguageModeling")
@patch("fine_tuning.TrainingArguments")
@patch("fine_tuning.Trainer")
def test_train_model_lora_basic(mock_trainer_cls, mock_args_cls, mock_collator):
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = MagicMock()
    trainer_instance = MagicMock()
    mock_trainer_cls.return_value = trainer_instance
    train_model_lora(model, tokenizer, dataset, eval_dataset=None)
    trainer_instance.train.assert_called()
    trainer_instance.save_model.assert_called()


@patch("fine_tuning.DataCollatorForLanguageModeling")
@patch("fine_tuning.TrainingArguments")
@patch("fine_tuning.Trainer")
def test_train_model_lora_with_eval_and_preprocess(mock_trainer_cls, mock_args_cls, mock_collator):
    model = MagicMock()
    tokenizer = MagicMock()
    dataset = MagicMock()
    eval_dataset = MagicMock()
    trainer_instance = MagicMock()
    mock_trainer_cls.return_value = trainer_instance
    train_model_lora(model, tokenizer, dataset, eval_dataset=eval_dataset, do_preprocess=True)
    trainer_instance.evaluate.assert_called()


@patch("fine_tuning.DataCollatorForLanguageModeling")
@patch("fine_tuning.TrainingArguments")
@patch("fine_tuning.Trainer", side_effect=Exception("init fail"))
def test_train_model_lora_trainer_init_fail(mock_trainer_cls, mock_args_cls, mock_collator):
    with pytest.raises(Exception):
        train_model_lora(MagicMock(), MagicMock(), MagicMock())
