"""
Developed by Francisco J. Lima, María Cuadrado, Fernando Gallego & Gloria Corpas, Lexytrad Research Group, University of Málaga.

This script evaluates baseline machine translation models (Spanish → English) 
using multiple folds of a parallel biomedical corpus. 
It leverages Hugging Face Transformers and the Datasets library to tokenize, 
train, and evaluate models across standard metrics (BLEU, chrF, TER).
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Callable

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import evaluate


# ======================
# Global configuration
# ======================
BASELINE_MODELS: List[str] = [
    "Helsinki-NLP/opus-mt-es-en",
    "facebook/m2m100_418M",
    "facebook/m2m100_1.2B",
    "Helsinki-NLP/opus-mt-tc-big-es-en",
    "facebook/nllb-200-distilled-600M",
]

KFOLD_DIR: str = "../data/kfold"
OUTPUT_DIR: str = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metrics
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")


def preprocess_function(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    src_lang: str = "es",
    tgt_lang: str = "en",
    max_length: int = 256
) -> Dict[str, Any]:
    """
    Tokenize input and target sequences for seq2seq training/evaluation.

    Args:
        examples: A dictionary containing source and target text pairs.
        tokenizer: Hugging Face tokenizer.
        src_lang: Source language key in dataset.
        tgt_lang: Target language key in dataset.
        max_length: Maximum sequence length.

    Returns:
        A dictionary with tokenized inputs and labels.
    """
    src_texts = examples[src_lang]
    tgt_texts = examples[tgt_lang]

    model_inputs = tokenizer(src_texts, max_length=max_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable:
    """
    Build a metrics computation function for seq2seq evaluation.

    Args:
        tokenizer: Hugging Face tokenizer.

    Returns:
        A function that computes BLEU, chrF, and TER metrics.
    """
    def compute_metrics(eval_preds: Any) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        result_bleu = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        result_chrf = chrf.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        result_ter = ter.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": result_bleu["score"],
            "chrf": result_chrf["score"],
            "ter": result_ter["score"],
        }

    return compute_metrics


def evaluate_model(
    model_name: str,
    raw_datasets: Dict[str, Dataset],
    batch_size: int = 16,
    max_length: int = 256,
    num_beams: int = 4
) -> Dict[str, Any]:
    """
    Evaluate a given model on the provided datasets.

    Args:
        model_name: Hugging Face model identifier.
        raw_datasets: Dictionary of datasets with "train", "validation", and "test" splits.
        batch_size: Evaluation batch size.
        max_length: Maximum sequence length.
        num_beams: Beam search width for generation.

    Returns:
        A dictionary with evaluation metrics and model name.
    """
    print(f"\nEvaluating model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Handle language-specific settings
    if "m2m100" in model_name.lower():
        tokenizer.src_lang = "es"
        tokenizer.tgt_lang = "en"
        model.config.forced_bos_token_id = tokenizer.get_lang_id("en")
    elif "nllb" in model_name.lower():
        tokenizer.src_lang = "spa_Latn"
        tokenizer.tgt_lang = "eng_Latn"
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    # Tokenize datasets
    tokenized_datasets = {
        split: raw_datasets[split].map(
            lambda examples: preprocess_function(examples, tokenizer, max_length=max_length),
            batched=True
        )
        for split in raw_datasets
    }

    args = Seq2SeqTrainingArguments(
        output_dir=f"./results-{model_name.split('/')[-1]}",
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=max_length,
        generation_num_beams=num_beams,
        report_to=[],
        logging_strategy="no",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=build_compute_metrics(tokenizer),
    )

    metrics = trainer.evaluate()
    metrics["model"] = model_name
    return metrics


def main() -> None:
    """
    Main entry point for evaluating baseline translation models across folds.
    """
    all_results: List[Dict[str, Any]] = []

    folds = sorted([d for d in os.listdir(KFOLD_DIR) if d.startswith("fold_")])

    for fold in folds:
        print(f"\n=== Evaluating {fold} ===")
        fold_dir = os.path.join(KFOLD_DIR, fold)

        # Load datasets
        train_df = pd.read_csv(os.path.join(fold_dir, "train.tsv"), sep="\t")
        valid_df = pd.read_csv(os.path.join(fold_dir, "val.tsv"), sep="\t")
        test_df = pd.read_csv(os.path.join(fold_dir, "test.tsv"), sep="\t")

        raw_datasets = {
            "train": Dataset.from_pandas(train_df[["es", "en"]]),
            "validation": Dataset.from_pandas(valid_df[["es", "en"]]),
            "test": Dataset.from_pandas(test_df[["es", "en"]]),
        }

        fold_results: List[Dict[str, Any]] = []
        for model_name in BASELINE_MODELS:
            try:
                metrics = evaluate_model(model_name, raw_datasets)
                metrics["fold"] = fold
                fold_results.append(metrics)
            except Exception as e:
                print(f"Error with {model_name} in {fold}: {e}")

        # Save fold results
        df_fold = pd.DataFrame(fold_results)
        df_fold.to_csv(os.path.join(OUTPUT_DIR, f"results_{fold}.tsv"), sep="\t", index=False)
        all_results.extend(fold_results)

    # Save global results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(OUTPUT_DIR, "results_all_folds.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        sys.exit(1)
