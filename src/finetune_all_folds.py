"""
Developed by Francisco J. Lima, María Cuadrado, Fernando Gallego & Gloria Corpas, Lexytrad Research Group, University of Málaga.

This script fine-tunes and evaluates a baseline machine translation model 
(Spanish → English) across multiple folds of a biomedical parallel corpus. 
It uses Hugging Face Transformers and the Datasets library to train and 
evaluate the model using standard metrics (BLEU, chrF, TER).
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
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import evaluate


# ======================
# Global configuration
# ======================
MODEL_NAME: str = "Helsinki-NLP/opus-mt-es-en"
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
    Tokenize input and target sequences for seq2seq training.

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
            "eval_bleu": result_bleu["score"],
            "eval_chrf": result_chrf["score"],
            "eval_ter": result_ter["score"],
        }

    return compute_metrics


def train_and_evaluate_fold(
    fold_dir: str,
    fold_name: str,
    model_name: str = MODEL_NAME,
    num_train_epochs: int = 5
) -> Dict[str, Any]:
    """
    Train and evaluate a translation model on a single fold.

    Args:
        fold_dir: Path to the fold directory containing train/val/test TSVs.
        fold_name: Fold identifier.
        model_name: Hugging Face model identifier.
        num_train_epochs: Number of fine-tuning epochs.

    Returns:
        A dictionary with test evaluation metrics, model name, and fold name.
    """
    print(f"\n=== Training and evaluating {fold_name} ===")

    # Load datasets
    train_df = pd.read_csv(os.path.join(fold_dir, "train.tsv"), sep="\t")
    valid_df = pd.read_csv(os.path.join(fold_dir, "val.tsv"), sep="\t")
    test_df = pd.read_csv(os.path.join(fold_dir, "test.tsv"), sep="\t")

    train_ds = Dataset.from_pandas(train_df[["es", "en"]])
    valid_ds = Dataset.from_pandas(valid_df[["es", "en"]])
    test_ds = Dataset.from_pandas(test_df[["es", "en"]])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    tokenized_datasets = {
        split: dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, src_lang="es", tgt_lang="en"),
            batched=True
        )
        for split, dataset in zip(["train", "validation", "test"], [train_ds, valid_ds, test_ds])
    }

    model: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    args = Seq2SeqTrainingArguments(
        output_dir="./tmp",
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        generation_max_length=256,
        generation_num_beams=4,
        report_to=[],
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    test_metrics["model"] = model_name
    test_metrics["fold"] = fold_name

    return test_metrics


def main() -> None:
    """
    Main entry point for fine-tuning and evaluating the model across folds.
    """
    all_results: List[Dict[str, Any]] = []
    folds = sorted([d for d in os.listdir(KFOLD_DIR) if d.startswith("fold_")])

    for fold in folds:
        fold_dir = os.path.join(KFOLD_DIR, fold)
        try:
            metrics = train_and_evaluate_fold(fold_dir, fold, MODEL_NAME, num_train_epochs=5)
            all_results.append(metrics)

            # Save fold results
            df_fold = pd.DataFrame([metrics])
            df_fold.to_csv(
                os.path.join(OUTPUT_DIR, f"results_finetuned_{fold}.tsv"),
                sep="\t",
                index=False
            )
        except Exception as e:
            print(f"Error processing {fold}: {e}", file=sys.stderr)

    # Save global results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(
        os.path.join(OUTPUT_DIR, "results_finetuned_all.tsv"),
        sep="\t",
        index=False
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        sys.exit(1)
