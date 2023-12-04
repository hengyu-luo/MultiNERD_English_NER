import argparse
import numpy as np
import evaluate
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_from_disk

seqeval = evaluate.load("seqeval")

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def evaluate_model(tokenized_dataset, model_path, label_list):
    # Load the model
    model = BertForTokenClassification.from_pretrained(model_path)
    data_collator=DataCollatorForTokenClassification(tokenizer)
    print(label_list)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics( p, label_list )
    )

    # Evaluate the model
    evaluation_results = trainer.evaluate(tokenized_dataset["test"])
    print("Evaluation Results:", evaluation_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data_path", required=True, help="Path to the tokenized dataset")
    parser.add_argument("--system", choices=["a", "b"], required=True, help="Choose system to evaluate: 'a' or 'b'")
    args = parser.parse_args()
    # Load the dataset
    tokenized_dataset = load_from_disk(args.data_path)


    if args.system == "a":
        label_list = [
                    "O",       # 0
                    "B-PER",   # 1
                    "I-PER",   # 2
                    "B-ORG",   # 3
                    "I-ORG",   # 4
                    "B-LOC",   # 5
                    "I-LOC",   # 6
                    "B-ANIM",  # 7
                    "I-ANIM",  # 8
                    "B-BIO",   # 9
                    "I-BIO",   # 10
                    "B-CEL",   # 11
                    "I-CEL",   # 12
                    "B-DIS",   # 13
                    "I-DIS",   # 14
                    "B-EVE",   # 15
                    "I-EVE",   # 16
                    "B-FOOD",  # 17
                    "I-FOOD",  # 18
                    "B-INST",  # 19
                    "I-INST",  # 20
                    "B-MEDIA", # 21
                    "I-MEDIA", # 22
                    "B-MYTH",  # 23
                    "I-MYTH",  # 24
                    "B-PLANT", # 25
                    "I-PLANT", # 26
                    "B-TIME",  # 27
                    "I-TIME",  # 28
                    "B-VEHI",  # 29
                    "I-VEHI"   # 30
                ]  
    elif args.system == "b":
        label_list = [
                    "O",       # 0
                    "B-PER",   # 1
                    "I-PER",   # 2
                    "B-ORG",   # 3
                    "I-ORG",   # 4
                    "B-LOC",   # 5
                    "I-LOC",   # 6
                    "B-ANIM",  # 7
                    "I-ANIM",  # 8
                    "B-DIS",   # 9
                    "I-DIS",   # 10
                ]  


    evaluate_model(tokenized_dataset, args.model_path, label_list)
