import argparse
import numpy as np
import evaluate
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_from_disk

seqeval = evaluate.load("seqeval")

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def compute_metrics(predictions, labels, label_list):
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

def train_system(tokenized_dataset, system_name, num_labels, label_list):
    # Model Initialization Function
    def model_init():
        return BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=num_labels)

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{system_name}",
        evaluation_strategy="steps",
        eval_steps=2000,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb",
        run_name=f"{system_name}_training_run"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, label_list)
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    evaluation_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Evaluation Results for {system_name}:", evaluation_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["a", "b"], required=True, help="Choose system to train: 'a' or 'b'")
    args = parser.parse_args()
    NUM_LABEL_A = 31
    NUM_LABEL_B = 11


    if args.system == "a":
        tokenized_dataset_a = load_from_disk("./tokenized_dataset_a")
        label_list_a = [
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
                ]  # Label list for System A
        train_system(tokenized_dataset_a, "system_a", NUM_LABEL_A, label_list_a)
    elif args.system == "b":
        tokenized_dataset_b = load_from_disk("./tokenized_dataset_b")
        label_list_b = [
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
                ]  # Label list for System B
        train_system(tokenized_dataset_b, "system_b", NUM_LABEL_B, label_list_b)
