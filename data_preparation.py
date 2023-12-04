from datasets import load_dataset, load_from_disk
import os
from transformers import BertTokenizerFast

# Load and filter the MultiNERD dataset for English language examples
def load_and_filter_dataset():
    dataset = load_dataset("Babelscape/multinerd")
    english_dataset = dataset.filter(lambda example: example['lang'] == 'en')
    return english_dataset

# Function to tokenize and align labels for the NER task
def tokenize_and_align_labels(tokenizer, examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True)
    tokenized_inputs["labels"] = examples["ner_tags"]
    return tokenized_inputs

# Function to filter labels for System B (specific entity types only)
def filter_labels_system_b(examples):
    # Mapping of required entities to their indices
    required_entities = {
        1, 2,   # B-PER, I-PER
        3, 4,   # B-ORG, I-ORG
        5, 6,   # B-LOC, I-LOC
        13, 14, # B-DIS, I-DIS
        7, 8    # B-ANIM, I-ANIM
    }

    label_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 13: 9, 14: 10}
    # Filter labels, map to 0 if not in required entities
    examples["ner_tags"] = [label_mapping[label] if label in required_entities else 0 for label in examples["ner_tags"]]
    return examples


tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')


# Check if the processed dataset already exists, otherwise process and save
if not os.path.exists("./tokenized_dataset_a"):
    english_dataset = load_and_filter_dataset()

    tokenized_dataset_a = english_dataset.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)
    tokenized_dataset_a.save_to_disk("./tokenized_dataset_a")

if not os.path.exists("./tokenized_dataset_b"):
    filtered_dataset = english_dataset.map(filter_labels_system_b)
    tokenized_dataset_b = filtered_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_dataset_b.save_to_disk("./tokenized_dataset_b")
