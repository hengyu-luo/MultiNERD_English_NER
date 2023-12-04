# MultiNERD English Named Entity Recognition

## Description
This project involves training and evaluating Named Entity Recognition models based on `bert-base-cased` on HuggingFace with the English subset of the `Babelscape/multinerd` dataset on HuggingFace. It includes two systems: System A and System B.
- System A: Trains on the full set of entity types which contain 15 types in total.
- System B: Trains on a subset of entity types which contain 5 types, (`PERSON`, `ORGANIZATION`, `LOCATION`, `DISEASES`, `ANIMAL`).

## Installation
- Ensure the python version in your environment: Python 3.8.
- Clone this repository.
- Install dependencies: `pip install -r requirements.txt`.

## Usage
1. Run `python data_preparation.py` to prepare the dataset for both the training processes of system A and B.
2. Run `train.py` with a command-line argument to specify which system to train. For example:
    - To train System A: `python train.py --system a`
    - To train System B: `python train.py --system b`

The script determines which system to train, either A or B, based on the provided command-line argument.

## Evaluation
Models are evaluated using precision, recall, F1-score, and accuracy. These metrics are essential in NER tasks to assess the model's ability to correctly identify and classify entities. 

### Understanding NER Evaluation Metrics
- **Accuracy**: Measures the proportion of correct predictions (both entities and non-entities) over all predictions.
- **Precision**: Indicates the proportion of predicted entities that are correct.
- **Recall**: Measures the proportion of actual entities that the model correctly identified.
- **F1 Score**: Provides a balance between precision and recall, offering a single measure of overall model performance.

In NER, precision and recall are particularly crucial because the data often contains significantly more non-entity tokens than entities. The F1 score, as a harmonic mean of precision and recall, serves as a comprehensive measure of a model's effectiveness in recognizing and categorizing entities correctly.

## Note
This project supports WandB logging. If you have a WandB account, you can get logged in for training monitoring. More instructions from WandB will show at the start of the training process.
