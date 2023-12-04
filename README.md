# MultiNERD_English_NER (MultiNERD English Named Entity Recognition)
This project involves training and evaluating Named Entity Recognition models on the English subset of MultiNERD dataset. It includes two systems: System A and System B.

## Description
This project involves training and evaluating Named Entity Recognition models on the MultiNERD dataset. It includes two systems: System A and System B.

## Installation
- Clone this repository.
- Install dependencies: `pip install -r requirements.txt`.

## Usage
1. Run `python data_preparation.py` to prepare the dataset.
2. You can now run `train.py` with a command-line argument to specify which system to train. For example:

    To train System A: `python train.py --system a`
   
    To train System B: `python train.py --system b`
   
The script determines which system to train, either A or B, based on the provided command-line argument.

## Systems
- System A: Trains on the full set of entity types which contain 15 types in toal.
- System B: Trains on a subset of entity types which contain 5 types, (`PERSON`, `ORGANIZATION`, `LOCATION`, `DISEASES`, `ANIMAL`).

## Evaluation
Models are evaluated using precision, recall, F1-score, and accuracy.

## Note
Ensure you have a WandB account and are logged in for training monitoring.
