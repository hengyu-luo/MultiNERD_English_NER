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
### 1. Prepare the Dataset
Run `python data_preparation.py` to prepare the dataset for both the training processes of system A and B.
### 2. Train and Evaluate the Model
Run `train.py` with a command-line argument to specify which system to train. For example:

- To train System A: `python train.py --system a`

- To train System B: `python train.py --system b`
    
The script determines which system to train, either A or B, based on the provided command-line argument.
### 3. Want to Do Direct Evaluation with Existing Model (Checkpoints)?
Run `eval.py` with required arguments:

- Model Path: Path to the trained model checkpoint in Google Drive.
- Data Path: Path to the tokenized dataset.
- System: Specify the system ('a' or 'b').

Example command:

`python eval.py --model_path /content/results_system_b/checkpoint-xxx --data_path ./tokenized_dataset_b --system b`

## Evaluation
Models are evaluated using precision, recall, F1-score, and accuracy. These metrics are essential in NER tasks to assess the model's ability to correctly identify and classify entities. 

- **Accuracy**: Measures the proportion of correct predictions (both entities and non-entities) over all predictions.
- **Precision**: Indicates the proportion of predicted entities that are correct.
- **Recall**: Measures the proportion of actual entities that the model correctly identified.
- **F1 Score**: Provides a balance between precision and recall, offering a single measure of overall model performance.

In NER, precision and recall are particularly crucial because the data often contains significantly more non-entity tokens than entities. The F1 score, as a harmonic mean of precision and recall, serves as a comprehensive measure of a model's effectiveness in recognizing and categorizing entities correctly.


## Training Hyperparameters
The models are trained with the following hyperparameters:

| Hyperparameter                    | Value          |
| --------------------------------- | -------------- |
| Evaluation Strategy               | steps          |
| Evaluation Steps                  | 2000           |
| Learning Rate                     | 2e-5           |
| Batch Size per Device (Training)  | 16             |
| Batch Size per Device (Evaluation)| 16             |
| Number of Training Epochs         | 2              |
| Weight Decay                      | 0.01           |
| Save Strategy                     | epoch          |
| Save Total Limit                  | 2              |
| Reporting to WandB                | Enabled        |
| Run Name                          | system_a/system_b (dynamically set) |


## Note
This project supports WandB logging. If you have a WandB account, you can get logged in for training monitoring. More instructions from WandB will show at the start of the training process.
