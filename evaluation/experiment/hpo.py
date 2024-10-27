import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union
import torch.nn.functional as F
import pandas as pd
import numpy as np
import bert_layers
import sklearn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import optuna
import json

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Finetuned
model_path = "/home/jeff_lab/fyp/dnabert/finetune/output/dnabert2/checkpoint-3400"
config = AutoConfig.from_pretrained(f"{model_path}/config.json", trust_remote_code=True)
model = bert_layers.BertForSequenceClassification.from_pretrained(f"{model_path}/pytorch_model.bin", config=config, trust_remote_code=True)



# Load the model with the modified configuration


tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    config=f"{model_path}/tokenizer_config.json",
    special_tokens_map=f"{model_path}/special_tokens_map.json"
)

data_files = {"train": "./ft_data/train.csv", "test": "./ft_data/test.csv", "val": "./ft_data/dev.csv"}
dataset = load_dataset('csv', data_files=data_files)
print(dataset)

def encode_dataset(dataset):
    return dataset.map(lambda x: tokenizer(x['sequence'], return_tensors='pt', padding=True, truncation=True).to(device), batched=True)

encoded_train_dataset = encode_dataset(dataset['train'])
encoded_test_dataset = encode_dataset(dataset['test'])
encoded_eval_dataset = encode_dataset(dataset['val'])

model = model.to(device)

def model_init():
    return model

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "roc_auc_score": sklearn.metrics.roc_auc_score(
            valid_labels, valid_predictions, average="macro"
        )
    }

def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)

def save_trials_to_json(study, filename):
    # Create a list to hold trial data
    trials_data = []

    # Iterate through each trial in the study
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state,
            'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        }
        trials_data.append(trial_info)

    with open(filename, 'w') as json_file:
        json.dump(trials_data, json_file, indent=4)
    
result_list = []
def objective(trial):

    training_args = TrainingArguments(
        output_dir='./hpo_result',
        evaluation_strategy="steps",
        eval_steps=30,
        learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64]),
        num_train_epochs=trial.suggest_categorical("num_train_epochs", [10, 15, 20])
    )
    
    # lora_config = LoraConfig(
    #     r=model_args.lora_r,
    #     lora_alpha=model_args.lora_alpha,
    #     target_modules=list(model_args.lora_target_modules.split(",")),
    #     lora_dropout=model_args.lora_dropout,
    #     bias="none",
    #     task_type="SEQ_CLS",
    #     inference_mode=False,
    # )
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    trainer = Trainer(
        model=model,  # Your model here
        args=training_args,
        train_dataset=encoded_train_dataset,  # Your encoded training dataset
        eval_dataset=encoded_test_dataset,  # Your encoded evaluation dataset
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,    
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=encoded_eval_dataset)
    print(training_args.learning_rate)
    
    result_list.append(metrics)
    
    return metrics['eval_roc_auc_score']

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=135) # 15 learning rate tested, *9 fixed 

trial = study.best_trial
trial_index = study.best_trial.number  
trial_metrics = result_list[trial_index]  
print(f"Best trial: {trial_index}")
print(f"  Value: {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("Best trial evaluation metrics:")
for metric_name, metric_value in trial_metrics.items():
    print(f"  {metric_name}: {metric_value}")

save_trials_to_json(study, './hpo_result/optuna_results.json')