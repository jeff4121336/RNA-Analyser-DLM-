"""
  1. Read original.csv return X (2D numpy array encoded), and y (1D numpy array encoded)
  2. Run smoke alogrithms 
  3. out
"""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch
from transformers import AutoTokenizer, AutoModel
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Encode the sequences

data = pd.read_csv("./ft_data/multiclass.csv")

encoded_sequences = [tokenizer(sequence, truncation=True, padding='max_length', max_length=512, return_tensors='pt') for sequence in data['sequence']]
input_ids = torch.cat([encoded_sequence['input_ids'] for encoded_sequence in encoded_sequences], dim=0)

X = input_ids
y = data['labels']
print(f"Before: Counter(y)")


smote_enn = SMOTE(random_state=42, sampling_strategy={0: 160, 1: 160, 2: 160})
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

decoded_x = [tokenizer.decode(x, skip_special_tokens=True).replace(" ", "")[:60] for x in X_resampled]

uo_sample = {str(decoded_x[i]): y_resampled[i] for i in range(len(X_resampled))}
uo_data = pd.DataFrame(list(uo_sample.items()), columns=['sequence', 'labels'])

shuffled_uo_data = uo_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"After: {Counter(y_resampled)}")

shuffled_uo_data.to_csv('./ft_data/smoted_multitrain.csv', index=False)