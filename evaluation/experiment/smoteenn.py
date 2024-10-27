"""
  1. Read original.csv return X (2D numpy array encoded), and y (1D numpy array encoded)
  2. Run smoke alogrithms 
  3. out
"""
import pandas as pd
from imblearn.combine import SMOTEENN
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

data = pd.read_csv("./ft_data/train.csv")

encoded_sequences = [tokenizer(sequence, truncation=True, padding='max_length', max_length=512, return_tensors='pt') for sequence in data['sequence']]
input_ids = torch.cat([encoded_sequence['input_ids'] for encoded_sequence in encoded_sequences], dim=0)

X = input_ids
y = data['labels']
print(Counter(y))


smote_enn = SMOTEENN(random_state=42, sampling_strategy=0.9)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)


decoded_x = [tokenizer.decode(x, skip_special_tokens=True).replace(" ", "")[:60] for x in X_resampled]

uo_sample = {str(decoded_x[i]): y_resampled[i] for i in range(len(X_resampled))}
uo_data = pd.DataFrame(list(uo_sample.items()), columns=['sequence', 'labels'])

shuffled_uo_data = uo_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"sampling_strategy: {Counter(y_resampled)}")

# total_rows = len(shuffled_uo_data)
# train_rows = int(0.8 * total_rows)

# train_data = shuffled_uo_data.iloc[:train_rows]
# test_data = shuffled_uo_data.iloc[train_rows:]

shuffled_uo_data.to_csv('./ft_data/smoted_train.csv', index=False)
# test_data.to_csv('./ft_data/smoted_test.csv', index=False)

# combined_data = pd.concat([train_data, test_data], ignore_index=True)
# combined_data.to_csv('./ft_data/smoted_original.csv', index=False)

# valdata = pd.concat([pd.read_csv("./ft_data/smoted_train.csv"), pd.read_csv("./ft_data/smoted_test.csv")], ignore_index=True) # Combined data from train.csv and dev.csv
# valdata = valdata.sample(frac=0.2, random_state=42).reset_index(drop=True) # take part of train and test data 20% here
# dev_data = pd.concat([valdata, dev_data], ignore_index=True) 

# # Train a machine learning model (Random Forest as an example)
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train, y_train)

# # Make predictions
# y_pred = clf.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# Accuracy: 0.9407407407407408