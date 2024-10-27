import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
import pandas as pd
import bert_layers
from collections import Counter

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Finetuned
model_path = "/home/jeff_lab/fyp/dnabert/finetune_ft1/output/dnabert2_20"
config = AutoConfig.from_pretrained(f"{model_path}/config.json", trust_remote_code=True)
model = bert_layers.BertForSequenceClassification.from_pretrained(f"{model_path}/pytorch_model.bin", config=config, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    config=f"{model_path}/tokenizer_config.json",
    special_tokens_map=f"{model_path}/special_tokens_map.json"
)

# Pretrained
# tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
# model = bert_layers.BertForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)


dna = pd.read_csv("./ft_data/original.csv")
sequences = dna['sequence'].tolist()
correct_labels = dna['labels'].tolist()
tokenized_input = tokenizer(sequences, return_tensors='pt', padding=True, truncation=True).to(device)
input_ids = tokenized_input['input_ids'].to(device)
attention_mask = tokenized_input['attention_mask'].to(device)


model.to(device)
# model = bert_layers.BertForSequenceClassification(model)

hidden_states = model(input_ids)[0] # [1, sequence_length, 768]
# print(model)

with torch.no_grad():
  outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# print(model.can_generate())
# predicted_token_ids = outputs[0].argmax(dim=-1) # Get the predicted token IDs
# print(predicted_token_ids)
# Decode the predicted token IDs back to text
# decoded_output = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=False)
logits_list = outputs['logits']
probabilities_list = []
predicted_classes = []
# Process each tensor in the logits list
for logits in logits_list:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # Check if logits is 2D -> Make it 2D (1, 2) if it was (2,)

    probabilities = F.softmax(logits, dim=1)  # Calculate probabilities
    # print(probabilities)  # Print the probabilities
    predicted_class = torch.argmax(probabilities, dim=1).item()  # Get the predicted class
    # print(predicted_class)  # Print the predicted class
    probabilities_list.append(probabilities)
    predicted_classes.append(predicted_class)

# print("All probabilities:", probabilities_list)
# print("All predicted classes:", predicted_classes)

df = pd.DataFrame({'sequence': sequences, 'predict': predicted_classes, 'truth': correct_labels})
df['result'] = 'O'  # Initialize all values as 'O'
df.loc[df['predict'] != df['truth'], 'result'] = 'X'  

df.to_csv('ft1_run2.csv', index=False)

total_samples = len(df)
correct_predictions = (df['result'] == 'O').sum()
accuracy = correct_predictions / total_samples
print(f"Model Accuracy: {accuracy * 100:.2f}%")