import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
import pandas as pd
import bert_layers
import pandas as pd
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


dna = pd.read_csv("./ft_data/abnormal.csv")
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

df.to_csv('ft1_epoch20_abnormal.csv', index=False)

total_samples = len(df)
correct_predictions = (df['result'] == 'O').sum()
accuracy = correct_predictions / total_samples
print(f"Model Accuracy: {accuracy * 100:.2f}%")



# # # Define the filenames of the CSV files
# # file1 = '/home/jeff_lab/fyp/dnabert/ft_data/dev.csv'
# # file2 = '/home/jeff_lab/fyp/dnabert/ft_data/test.csv'
# # file3 = '/home/jeff_lab/fyp/dnabert/ft_data/train.csv'

# # # Function to read CSV file and count the labels
# # def count_labels(filename):
# #     # Read the CSV file into a DataFrame
# #     df = pd.read_csv(filename)

# #     # Count the occurrences of each label
# #     label_counts = df['labels'].value_counts()

# #     return label_counts

# # # Count labels for each file
# # label_counts_file1 = count_labels(file1)
# # label_counts_file2 = count_labels(file2)
# # label_counts_file3 = count_labels(file3)

# # # Display the total number of labels for each file
# # print("Total number of labels in", file1, ":\n", label_counts_file1)
# # print("\nTotal number of labels in", file2, ":\n", label_counts_file2)
# # print("\nTotal number of labels in", file3, ":\n", label_counts_file3)

# import json
# import pandas as pd
# # JSON data
# json_data =  [
# {"loss": 0.5249305367469788, "accuracy": 0.7730769230769231, "f1": 0.7486852667966382, "mcc": 0.5638726942181075, "precision": 0.823076923076923, "recall": 0.7460346070656092}
# ,{"loss": 0.6594261527061462, "accuracy": 0.6298076923076923, "f1": 0.4205289626976374, "mcc": 0.14236952426678023, "precision": 0.78125, "recall": 0.518016961279955}
# ,{"loss": 0.49608859419822693, "accuracy": 0.7884615384615384, "f1": 0.7589305281612974, "mcc": 0.5774384463091252, "precision": 0.8334147930922124, "recall": 0.7500152709058701}
# ,{"loss": 0.654260516166687, "accuracy": 0.6125, "f1": 0.38217062833978255, "mcc": 0.010136311386946763, "precision": 0.5563583815028902, "recall": 0.500455765432724}
# ,{"loss": 0.48760750889778137, "accuracy": 0.8192307692307692, "f1": 0.7885776570528902, "mcc": 0.6426711654424228, "precision": 0.8775179242062137, "recall": 0.7735143157504174}
# ]
    
# # Function to round numeric values to 3 decimal places
# def round_to_4dp(data):
#     for item in data:
#         for key, value in item.items():
#             if isinstance(value, (int, float)):
#                 item[key] = round(value, 4)
#     return data

# # Round the data to 3 decimal places
# rounded_data = round_to_4dp(json_data)

# df = pd.DataFrame(rounded_data)

# # Write the DataFrame to an Excel file (XLSX format)
# df.to_excel('output_rounded.xlsx', index=False)

# print("Excel file 'output_rounded.xlsx' created successfully.")
