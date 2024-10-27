import fm  # for development with RNA-FM
from pathlib import Path
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from sklearn.manifold import TSNE  # for dimension reduction
from sklearn.model_selection import train_test_split  # for splitting train/val/test
from tqdm.notebook import tqdm  # for showing progress
import matplotlib.pyplot as plt
from fasta_data import group_data
from sklearn.decomposition import PCA
from dslayer import RNATypeClassifier, RNATypeDataset
import sklearn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score
# max_length = 0
# for i in tqdm(range(0, len(seqs), chunk_size)):
#     data = seqs[i:i + chunk_size]
#     batch_labels, batch_strs, batch_tokens = batch_converter(data)
#     max_length = max(max_length, batch_tokens.shape[1])
# print(max_length)
#   File "rnafm.py", line 39, in <module>
#     token_embeddings = np.zeros((len(labels), max_length, 640))
# numpy.core._exceptions.MemoryError: Unable to allocate 4.96 TiB for an array with shape (12953, 82172, 640) and data type float64

# Pre-allocate the space to save memory
batch_size = 8
lr = 1e-3
epochs = 161

rfam_list = ["5S_rRNA", "5_8S_rRNA", "tRNA", "ribozyme", "CD-box", "miRNA",
             "Intron_gpI", "Intron_gpII", "scaRNA", "HACA-box", "riboswitch", "IRES", "leader"]

def calculate_binary_metrics(y_true, y_pred, target_class):
    # Convert labels to binary: 1 for the target class, 0 for all others
    binary_y_true = (y_true == target_class).astype(int)
    binary_y_pred = (y_pred == target_class).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(binary_y_true, binary_y_pred)
    
    # Extract metrics from confusion matrix
    TP = cm[1, 1]  # True Positives
    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]  # False Positives
    FN = cm[1, 0]  # False Negatives
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(binary_y_true, binary_y_pred)
    recall = recall_score(binary_y_true, binary_y_pred)
    f1 = f1_score(binary_y_true, binary_y_pred)
    mcc = matthews_corrcoef(binary_y_true, binary_y_pred)
    acc = accuracy_score(binary_y_true, binary_y_pred)
    return {
        'Confusion Matrix': cm,
        'True Positives': TP,
        'True Negatives': TN,
        'False Positives': FP,
        'False Negatives': FN,
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    }

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
    }

def construct_xy(dataset):
    chunk_size = 20
    token_embeddings = np.zeros((len(dataset), 1024, 640))
    y_labels = np.zeros(len(dataset), dtype=int)
    input_strs = np.empty(len(dataset), dtype=object)
    seqs = dataset
    
    # print("Initial shape of token_embeddings:", token_embeddings.shape)
    # Divide all the sequences into chunks for processing due to GPU memory limit
    for i in tqdm(range(0, len(dataset), chunk_size)):
        data = seqs[i:i + chunk_size]
        data_tuples = [items[0] for items in data]
        data_labels = [items[1] for items in data]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(data_tuples)
        if batch_tokens.shape[1] > 1024: # print omit data for clarity
            # print(f"{data} omitted")
            continue
        # Use GPU
        with torch.no_grad():
            results = model(batch_tokens.to(device), repr_layers=[12])
        # print(batch_tokens.shape)

        emb = results['representations'][12].cpu().numpy()  # Convert embeddings to NumPy
        # print(f"emb: {emb.shape}")
        # Check the shape of batch_tokens
        if batch_tokens.shape[1] > 640:
            # Apply PCA to reduce to 640 dimensions
            reshaped_emb = emb.reshape(-1, 640) # (chucksize * oversize embed, 640)
            pca = PCA(n_components=640)
            reshaped_emb = pca.fit_transform(reshaped_emb)  # Fit and transform the embeddings      
            reshaped_back = reshaped_emb.reshape(chunk_size, batch_tokens.shape[1], 640)  

            emb = reshaped_back[:, :640, :] 


        token_embeddings[i:i+chunk_size, :emb.shape[1], :] = emb
        input_strs[i:i+chunk_size] = batch_strs
        y_labels[i:i+chunk_size] = data_labels
    
    print(f"Done: {y_labels.shape}")
    return token_embeddings, y_labels, input_strs

def train(dataset):
    x_train, y_train, train_seqs = construct_xy(dataset[0])
    x_val, y_val, val_seqs = construct_xy(dataset[2])

    train_dataset = RNATypeDataset(x_train, y_train)
    val_dataset = RNATypeDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    model = RNATypeClassifier(len(rfam_list)).to(device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    max_val_acc = -1
    best_epoch = -1

    train_loss_history = []
    val_loss_history = []

    train_acc_history = []
    val_acc_history = []

    for epoch in tqdm(range(epochs)):

        # train the model
        train_losses = []
        train_preds = []
        train_targets = []

        model.train()

        for batch in train_loader:
            x, y = batch
            x, y = x.to(device).float(), y.to(device).long()

            # no need to apply the softmax function since it has been included in the loss function
            y_pred = model(x)

            # y_pred: (B, C) with class probabilities, y shape: (B,) with class indices
            loss = criterion(y_pred, y)

            train_losses.append(loss.item())
            train_preds.append(torch.max(y_pred.detach(),1)[1])
            train_targets.append(y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate the model
        val_losses = []
        val_preds = []
        val_targets = []

        model.eval()

        for batch in val_loader:
            x, y = batch
            x, y = x.to(device).float(), y.to(device).long()

            y_pred = model(x)

            # y_pred: (B, C) with class probabilities, y shape: (B,) with class indices
            loss = criterion(y_pred, y)

            val_losses.append(loss.item())
            val_preds.append(torch.max(y_pred.detach(),1)[1])
            val_targets.append(y)

        # calculate the accuracy
        train_preds = torch.cat(train_preds, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_acc = (train_preds == train_targets).float().mean().cpu()

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        val_acc = (val_preds == val_targets).float().mean().cpu()

        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # save the model checkpoint for the best validation accuracy
        if val_acc > max_val_acc:
            torch.save({'model_state_dict': model.state_dict()}, 'rna_type_checkpoint.pt')
            best_epoch = epoch
            max_val_acc = val_acc

        # show intermediate steps
        if epoch % 20 == 1:
            tqdm.write(f'epoch {epoch}/{epochs}: train loss={np.mean(train_loss_history):.6f}, '
                    f'train acc={train_acc:.6f}, '
                    f'val loss={np.mean(val_loss_history):.6f}, '
                    f'val acc={val_acc:.6f}')

        train_loss_history.append(np.mean(train_losses))
        val_loss_history.append(np.mean(val_losses))

    return vizuation(train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_epoch)

def test(dataset, view=0, output_csv='test_results.csv'):
    x_test, y_test, test_seqs = construct_xy(dataset[1])
    test_dataset = RNATypeDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(test_seqs)
    test_preds = []
    model = RNATypeClassifier(len(rfam_list)).to(device)
    model.load_state_dict(torch.load('rna_type_checkpoint.pt')['model_state_dict'])
    model.eval()

    all_data = []  # List to store data for CSV

    for batch in test_loader:
        x, y = batch
        x, y = x.to(device).float(), y.to(device).long()

        output = model(x)

        _, y_pred = torch.max(output.data, 1)  # Get predicted class indices
        test_preds.append(y_pred.cpu().numpy())

        # Store each sample's data, prediction, and truth for CSV
        all_data.extend(zip(test_seqs, y.cpu().numpy(), y_pred.cpu().numpy()))

    test_preds = np.concatenate(test_preds)
    
    df = pd.DataFrame(all_data, columns=['Input Data', 'True Label', 'Predicted Label'])
    df.to_csv(output_csv, index=False)

    if view == 0: # overall performance
        result = calculate_metric_with_sklearn(test_preds, y_test)
        print(result)
    else: # performance on seperate classes
        results_per_class = {}
        for i in range(len(rfam_list)):
            # filtered_indices = (y_test == i)
            # filtered_labels = y_test[filtered_indices]
            # filtered_predictions = test_preds[filtered_indices]
            # Calculate True Positives, True Negatives, False Positives, False Negatives
            # result = calculate_metric_with_sklearn(filtered_predictions, filtered_labels)
            result = calculate_binary_metrics(y_test, test_preds, target_class=i)
            results_per_class[i] = result
            # use zip for the csv construction here
       
        df = pd.DataFrame.from_dict(results_per_class, orient='index')
        df['Class Label'] = ["5S_rRNA", "5_8S_rRNA", "tRNA", "ribozyme", "CD-box", "miRNA", "Intron_gpI", "Intron_gpII", "scaRNA", "HACA-box", "riboswitch", "IRES", "leader"]
        df = df[['Class Label', 'Confusion Matrix', 'True Positives', 'True Negatives', 
         'False Positives', 'False Negatives', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']]
        numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC']

        df[numeric_columns] = df[numeric_columns].round(3)

        df.to_csv("separate_class_result.csv", index=False)
        
        print(results_per_class)

    return

def vizuation(train_loss_history, val_loss_history, train_acc_history, val_acc_history, best_epoch):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='train loss')
    plt.plot(val_loss_history, label='val loss')

    # the epoch with best validation loss
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()

    plt.savefig("loss_history.png")

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_history, label='train accuracy')
    plt.plot(val_acc_history, label='val accuracy')

    # the epoch with best validation accuracy
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy History')
    plt.legend()
    
    plt.savefig("accuracy.png")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device} device')
    data_dir = './Models_RNA/'

    model, alphabet = fm.pretrained.rna_fm_t12(Path(data_dir, 'RNA-FM_pretrained.pth'))
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval() # disables dropout for deterministic results
    print(model)

    dataset = group_data() # Assuming get_data returns seqs, labels, rfam_list
    # train(dataset)
    # test(dataset, 0) #overall evaluation
    test(dataset, 1) #evaluation on classes


# token_embeddings = np.mean(token_embeddings, axis=1)
# print(token_embeddings.shape)


#T-SNE
# pca_t = PCA(n_components=2)
# token_embeddings = pca_t.fit_transform(token_embeddings)
# tsne = TSNE(n_components=2, random_state=42)  # n_components is the dimension of the reduced data
# embeddings = tsne.fit_transform(token_embeddings)
# print(embeddings.shape)

# rfam_list = ["5S_rRNA", "5_8S_rRNA", "tRNA", "ribozyme", "CD-box", "miRNA",
#              "Intron_gpI", "Intron_gpII", "HACA-box", "riboswitch", "IRES", "leader",
#              "scaRNA", "mRNA"]

# plt.figure(figsize=(10, 10))

# print(sorted(list(set(labels))))

# for i, label in enumerate(sorted(list(set(labels)))):
#     # find the data points corresponding to the current label
#     print(i)
#     indices = [j for j, l in enumerate(labels) if l == label]
#     plt.scatter(embeddings[indices, 0], embeddings[indices, 1], s=5, alpha=0.5, label=rfam_list[i])

# plt.legend()
# plt.xticks([])
# plt.yticks([])

# plt.savefig("./test_ncrtrain.png")