import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv('./m1_13c.csv')
df2 = pd.read_csv('./bl1.csv')
df3 = pd.read_csv('./bl2.csv')

# Confirm the structure of the DataFrames
print(df1.head())
print(df2.head())
print(df3.head())

# Set the Class Label as the index for easy plotting
df1.set_index('Class Label', inplace=True)
df2.set_index('Class Label', inplace=True)
df3.set_index('Class Label', inplace=True)

# Ensure Class Labels are treated as categorical and maintain order
class_labels_order = df1.index.tolist()  # Assuming df1 has the order we want

# Plotting
plt.figure(figsize=(12, 12))

# Accuracy Plot
plt.subplot(3, 1, 1)
plt.plot(class_labels_order, df1.loc[class_labels_order, 'Accuracy'], color='red', marker='o', label='Our Model Accuracy', linewidth=2)
plt.plot(class_labels_order, df2.loc[class_labels_order, 'Accuracy'],marker='s', label='State-of-the-art Accuracy', linewidth=2)
plt.plot(class_labels_order, df3.loc[class_labels_order, 'Accuracy'], marker='^', label='Baseline Accuracy', linewidth=2)
plt.title('Accuracy Comparison')
plt.xlabel('Class Label')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.legend()
plt.grid()

# F1-Score Plot
plt.subplot(3, 1, 2)
plt.plot(class_labels_order, df1.loc[class_labels_order, 'F1-score'], color='red', marker='o', label='Our Model F1-Score', linewidth=2)
plt.plot(class_labels_order, df2.loc[class_labels_order, 'F1-score'],marker='s', label='State-of-the-art F1-Score', linewidth=2)
plt.plot(class_labels_order, df3.loc[class_labels_order, 'F1-score'], marker='^', label='Baseline F1-Score', linewidth=2)
plt.title('F1-Score Comparison')
plt.xlabel('Class Label')
plt.ylabel('F1-Score')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# MCC Plot
plt.subplot(3, 1, 3)
plt.plot(class_labels_order, df1.loc[class_labels_order, 'MCC'], color='red', marker='o', label='Our Model MCC', linewidth=2)
plt.plot(class_labels_order, df2.loc[class_labels_order, 'MCC'],marker='s', label='State-of-the-art MCC', linewidth=2)
plt.plot(class_labels_order, df3.loc[class_labels_order, 'MCC'], marker='^', label='Baseline MCC', linewidth=2)
plt.title('MCC Comparison')
plt.xlabel('Class Label')
plt.ylabel('MCC')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('./m1_comparison.png')