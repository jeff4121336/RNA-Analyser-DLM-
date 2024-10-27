import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
import re 
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Data
# t = pd.concat([pd.read_csv("./ft_data/train.csv"), pd.read_csv("./ft_data/test.csv")], ignore_index=True) # Combined data from train.csv and dev.csv
# yt = t['labels']
# print(Counter(yt)[0]+Counter(yt)[1])

# s = pd.concat([pd.read_csv("./ft_data/smoted_train.csv"), pd.read_csv("./ft_data/smoted_test.csv")], ignore_index=True) # Combined data from train.csv and dev.csv
# ys = s['labels']
# print(Counter(ys))

# xcoor = [2, 3, 5, 6, 8, 9]
# data = [Counter(yt)[0]+Counter(yt)[1],Counter(ys)[0]+Counter(ys)[1] ,Counter(yt)[0],Counter(ys)[0],Counter(yt)[1],Counter(ys)[1]]
# label = ['Total 0', 'Total 1', 'Intron 0', 'Intron 1', 'Exon 0', 'Exon 1']

# plt.bar(xcoor, data, tick_label = label, width=0.5, color=['blue', 'green'])

# for i, v in enumerate(data):
#     plt.text(xcoor[i], v + 10, str(v), color='black', ha='center')

#Smoke result
def smoke():
	x = [0.6, 0.7, 0.8, 0.9, 1.0]
	y1 = [0.6539077877998352, 0.6074507107059464, 0.44847185011006063, 0.0005155064085839653, 0.42015249631561125, 0.5004253336481225]
	y2 = [0.65798220038414, 0.6120456905503634, 0.4567801948699629, 0.019071119595870534, 0.42643388188722575, 0.5094928464492116]
	y3 = [0.6740369021892547, 0.5708947885939036, 0.45506624679056423, 0.04530273833330341, 0.41811558908059726, 0.5227729983871611]
	y4 = [0.669369912147522, 0.5863776460351151, 0.5009546049403868, 0.12224109505535974, 0.5979050807255353, 0.5435103199138546]
	y5 = [0.6750236392021179, 0.5571347521345738, 0.49300315835069347, 0.07968672520204188, 0.5349325152417649, 0.5315057263811462]

	z0 = [y1[0], y2[0], y3[0], y4[0], y5[0]]  
	z1 = [y1[1], y2[1], y3[1], y4[1], y5[1]] 
	z2 = [y1[2], y2[2], y3[2], y4[2], y5[2]]  
	z3 = [y1[3], y2[3], y3[3], y4[3], y5[3]] 
	z4 = [y1[4], y2[4], y3[4], y4[4], y5[4]]  
	z5 = [y1[5], y2[5], y3[5], y4[5], y5[5]] 

	# plotting the points 
	plt.figure(1)
	plt.plot(x, z0, label="loss")
	plt.plot(x, z1, label="accuracy")

	# evaluation metrics
	plt.legend()
	plt.ylabel("Loss and accuracy")
	plt.xlabel("sampling strategy value")
	plt.title("sampling strategy affection on evaluation metrics (1)")
	plt.savefig('png1.png')

	plt.figure(2)
	plt.plot(x, z2, label="f1")
	plt.plot(x, z4, label="precision")
	plt.plot(x, z5, label="recall")

	plt.legend()
	plt.ylabel("F1 component")
	plt.xlabel("sampling strategy value")
	plt.title("sampling strategy affection on evaluation metrics (2)")
	plt.savefig('png2.png')

	plt.figure(3)
	plt.plot(x, z3, label="mcc")

	plt.legend()
	plt.ylabel("mcc")
	plt.xlabel("sampling strategy value")
	plt.title("sampling strategy affection on evaluation metrics (3)")
	plt.savefig('png3.png')

def script(inputf, outputf):
	with open(inputf, "r") as file:
		lines = file.readlines()

	# Extract numbers after "Model Accuracy:" and save them
	accuracies = []
	for line in lines:
		if "Model Accuracy:" in line:
			accuracy = re.findall(r'\d+\.\d+', line)
			if accuracy:
				accuracies.append(float(accuracy[0]))

	# Draw the distribution on a graph
	plt.hist(accuracies, bins=9, range=(10,100), color='skyblue', edgecolor='black')
	plt.xlabel('Accuracy')
	plt.ylabel('Frequency')
	plt.title('Distribution of Model Accuracies')
	plt.savefig(outputf)

def confusion(csvf, out):
	
	df = pd.read_csv(csvf)
	
	predicted = df['predict'].tolist()
	truth = df['truth'].tolist()

	cm = confusion_matrix(truth, predicted)

	print(cm)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0 (Intron)', 'Predicted 1 (Exon)'], yticklabels=['True 0 (Intorn)', 'True 1 (Exon)'])
	plt.xlabel('Predicted labels')
	plt.ylabel('Actual labels')
	plt.title(f'Confusion Matrix {csvf}')
	plt.savefig(out)


if __name__ == "__main__":
   	# script("output_20.txt", "epoch20.png")
	confusion('./ft1_epoch20_5050ds2.csv', '5050ds2_confusion.png')