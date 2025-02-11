from flask import Flask, render_template, request, jsonify, send_from_directory
import fm, json
import torch
from dslayer import RNAClassifier_1, RNAClassifier_2
from utils import prepare
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os, shutil


app = Flask(__name__, static_url_path='/static')
modelSelected = None

num_channels = 30
kernel_size = 5
dropout_rate = 0.2
padding = 2


rna_list0 = ["5S rRNA", "5.8S rRNA", "tRNA", "ribozyme", "CD-box", "miRNA",
			 "Intron_gpI", "Intron_gpII", "scaRNA", "HACA-box", "riboswitch", 
			 "IRES", "leader", "unknown", "pad"]

rna_list1 = ["5S rRNA", "5.8S rRNA", "tRNA", "ribozyme", "CD-box", "miRNA",
			 "Intron_gpI", "Intron_gpII", "scaRNA", "HACA-box", "riboswitch", 
			 "IRES", "leader", "mRNA", "unknown", "pad"]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
app = Flask(__name__)

# Directory containing your images
IMAGE_FOLDER = 'static/images'

@app.route('/' , methods=['GET'])
def greeting():
	return render_template('index.html')

@app.route('/api/images', methods=['GET'])
def get_images():
	try:
		files = os.listdir(IMAGE_FOLDER) # List all files in the image directory
		images = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))] # Filter for specific image types
		return jsonify(images)
	except Exception as e:
		return jsonify({'error': str(e)}), 500

@app.route('/static/images/<path:filename>', methods=['GET'])
def serve_image(filename):
	return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/change' , methods=['POST'])
def changeModel():
	global modelSelected
	data = request.get_json()  # Get JSON data from the request
	modelSelected = data.get('model', 'Default Value')  # Change 'model' to the key you're using
	return jsonify({"modelSelected": modelSelected}), 200 

@app.route('/predict', methods=['POST'])
def predict():
	if modelSelected is None:
		return jsonify({"message": "Model not loaded"}), 500  # Return a 500 error if the model is not loaded
	
	for filename in os.listdir('./static/images/'):
		file_path = os.path.join('./static/images/', filename)
		os.remove(file_path) 

	seq = request.json
	seq = seq['input']
	seq = seq.replace(' ', ''); 
	x, tt = prepare(seq)
	
	if modelSelected == "m1":
		model = RNAClassifier_1(len(rna_list0), num_channels, kernel_size, dropout_rate, padding).to(device)
		model.load_state_dict(torch.load('./models/scc1.pt')['model_state_dict'])
	else:
		model = RNAClassifier_1(len(rna_list1), num_channels, kernel_size, dropout_rate, padding).to(device)
		model.load_state_dict(torch.load('./models/scc2.pt')['model_state_dict'])
		model.eval()

	output, feature_maps = model(x)  
	# Prediction
	final_output = torch.sum(output, dim=1)
	final_output = final_output[:, :-2]  # Assuming you want to exclude the last 2 dimensions

	scores_list = []

	# print(final_output.shape)
	if final_output.shape[0] != 1 :
		final_output = torch.sum(final_output, dim=0)
		final_output = final_output.view(1, final_output.shape[0])

	min_val = final_output.min(dim=1, keepdim=True).values 
	max_val = final_output.max(dim=1, keepdim=True).values  
	epsilon = 1e-8
	final_output_normalized = (final_output - min_val) / (max_val - min_val + epsilon)
	scores_list = final_output_normalized.flatten().tolist()
	
	ranked_scores = [(rna_list1[i], round(score * 100, 3)) for i, score in enumerate(scores_list)]
	ranked_scores.sort(key=lambda x: x[1], reverse=True)
	formatted_scores = '</br>'.join([f"{rna_type}: {score}%" for rna_type, score in ranked_scores])

	prob, return_prediction = torch.max(final_output_normalized, dim=1)
	y_pred = return_prediction.cpu().numpy().tolist()[0]
	predicted = rna_list1[y_pred]
	 
	# Heatmap
	y_pred_reshaped = output.view(-1, 15) if modelSelected == "m1" else output.view(-1, 16)
	tt[tt == -1] = y_pred
	for i in range(output.shape[0]):
		print(f"Processing sample {i}")
		# Change the input for each iteration if possible
		# Here we assume x is a batch of data; you might want to slice or modify it
		x_i = x[i : i + 1]  # Modify this if you have a specific way to select different inputs
		y_pred_reshaped_cut = y_pred_reshaped[30 * i: 30 * (i + 1), :]
		
		criterion = nn.CrossEntropyLoss()
		t = tt[i, :]
		print(f"Target shape: {tt.shape}, Predicted shape: {y_pred_reshaped_cut.shape}")

		loss = criterion(y_pred_reshaped_cut, t)
		loss.backward(retain_graph=True)  # Retain the graph 
		weight_gradients = model.dense2.dense.weight.grad.detach()
		bias_gradients = model.dense2.dense.bias.grad.detach()

		# Get conv output
		conv_output = model.conv1(x_i).detach()  # Use modified input
		pooled_gradients = torch.mean(weight_gradients, dim=0) + torch.mean(bias_gradients, dim=0)
		for j in range(conv_output.size(1)):
			conv_output[0, j, :] *= pooled_gradients[j]
	
		# (1, 30, 640) -> (1, 30) -> (30, )
		heatmap = torch.mean(conv_output, dim=2)
		heatmap = heatmap.view(-1).detach().cpu().numpy()

		heatmap = np.maximum(heatmap, 0)  # Use positive contributions only
		heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1  # Avoid division by zero
		# Normalize the heatmap
		heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
		
		print(f"Heatmap shape for sample {i}: {heatmap.shape}")

		start_idx = 1 + 30 * i
		end_idx = 31 + 30 * i
		plt.figure(figsize=(10, 5))
		plt.bar(range(start_idx, end_idx), heatmap, color='blue')
		plt.xlabel('Segment Index')
		plt.ylabel('Average Activation')
		plt.title('Average Activation per Segment')
		plt.xticks(range(start_idx, end_idx))
		plt.savefig(f"static/images/importance_{start_idx}_to_{end_idx}.png")
		plt.close()  # Close the figure to free up memory

		model.zero_grad()  # Reset gradients for the next iteration

	return jsonify({"prediction": predicted, "score": formatted_scores }), 200


if __name__ == "__main__":
	app.run(port=5500, debug=True)