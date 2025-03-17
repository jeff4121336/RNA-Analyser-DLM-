import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

class Gene_ML:	
	def __init__(self):
		# Replace with real data (edit)
		X, y = make_multilabel_classification(n_samples=1000, n_features=20, n_classes=5, n_labels=2, random_state=1234)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
		
	def load_sequence(self):
		# Feature Scaling
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(self.X_train)
		X_test_scaled = scaler.transform(self.X_test)

		# PCA 
		pca = PCA(n_components=0.95)  # Choose the number of components to explain 95% variance (good features)
		X_train_pca = pca.fit_transform(X_train_scaled)
		X_test_pca = pca.transform(X_test_scaled)
		
		y_train = self.y_train
		y_test = self.y_test

		return X_train_pca, X_test_pca, y_train, y_test

	def classifer(self, X_train_pca, X_test_pca, y_train, y_test, use_chain = False, classifier_choice='svm'):
		
		# Train and Evaluate the Model
		# Paper method - OvR -> all problems into binary classification problems
		if classifier_choice == 'svm':
			base_model = SVC(kernel='linear', probability=True)
		elif classifier_choice == 'rf':
			base_model = RandomForestClassifier(n_estimators=6, random_state=1234)
		else:
			raise ValueError("Invalid classifier choice. Please choose 'svm' or 'rf'.")

		# New method - Consider all labels relationship too 
		# = each label is (a feature) related to other labels. 
		if use_chain: 
			model = ClassifierChain(base_model)
		else:
			model = OneVsRestClassifier(base_model)

		# Fit, Predict and Evaluate the Model
		model.fit(X_train_pca, y_train)
		y_pred = model.predict(X_test_pca)
		print(classification_report(y_test, y_pred))

		return y_pred

if __name__ == "__main__":
	classifer = 'svm'
	#classifer = 'rf'
	use_chain = False

	gene_data_ml = Gene_ML()
	X_train_pca, X_test_pca, y_train, y_test = gene_data_ml.load_sequence()
	y_pred = gene_data_ml.classifer(X_train_pca, X_test_pca, y_train, y_test, use_chain, classifer)