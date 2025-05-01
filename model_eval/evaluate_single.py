import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from data_loader import getCrossDataSet
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	classification_report,
	roc_auc_score,
)
import random

random_number = random.randint(0, 100000)

# Define the names of all features
ALL_FEATURE_NAME = [
	"contract_address",
	"value",
	"gasUsed",
	"nonce",
	"txDataLen",
	"contract_bytecodeLen",
	"verify_tag",
	"fund_from_label",
	"PublicFuncNum",
	"PrivateFuncNum",
	"totalFuncNum",
	"publicFuncProp",
	"flashloanCallbackNum",
	"flashloanCallbackProp",
	"externalCallNum",
	"internalCallNum",
	"totalCallNum",
	"externalCallProp",
	"delegateCallNum",
	"tokenCallNum",
	"tokenCallProp",
	"maxTokenCallNum",
	"avgTokenCallNum",
	"isTokenContract",
	"isERC1967Proxy",
	"isSelfDestructive",
	"transferCallCount",
	"balanceOfCallCount",
	"approveCallCount",
	"totalSupplyCallCount",
	"allowanceCallCount",
	"transferFromCallCount",
	"mintCallCount",
	"burnCallCount",
	"withdrawCallCount",
	"depositCallCount",
	"skimCallCount",
	"syncCallCount",
	"token0CallCount",
	"token1CallCount",
	"getReservesCallCount",
	"swapCallCount",
	"output_text_replaced",
]


def get_classification_report(y_true, y_pred):
	"""
	Calculate various indicators of the classification report.
	:param y_true: True labels
	:param y_pred: Predicted labels
	:return: Accuracy, precision, recall, F1 score, false positive rate
	"""
	accuracy = accuracy_score(y_true, y_pred)
	conf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
	report = classification_report(
		y_true,
		y_pred,
		digits=4,
		target_names=["Benign", "Adversarial"],
		output_dict=True,
	)
	TP = conf_matrix[0][0]
	FP = conf_matrix[1][0]
	FN = conf_matrix[0][1]
	TN = conf_matrix[1][1]
	precision = report["Adversarial"]["precision"]
	# False Positive Rate (FPR) = FP / (FP + TN)
	FPR = FP / (FP + TN)
	# True Positive Rate (TPR) = TP / (TP + FN)
	TPR = report["Adversarial"]["recall"]
	f1_score = report["Adversarial"]["f1-score"]
	auc = roc_auc_score(y_true, y_pred)
	return accuracy, precision, TPR, f1_score, FPR


class SingleModelEvaluator:
	def __init__(self, dataset_dic):
		"""
		Initialize the evaluator.
		:param dataset_dic: A dictionary containing the training set and test set
		"""
		self.train_dataset = dataset_dic["train"]
		self.test_dataset = dataset_dic["test"]

		print(f"train dataset size: {len(self.train_dataset)}")
		print(f"test dataset size: {len(self.test_dataset)}")

		self.datasetInit()
		self.modelInit()
		self.modelTrainEvaluate()

	def modelTrainEvaluate(self):
		"""
		Train and evaluate classifiers.
		"""
		self.classifiers_predictions = []
		# Train and evaluate the classifiers
		print("Now training and evaluating the classifiers.")
		for i, classifier in enumerate(self.classifiers.values()):
			classifier.fit(self.train_X_stardard, self.train_Y)
			pred_Y = classifier.predict(self.test_X_stardard)
			self.classifiers_predictions.append(pred_Y)

		print(f"train dataset size: {len(self.train_dataset)}")
		print(f"test dataset size: {len(self.test_dataset)}")
		print(
			"Model Name                              acc    prec   recall f1     FPR    "
		)

		self.f1_results = []
		# Classifier results
		for classifier_idx in range(len(self.classifiers.values())):
			accuracy, precision, tpr, f1_score, fpr = get_classification_report(
				self.test_Y, self.classifiers_predictions[classifier_idx]
			)
			print(f"{list(self.classifiers.keys())[classifier_idx]:<40}", end="")
			print(
				f"{accuracy:.4f} {precision:.4f} {tpr:.4f} {f1_score:.4f} {fpr:.4f}\n",
				end="",
			)
			self.f1_results.append(f1_score)

	def modelInit(self):
		"""
		Initialize classifiers.
		"""
		self.classifiers = {
			"Logistic Regression": LogisticRegression(
				tol=0.01, random_state=random_number
			),
			"Decision Trees Classifier": DecisionTreeClassifier(
				criterion="gini", random_state=random_number
			),
			"Random Forest Classifier": RandomForestClassifier(
				n_estimators=5, criterion="gini", random_state=random_number
			),
			"XGBoost Classifier": XGBClassifier(
				objective="binary:logistic",
				eval_metric="logloss",
				random_state=random_number,
			),
		}

	def datasetInit(self):
		"""
		Initialize the dataset, including feature preprocessing and standardization.
		"""
		self.train_X = self.featurePreprocess(self.train_dataset[ALL_FEATURE_NAME])
		self.train_Y = self.train_dataset["tag"].tolist()
		self.test_X = self.featurePreprocess(self.test_dataset[ALL_FEATURE_NAME])
		self.test_Y = self.test_dataset["tag"].tolist()

		self.train_X, self.train_Y = self.applyAdasyn(self.train_X, self.train_Y)

		scaler = StandardScaler()
		train_X_tmp = scaler.fit_transform(self.train_X)
		self.train_X_stardard = pd.DataFrame(train_X_tmp, columns=self.train_X.columns)

		test_X_tmp = scaler.transform(self.test_X)
		self.test_X_stardard = pd.DataFrame(test_X_tmp, columns=self.train_X.columns)

	def applyAdasyn(self, X, y):
		from collections import Counter

		print("Before oversampling:", Counter(y))

		ada = ADASYN(
			sampling_strategy="auto", random_state=random_number, n_neighbors=5
		)
		X_resampled, y_resampled = ada.fit_resample(X, y)

		print("After oversampling:", Counter(y_resampled))
		return X_resampled, y_resampled

	def featurePreprocess(self, df):
		"""
		Feature preprocessing.
		:param df: DataFrame
		:return: Processed DataFrame
		"""
		df["txDataLen"] = df["txDataLen"].astype(float)
		df["gasUsed"] = [int(item, 16) for item in df["gasUsed"].tolist()]
		df["value"] = [int(item, 16) for item in df["value"].tolist()]
		df["value"] = (df["value"] > 0).astype(int)
		df["nonce"] = [int(item, 16) for item in df["nonce"].tolist()]
		df["verify_tag"] = df["verify_tag"].replace({"True": 1, "False": 0})
		df_format = self.formatFeatures(df)
		df_to_predict_formatted = df_format.drop("contract_address", axis=1)
		return df_to_predict_formatted

	def formatFeatures(self, dataframe):
		"""
		Format features, including deleting columns and one-hot encoding.
		:param dataframe: DataFrame
		:return: Processed DataFrame
		"""
		dataframe.drop("output_text_replaced", axis=1, inplace=True)
		fund_from_label_list = [
			" Anonymous",
			"Safe",
			"unknown_low",
			"Bridge",
			"unknown_contract",
			"unknown_high",
		]
		dataframe["fund_from_label"] = pd.Categorical(
			dataframe["fund_from_label"], categories=fund_from_label_list
		)
		dataframe = pd.get_dummies(
			dataframe, columns=["fund_from_label"], prefix="fund_from_label"
		)
		return dataframe


def showpic(data):
	"""
	Draw a line chart of the F1 scores of different models.
	:param data: F1 score data
	"""
	# Model names
	model_names = [
		"Logistic Regression",
		"Decision Trees Classifier",
		"Random Forest Classifier",
		"XGBoost Classifier",
	]

	# Convert data to a NumPy array
	data = np.array(data)

	# Transpose the data so that each column represents a model
	data = data.T

	# Generate x-axis coordinates
	x = np.arange(1, len(data[0]) + 1)

	# Set the picture clarity
	plt.rcParams["figure.dpi"] = 100

	# Draw each line
	for i in range(len(data)):
		plt.plot(x, data[i], marker="o", label=model_names[i])

	# Set the chart title and axis labels
	plt.title("F1 Scores of Different Models")
	plt.xlabel("Index")
	plt.xticks(rotation=45)
	plt.ylabel("F1 Score")

	# Display the legend
	plt.legend()

	# Display the grid lines
	plt.grid(True)

	# Display the chart
	plt.savefig("result.png")


if __name__ == "__main__":
	fold_dataset = getCrossDataSet()

	data = []
	for i in range(4):
		modelEvaluator = SingleModelEvaluator(fold_dataset[i])
		data.append(modelEvaluator.f1_results)
	for item in data:
		print(item, end=",\n")
	# showpic(data)
