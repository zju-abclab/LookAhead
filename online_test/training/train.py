import sys
import os
import random
import pickle
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from data_loader import getDataSet
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	classification_report,
	roc_auc_score,
)

sys.path.append(
	os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_eval"))
)
from transformer_model import TransformerModel

random_number = random.randint(0, 100000)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

MODEL_PATH = "../models"

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


class SingleModelTrainer:
	def __init__(self, train_dataset):
		self.train_dataset = train_dataset

		print(f"train dataset size: {len(self.train_dataset)}")
		self.datasetInit()
		self.modelInit()
		self.modelTrain()
		self.modelSave()

	def modelSave(self):
		for name, model in self.classifiers.items():
			try:
				with open(f"{MODEL_PATH}/{name}.pkl", "wb") as f:
					pickle.dump(model, f)
				print(f"The {name} model has been successfully saved.")
			except Exception as e:
				print(f"An error occurred while saving the {name} model: {e}")

	def modelTrain(self):
		self.classifiers_predictions = []
		print("Now training the classifiers.")
		for i, classifier in enumerate(self.classifiers.values()):
			classifier.fit(self.train_X_stardard, self.train_Y)

		print(f"train dataset size: {len(self.train_dataset)}")

	def modelInit(self):
		"""
		Initialize classifiers.
		"""
		self.classifiers = {
			"Logistic_Regression": LogisticRegression(
				tol=0.01, random_state=random_number
			),
			"Decision_Trees_Classifier": DecisionTreeClassifier(
				criterion="gini", random_state=random_number
			),
			"Random_Forest_Classifier": RandomForestClassifier(
				n_estimators=5, criterion="gini", random_state=random_number
			),
			"XGBoost_Classifier": XGBClassifier(
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

		scaler = StandardScaler()
		train_X_tmp = scaler.fit_transform(self.train_X)
		self.train_X_stardard = pd.DataFrame(train_X_tmp, columns=self.train_X.columns)

		joblib.dump(scaler, f"{MODEL_PATH}/scaler.pkl")

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


class HybridModelTrainer:
	def __init__(self, train_dataset):
		"""
		Initialize the HybridModelEvaluator.

		Args:
		    dataset_dic (dict): Dictionary containing training and test datasets.
		"""
		self.train_dataset = train_dataset
		# self.test_dataset = dataset_dic['test']

		# Split the training dataset into two parts (3:1) for first and second level models
		split = StratifiedShuffleSplit(
			n_splits=1, test_size=0.25, random_state=random_number
		)
		for train1_index, train2_index in split.split(
			self.train_dataset, self.train_dataset["tag"]
		):
			self.train_X1 = self.train_dataset.loc[train1_index].reset_index(drop=True)
			self.train_X2 = self.train_dataset.loc[train2_index].reset_index(drop=True)

		print(f"train_X1 shape: {self.train_X1.shape}")
		print(f"train_X2 shape: {self.train_X2.shape}")
		print(f"train dataset size: {len(self.train_dataset)}")

		self.datasetInit()
		self.modelInit()
		self.trainFirstLevelXgboost()
		self.trainFirstLevelTransformer()
		self.trainSecondLevel()
		# self.trainFirstLevelAgain()
		self.modelSave()
		# self.testDataEvaluate()
		print("ok!")

	def modelSave(self):
		with open(f"{MODEL_PATH}/Hybrid_xgboost.pkl", "wb") as f:
			pickle.dump(self.firstLevelXgboost, f)
		self.firstLevelTransformer.saveModel(
			model_path=f"{MODEL_PATH}/transformer.keras",
			vectorize_layer_path=f"{MODEL_PATH}/transformer_vectorize_layer.keras",
		)
		with open(f"{MODEL_PATH}/Hybrid_knn.pkl", "wb") as f:
			pickle.dump(self.secondLevelKnn, f)
		print("Model saved!")

	def trainFirstLevelAgain(self):
		"""
		Train the first-level XGBoost model again on the whole training dataset.
		"""
		self.firstLevelXgboost.fit(self.train_X_stardard, self.train_Y)

	def testDataEvaluate(self):
		"""
		Evaluate the models on the test dataset.
		"""
		self.test_X_text = self.test_dataset["output_text_replaced"].to_numpy()
		self.test_Y = self.test_dataset["tag"].tolist()

		# First-level predictions
		xgboost_score_list = self._get_xgboost_scores(self.test_X_stardard)
		transformer_score_list = self.firstLevelTransformer.evaluate2(self.test_X_text)

		X_secondLevel = pd.DataFrame(
			{
				"xgboost_score": xgboost_score_list,
				"transformer_score": transformer_score_list,
			}
		)

		classifiers_predictions = []

		# Evaluate XGBoost and Transformer separately
		self._evaluate_single_model("xgboost", xgboost_score_list, self.test_Y)
		self._evaluate_single_model("transformer", transformer_score_list, self.test_Y)

		# Second-level predictions
		print(
			"Model Name                              acc    prec   recall f1     FPR    "
		)
		for key, classifier in self.secondLevelClassifiers.items():
			pred_Y = classifier.predict(X_secondLevel)
			classifiers_predictions.append(pred_Y)
			self._print_classification_metrics(key, self.test_Y, pred_Y)

	def _get_xgboost_scores(self, X):
		"""
		Get XGBoost scores for the given input.

		Args:
		    X (pd.DataFrame): Input data.

		Returns:
		    list: XGBoost scores.
		"""
		xgboost_score_list = self.firstLevelXgboost.predict_proba(X)
		return [item[1] for item in xgboost_score_list]

	def _evaluate_single_model(self, model_name, scores, y_true):
		"""
		Evaluate a single model and print the classification metrics.

		Args:
		    model_name (str): Name of the model.
		    scores (list): Model scores.
		    y_true (list): True labels.
		"""
		scores_01 = np.round(scores).astype(int)
		print(
			"Model Name                              acc    prec   recall f1     FPR    "
		)
		self._print_classification_metrics(model_name, y_true, scores_01)

	def _print_classification_metrics(self, model_name, y_true, y_pred):
		"""
		Print the classification metrics for a model.

		Args:
		    model_name (str): Name of the model.
		    y_true (list): True labels.
		    y_pred (list): Predicted labels.
		"""
		accuracy, precision, TPR, f1_score, FPR = get_classification_report(
			y_true, y_pred
		)
		print(f"{model_name:<40}", end="")
		line = f"{accuracy:.4f} {precision:.4f} {TPR:.4f} {f1_score:.4f} {FPR:.4f}"
		print(line)

	def trainFirstLevelXgboost(self):
		"""
		Train the first-level XGBoost model.
		"""
		self.firstLevelXgboost.fit(self.train_X1_stardard, self.train_Y1)

	def trainFirstLevelTransformer(self):
		"""
		Train the first-level Transformer model.
		"""
		self.train_X1_text = self.train_X1["output_text_replaced"].to_numpy()
		self.train_X2_text = self.train_X2["output_text_replaced"].to_numpy()

		self.firstLevelTransformer.train(
			10,
			self.train_X1_text,
			np.array(self.train_Y1),
			False,
			False,
			False,
			None,
			None,
		)

	def trainSecondLevel(self):
		"""
		Train the second-level models.
		"""
		self.train_X2_text = self.train_X2["output_text_replaced"].to_numpy()

		# First-level predictions
		xgboost_score_list = self._get_xgboost_scores(self.train_X2_stardard)
		transformer_score_list = self.firstLevelTransformer.evaluate2(
			self.train_X2_text
		)

		train_X_secondLevel_df = pd.DataFrame(
			{
				"xgboost_score": xgboost_score_list,
				"transformer_score": transformer_score_list,
			}
		)
		train_Y_secondLevel = self.train_X2["tag"].tolist()

		self.secondLevelKnn.fit(train_X_secondLevel_df, train_Y_secondLevel)

	def modelInit(self):
		"""
		Initialize the models.
		"""
		self.firstLevelXgboost = XGBClassifier(
			objective="binary:logistic", eval_metric="logloss"
		)

		train_X_text = self.train_dataset["output_text_replaced"].tolist()
		self.firstLevelTransformer = TransformerModel(dataset_X=np.array(train_X_text))

		self.secondLevelKnn = KNeighborsClassifier(n_neighbors=10)

	def datasetInit(self):
		"""
		Initialize the dataset by preprocessing and standardizing the data.
		"""
		self.train_X_formatted = self.featurePreprocess(
			self.train_dataset[ALL_FEATURE_NAME]
		)
		self.train_Y = self.train_dataset["tag"].tolist()
		self.train_X1_formatted = self.featurePreprocess(
			self.train_X1[ALL_FEATURE_NAME]
		)
		self.train_Y1 = self.train_X1["tag"].tolist()
		self.train_X2_formatted = self.featurePreprocess(
			self.train_X2[ALL_FEATURE_NAME]
		)
		self.train_Y2 = self.train_X2["tag"].tolist()

		scaler = StandardScaler()
		scaler.fit(self.train_X_formatted)

		self.train_X_stardard = self._standardize_data(scaler, self.train_X_formatted)
		self.train_X1_stardard = self._standardize_data(scaler, self.train_X1_formatted)
		self.train_X2_stardard = self._standardize_data(scaler, self.train_X2_formatted)

	def _standardize_data(self, scaler, data):
		"""
		Standardize the given data using the provided scaler.

		Args:
		    scaler (StandardScaler): Scaler object.
		    data (pd.DataFrame): Data to be standardized.

		Returns:
		    pd.DataFrame: Standardized data.
		"""
		data_tmp = scaler.transform(data)
		return pd.DataFrame(data_tmp, columns=data.columns)

	def featurePreprocess(self, df):
		"""
		Preprocess the features in the given DataFrame.

		Args:
		    df (pd.DataFrame): Input DataFrame.

		Returns:
		    pd.DataFrame: Preprocessed DataFrame.
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
		Format the features in the given DataFrame.

		Args:
		    dataframe (pd.DataFrame): Input DataFrame.

		Returns:
		    pd.DataFrame: Formatted DataFrame.
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


if __name__ == "__main__":
	train_dataset = getDataSet()

	modelTrainer = SingleModelTrainer(train_dataset)
	modelTrainer = HybridModelTrainer(train_dataset[:15000])
	# modelTrainer = HybridModelTrainer(train_dataset[])
