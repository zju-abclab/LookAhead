import pandas as pd
import joblib
import pickle
import time
import sys
import os

from contract_feature_handler import ALL_FEATURE_NAME

sys.path.append(
	os.path.abspath(os.path.join(os.path.dirname(__file__), "../../model_eval"))
)
from transformer_model import TransformerModel

MODEL_PATH = "../models"


class ClassificationEvaluator:
	def __init__(self):
		"""
		Initializes the ClassificationEvaluator class by loading the models and scaler.
		"""
		self.classifiers = {}  # A dictionary to store all classifiers

		# Load pre-trained models and scaler
		self.load_single_models()
		print(f"Single Models load ok!")
		self.load_hybridModel()
		print(f"HybridModel load ok!")
		self.loadScaler()
		print("Scaler load ok!")

	def evaluate_single(self, df, outputflag=True):
		# Select only relevant features for prediction
		df = df[ALL_FEATURE_NAME]

		# Preprocess the features for prediction
		df_to_predict_stardard = self.featurePreprocess(df)

		pred_Y_list = []  # List to hold predictions from each model
		time_start = time.time()  # Track the start time for performance

		# Loop through each classifier and predict probabilities
		for item in self.classifiers.keys():
			pred_Y = self.classifiers[item].predict_proba(df_to_predict_stardard)
			pred_Y = [
				item[1] for item in pred_Y
			]  # Extract the probability for the positive class
			pred_Y_list.append(pred_Y)

		time_end = time.time()  # End time for performance tracking
		time_exec = time_end - time_start  # Calculate the execution time
		self.last_time_exec = time_exec  # Store the last execution time

		if outputflag:
			for i, item in enumerate(self.classifiers.keys()):
				# print(f'Evaluate time ==> {time_exec}s')
				print(f"{item.ljust(30)}===> {pred_Y_list[i]}")

		return pred_Y_list

	def evaluate_hybrid(self, df, outputflag=True):
		# Select only relevant features for prediction
		df = df[ALL_FEATURE_NAME]

		# Preprocess the features for prediction
		df_to_predict_stardard = self.featurePreprocess(df)

		test_X_text = df["output_text_replaced"].to_numpy()

		xgboost_score_list = self.firstLevelXgboost.predict_proba(
			df_to_predict_stardard
		)
		xgboost_score_list = [item[1] for item in xgboost_score_list]
		# print(f'xgboost_score==>{xgboost_score_list}')

		transformer_score_list = self.firstLevelTransformer.evaluate2(test_X_text)
		# print(f'transformer_score==>{transformer_score_list}')

		X_secondLevel = pd.DataFrame(
			{
				"xgboost_score": xgboost_score_list,
				"transformer_score": transformer_score_list,
			}
		)

		meta_score_list = self.secondLevelKnn.predict_proba(X_secondLevel)
		meta_score_list = [item[1] for item in meta_score_list]

		print(f"meta_model  ===> {meta_score_list}")

		return meta_score_list

	def featurePreprocess(self, df):
		"""
		Preprocesses the features by handling various types and converting them to a standard format.

		Args:
		    df (DataFrame): The input data to preprocess.

		Returns:
		    DataFrame: The preprocessed data ready for prediction.
		"""
		# Convert hexadecimal strings to integers for 'gasUsed', 'value', and 'nonce'
		df["txDataLen"] = df["txDataLen"].astype(float)
		df["gasUsed"] = [int(item, 16) for item in df["gasUsed"].tolist()]
		df["value"] = [int(item, 16) for item in df["value"].tolist()]
		df["value"] = (df["value"] > 0).astype(int)  # Convert value to binary (0 or 1)
		df["nonce"] = [int(item, 16) for item in df["nonce"].tolist()]

		# Format the features and standardize them
		df_format = self.formatFeatures(df)
		df_to_predict_formatted = df_format.drop("contract_address", axis=1)
		df_to_predict_stardard = self.standardize(df_to_predict_formatted)

		return df_to_predict_stardard

	def formatFeatures(self, df):
		"""
		Formats the features by creating dummy variables and handling categorical data.

		Args:
		    df (DataFrame): The input data to format.

		Returns:
		    DataFrame: The formatted data with categorical features converted to dummy variables.
		"""
		dataframe = df[ALL_FEATURE_NAME]
		dataframe.drop(
			"output_text_replaced", axis=1, inplace=True
		)  # Drop unnecessary feature

		# Define the possible categories for the 'fund_from_label' feature
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

		# Convert categorical 'fund_from_label' into dummy variables
		dataframe = pd.get_dummies(
			dataframe, columns=["fund_from_label"], prefix="fund_from_label"
		)

		return dataframe

	def standardize(self, df):
		"""
		Standardizes the features using the pre-loaded scaler.

		Args:
		    df (DataFrame): The input data to standardize.

		Returns:
		    DataFrame: The standardized data.
		"""
		all_X_stardard = self.scaler.transform(df)
		df_stardard = pd.DataFrame(all_X_stardard, columns=df.columns)
		return df_stardard

	def loadScaler(self):
		"""
		Loads the pre-trained scaler from disk.
		"""
		self.scaler = joblib.load(f"{MODEL_PATH}/scaler.pkl")

	def load_single_models(self):
		model_names = [
			"Logistic_Regression",
			"Decision_Trees_Classifier",
			"Random_Forest_Classifier",
			"XGBoost_Classifier",
		]

		self.classifiers = {}

		# Load each model
		for name in model_names:
			try:
				with open(f"{MODEL_PATH}/{name}.pkl", "rb") as f:
					model = pickle.load(f)
					self.classifiers[name] = model
				print(f"The {name} model has been successfully loaded.")
			except FileNotFoundError:
				print(f"The {name}.pkl file was not found.")
			except Exception as e:
				print(f"An error occurred while loading the {name} model: {e}")

	def load_hybridModel(self):
		with open(f"{MODEL_PATH}/Hybrid_xgboost.pkl", "rb") as f:
			self.firstLevelXgboost = pickle.load(f)
		with open(f"{MODEL_PATH}/Hybrid_knn.pkl", "rb") as f:
			self.secondLevelKnn = pickle.load(f)

		self.firstLevelTransformer = TransformerModel(
			saved_model_path=f"{MODEL_PATH}/transformer.keras",
			saved_vectorize_layer_path=f"{MODEL_PATH}/transformer_vectorize_layer.keras",
		)
