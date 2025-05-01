import numpy as np

from tensorflow.data import Dataset
from tensorflow.keras import Model, Sequential, saving
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import shape, range, string, convert_to_tensor, distribute
from tensorflow.keras.layers import (
	TextVectorization,
	MultiHeadAttention,
	LayerNormalization,
	Layer,
	Input,
	Dense,
	Dropout,
	Embedding,
	GlobalAveragePooling1D,
)
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	classification_report,
	roc_auc_score,
)


def get_classification_report(y_true, y_pred):
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


SEQUENCE_LENGTH = 1500
MAX_TOKENS = 25000
EMBEDDING_DIMENSION = 256
TOTAL_HEADS = 8
TOTAL_DENSE_UNITS = 128

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1


class EmbeddingLayer(Layer):
	def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
		super(EmbeddingLayer, self).__init__(**kwargs)
		self.word_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
		self.position_embedding = Embedding(
			input_dim=sequence_length, output_dim=embed_dim
		)

	def build(self):
		pass

	def call(self, tokens):
		sequence_length = shape(tokens)[-1]
		all_positions = range(start=0, limit=sequence_length, delta=1)
		positions_encoding = self.position_embedding(all_positions)
		words_encoding = self.word_embedding(tokens)
		return positions_encoding + words_encoding


class EncoderLayer(Layer):
	def __init__(self, total_heads, total_dense_units, embed_dim, **kwargs):
		super(EncoderLayer, self).__init__(**kwargs)
		self.multihead = MultiHeadAttention(num_heads=total_heads, key_dim=embed_dim)
		self.nnw = Sequential(
			[Dense(total_dense_units, activation="gelu"), Dense(embed_dim)]
		)
		self.normalize_layer = LayerNormalization()

	def build(self):
		pass

	def call(self, inputs):
		attn_output = self.multihead(inputs, inputs)
		normalize_attn = self.normalize_layer(inputs + attn_output)
		nnw_output = self.nnw(normalize_attn)
		final_output = self.normalize_layer(normalize_attn + nnw_output)
		return final_output


class TransformerModel:
	def __init__(
		self,
		use_dropout=False,
		dataset_X=None,
		saved_model_path=None,
		saved_vectorize_layer_path=None,
	):
		# Automatically distribute the model across multiple GPUs if available
		strategy = distribute.MirroredStrategy()
		print("Number of GPUs: {}".format(strategy.num_replicas_in_sync))

		# Load the trained model from the disk if the path is provided
		if (
			saved_model_path is not None
			and saved_vectorize_layer_path is not None
			and dataset_X is None
		):
			self.transformer_model = saving.load_model(saved_model_path)
			self.vectorize_layer = saving.load_model(saved_vectorize_layer_path).layers[
				0
			]
			return
		elif dataset_X is None:
			raise ValueError("Provide either a dataset or a saved model path")

		# Compile the model
		with strategy.scope():
			self.vectorize_layer = TextVectorization(
				output_sequence_length=SEQUENCE_LENGTH, max_tokens=MAX_TOKENS
			)
			self.vectorize_layer.adapt(
				Dataset.from_tensor_slices(dataset_X).batch(BATCH_SIZE)
			)
			VOCAB_SIZE = len(self.vectorize_layer.get_vocabulary())
			print("Vocabulary size: ", VOCAB_SIZE)

			embedding_layer = EmbeddingLayer(
				SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIMENSION
			)
			encoder_layer = EncoderLayer(
				TOTAL_HEADS, TOTAL_DENSE_UNITS, EMBEDDING_DIMENSION
			)

			# Transformer layers
			inputs = Input(shape=(SEQUENCE_LENGTH,))

			emb = embedding_layer(inputs)
			if use_dropout:
				emb = Dropout(0.3)(emb)

			enc = encoder_layer(emb)
			if use_dropout:
				enc = Dropout(0.3)(enc)

			gap = GlobalAveragePooling1D()(enc)
			outputs = Dense(1, activation="sigmoid")(gap)

			# Create the model
			self.transformer_model = Model(inputs=inputs, outputs=outputs)
			self.transformer_model.compile(
				optimizer="adam",
				loss="binary_crossentropy",
				metrics=["BinaryAccuracy"],
			)

	def train(
		self,
		epoch_count,
		train_X,
		train_Y,
		early_stopping=False,
		dynamic_lr_schedule=False,
		evaluate_immediately=False,
		test_X=None,
		test_Y=None,
	):
		# Convert inputs to vectorized tensors
		train_X_vectorized = self.vectorize_layer(
			convert_to_tensor(train_X, dtype=string)
		)

		# Define the callbacks
		callbacks = []
		if early_stopping:
			callbacks.append(
				EarlyStopping(
					monitor="val_loss",
					min_delta=1e-2,
					patience=2,
					verbose=1,
					restore_best_weights=False,
				)
			)
		if dynamic_lr_schedule:
			callbacks.append(
				ReduceLROnPlateau(
					monitor="val_loss", factor=0.2, patience=2, min_lr=0.001
				)
			)

		# Train the model
		self.transformer_model.fit(
			train_X_vectorized,
			train_Y,
			batch_size=BATCH_SIZE,
			epochs=epoch_count,
			validation_split=VALIDATION_SPLIT,
			callbacks=callbacks,
		)

		if evaluate_immediately:
			self.evaluate(test_X, test_Y)

	def saveModel(
		self,
		model_path="../models/transformer.keras",
		vectorize_layer_path="../models/transformer_vectorize_layer.keras",
	):
		# Save the trained model to the disk
		self.transformer_model.save(model_path, True)

		# Save the vectorize_layer separately to avoid issues with multi-GPU training
		vectorize_layer_model = Sequential([self.vectorize_layer])
		vectorize_layer_model.save(vectorize_layer_path, True)

		print("Transformer and vectorize_layer models saved")

	def evaluate2(self, test_X):
		train_X_vectorized = self.vectorize_layer(
			convert_to_tensor(test_X, dtype=string)
		)
		test_Y_pred = self.transformer_model.predict(train_X_vectorized)
		test_Y_pred = [item[0] for item in test_Y_pred]
		return test_Y_pred

	def evaluate(self, test_X, test_Y):
		train_X_vectorized = self.vectorize_layer(
			convert_to_tensor(test_X, dtype=string)
		)

		test_Y_pred = self.transformer_model.predict(train_X_vectorized)
		# print(test_Y_pred)
		test_Y_pred_01 = [item[0] for item in test_Y_pred]
		test_Y_pred_01 = np.round(test_Y_pred_01).astype(int)

		# print(test_Y_pred.tolist().count(1),test_Y_pred.tolist().count(0))
		# print()

		# print(metrics.classification_report(test_Y, test_Y_pred,digits=4))
		accuracy, precision, TPR, f1_score, FPR = get_classification_report(
			test_Y, test_Y_pred_01
		)
		print(
			"Model Name                              acc    prec   recall f1     FPR    "
		)
		print(f"{"transformer":<40}", end="")
		line = f"{accuracy:.4f} {precision:.4f} {TPR:.4f} {f1_score:.4f} {FPR:.4f}"
		print(line)
		return test_Y_pred

	def predict(self, test_X):
		train_X_vectorized = self.vectorize_layer(
			convert_to_tensor(test_X, dtype=string)
		)

		return self.transformer_model.predict(train_X_vectorized)
