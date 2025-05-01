import pandas as pd
import sqlite3

DATASET_PATH = "../../dataset"


# Feature Database
class FeatureDB:
	def __init__(self, db_name):
		self.db_name = db_name
		self.conn = sqlite3.connect(db_name)
		self.cursor = self.conn.cursor()
		self.create_all_table()

	def create_all_table(self):
		self.create_table_template("benign_table")
		self.create_table_template("attack_ether_table")
		self.create_table_template("attack_bsc_table")
		self.create_table_template("to_predict_table")

	def create_table_template(self, table_name):
		gigahorse_feature_list = [
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
		]
		feature_list = [item + " TEXT NOT NULL" for item in gigahorse_feature_list]
		feature_string = ",".join(feature_list)
		create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                contract_address TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL,
                gasUsed TEXT NOT NULL,
                nonce TEXT NOT NULL,
                txDataLen TEXT NOT NULL,
                contract_bytecodeLen TEXT NOT NULL,
                verify_tag TEXT NOT NULL,
                fund_from_label TEXT NOT NULL,
                {feature_string},
                output_text_replaced TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def insert_data_template(self, table_name, data_to_insert):
		self.__insert_data(table_name, data_to_insert)

	def query_table_template(self, table_name, col_list=[]):
		return self.__query_table(table_name, col_list)

	def __insert_data(self, table_name, data_to_insert):
		columns = ", ".join(data_to_insert.keys())
		placeholders = ", ".join("?" * len(data_to_insert))
		values = tuple(data_to_insert.values())
		self.cursor.execute(
			f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", values
		)
		self.conn.commit()

	def __query_exec(self, query_string):
		self.cursor.execute(query_string)
		rows = self.cursor.fetchall()
		return rows

	def __query_table(self, table_name, col_list):
		query_string = ""
		if len(col_list) == 0:
			query_string = f"SELECT * FROM {table_name}"
		else:
			col_string = ",".join(col_list)
			query_string = f"SELECT {col_string} FROM {table_name}"
		rows = self.__query_exec(query_string)
		return rows

	def __create_table(self, create_table_query):
		self.cursor.execute(create_table_query)
		self.conn.commit()

	def close_connection(self):
		self.conn.close()

	def query_exec(self, query_string):
		return self.__query_exec(query_string)


class DataDB:
	# Transaction and data database
	# txhash_table
	#     - contract_address
	#     - tx_hash
	# tx_info_table
	#     - tx_hash
	#     - contract_address
	#     - msg_sender
	#     - value
	#     - gasUsed
	#     - nonce
	#     - input
	#     - blockNumber
	#     - transactionIndex
	# verify_table
	#     - contract_address
	#     - verify_tag
	# bytecode_table
	#     - contract_address
	#     - bytecode
	# fundsource_table
	#     - msg_sender
	#     - label
	#     - trace_list
	#     - hop_count
	def __init__(self, db_name):
		self.db_name = db_name
		self.conn = sqlite3.connect(db_name)
		self.cursor = self.conn.cursor()
		self.create_all_table()

	def create_all_table(self):
		"""Create all necessary tables."""
		self.create_txhash_table()
		self.create_tx_info_table()
		self.create_verify_table()
		self.create_bytecode_table()
		self.create_fundsource_table()

	def create_txhash_table(self):
		"""Create the 'txhash_table' to store contract address and transaction hash."""
		create_table_query = """
            CREATE TABLE IF NOT EXISTS txhash_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                tx_hash TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def create_tx_info_table(self):
		"""Create the 'tx_info_table' to store transaction information."""
		create_table_query = """
            CREATE TABLE IF NOT EXISTS tx_info_table (
                tx_hash TEXT PRIMARY KEY NOT NULL,
                contract_address TEXT NOT NULL,
                msg_sender TEXT NOT NULL,
                value TEXT NOT NULL,
                gasUsed TEXT NOT NULL,
                nonce TEXT NOT NULL,
                input TEXT NOT NULL,
                blockNumber TEXT NOT NULL,
                transactionIndex TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def create_verify_table(self):
		"""Create the 'verify_table' to store contract address and verification tag."""
		create_table_query = """
            CREATE TABLE IF NOT EXISTS verify_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                verify_tag TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def create_bytecode_table(self):
		"""Create the 'bytecode_table' to store contract address and bytecode."""
		create_table_query = """
            CREATE TABLE IF NOT EXISTS bytecode_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                bytecode TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def create_fundsource_table(self):
		"""Create the 'fundsource_table' to store information about fund sources."""
		create_table_query = """
            CREATE TABLE IF NOT EXISTS fundsource_table (
                msg_sender TEXT PRIMARY KEY NOT NULL,
                label TEXT NOT NULL,
                trace_list TEXT NOT NULL,
                hop_count TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def insert_txhash_table(self, data_to_insert):
		"""Insert data into the 'txhash_table'."""
		self.__insert_data("txhash_table", data_to_insert)

	def insert_tx_info_table(self, data_to_insert):
		"""Insert data into the 'tx_info_table'."""
		self.__insert_data("tx_info_table", data_to_insert)

	def insert_verify_table(self, data_to_insert):
		"""Insert data into the 'verify_table'."""
		self.__insert_data("verify_table", data_to_insert)

	def insert_bytecode_table(self, data_to_insert):
		"""Insert data into the 'bytecode_table'."""
		self.__insert_data("bytecode_table", data_to_insert)

	def insert_fundsource_table(self, data_to_insert):
		"""Insert data into the 'fundsource_table'."""
		self.__insert_data("fundsource_table", data_to_insert)

	def query_txhash_table(self, col_list=[]):
		"""Query the 'txhash_table' and return the results."""
		return self.__query_table("txhash_table", col_list)

	def query_tx_info_table(self, col_list=[]):
		"""Query the 'tx_info_table' and return the results."""
		return self.__query_table("tx_info_table", col_list)

	def query_verify_table(self, col_list=[]):
		"""Query the 'verify_table' and return the results."""
		return self.__query_table("verify_table", col_list)

	def query_bytecode_table(self, col_list=[]):
		"""Query the 'bytecode_table' and return the results."""
		return self.__query_table("bytecode_table", col_list)

	def query_fundsource_table(self, col_list=[]):
		"""Query the 'fundsource_table' and return the results."""
		return self.__query_table("fundsource_table", col_list)

	def __insert_data(self, table_name, data_to_insert):
		"""Helper function to insert data into any table."""
		columns = ", ".join(data_to_insert.keys())
		placeholders = ", ".join("?" * len(data_to_insert))
		values = tuple(data_to_insert.values())
		self.cursor.execute(
			f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", values
		)
		self.conn.commit()

	def __query_exec(self, query_string):
		"""Execute a custom SQL query and return the result rows."""
		self.cursor.execute(query_string)
		rows = self.cursor.fetchall()
		return rows

	def __query_table(self, table_name, col_list):
		"""Helper function to query any table."""
		query_string = (
			f"SELECT {', '.join(col_list) if col_list else '*'} FROM {table_name}"
		)
		return self.__query_exec(query_string)

	def __create_table(self, create_table_query):
		"""Create a table using the provided SQL query."""
		self.cursor.execute(create_table_query)
		self.conn.commit()

	def close_connection(self):
		"""Close the database connection."""
		self.conn.close()

	def query_exec(self, query_string):
		"""Execute a custom query."""
		return self.__query_exec(query_string)


def read_db_to_dataframe(db_name, table_name):
	featute_db = FeatureDB(db_name)
	rows = featute_db.query_exec(f"PRAGMA table_info({table_name})")
	column_names = [col[1] for col in rows]

	rows = featute_db.query_table_template(table_name)
	datas = rows
	df = pd.DataFrame(datas, columns=column_names)
	return df


def read_attack_df_sorted(feature_db_file, attack_timestamp_csv):
	attack_bsc_df = read_db_to_dataframe(feature_db_file, "attack_bsc_table")
	attack_ether_df = read_db_to_dataframe(feature_db_file, "attack_ether_table")
	attack_df = pd.concat([attack_bsc_df, attack_ether_df])
	attack_df = attack_df[attack_df["isTokenContract"] == "0"]
	attack_df = attack_df[attack_df["isERC1967Proxy"] == "0"]
	attack_df = attack_df.reset_index(drop=True)

	attack_df_with_timestamp = pd.read_csv(attack_timestamp_csv)
	attack_df_sorted = pd.merge(
		attack_df_with_timestamp, attack_df, on="contract_address", how="inner"
	)
	attack_df_sorted = attack_df_sorted.sort_values(by="timeStamp", ascending=False)
	# Sort with the most recent time first
	attack_df_sorted = attack_df_sorted.reset_index(drop=True)
	attack_df_sorted = attack_df_sorted.drop(columns=["blockNumber", "timeStamp"])
	return attack_df_sorted


def read_benign_df_sorted(feature_db_file, evm_normal_db_file, strip_flag=True):
	normal_df = read_db_to_dataframe(feature_db_file, "benign_table")
	if strip_flag:
		normal_df = normal_df[normal_df["isTokenContract"] == "0"]
		normal_df = normal_df[normal_df["isERC1967Proxy"] == "0"]
		normal_df = normal_df.reset_index(drop=True)

	evm_normal_db = DataDB(evm_normal_db_file)
	rows = evm_normal_db.query_tx_info_table(["contract_address", "blockNumber"])
	evm_normal_with_timestamp = pd.DataFrame(
		rows, columns=["contract_address", "blockNumber"]
	)
	evm_normal_with_timestamp["blockNumber"] = [
		int(item, 16) for item in evm_normal_with_timestamp["blockNumber"].tolist()
	]

	normal_df_sorted = pd.merge(
		evm_normal_with_timestamp, normal_df, on="contract_address", how="inner"
	)
	normal_df_sorted = normal_df_sorted.sort_values(by="blockNumber", ascending=False)
	# Sort with the most recent time first
	normal_df_sorted = normal_df_sorted.reset_index(drop=True)
	normal_df_sorted = normal_df_sorted.drop(columns=["blockNumber"])
	return normal_df_sorted


def getDataSet():
	feature_db_file = f"{DATASET_PATH}/features.db"
	attack_timestamp_csv_file = f"{DATASET_PATH}/attack_timestamp.csv"
	evm_normal_db_file = f"{DATASET_PATH}/evm_data.db"

	attack_df_sorted = read_attack_df_sorted(feature_db_file, attack_timestamp_csv_file)
	normal_df_sorted = read_benign_df_sorted(feature_db_file, evm_normal_db_file)

	attack_df_sorted["tag"] = [1] * len(attack_df_sorted)
	normal_df_sorted["tag"] = [0] * len(normal_df_sorted)

	# Arrange in chronological order from oldest to newest
	attack_df_sorted = attack_df_sorted.iloc[::-1].reset_index(drop=True)
	normal_df_sorted = normal_df_sorted.iloc[::-1].reset_index(drop=True)

	all_df = pd.concat([attack_df_sorted, normal_df_sorted], ignore_index=True)

	return all_df
