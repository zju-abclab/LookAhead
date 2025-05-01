import sqlite3


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


# Record gigahorse output
# gigahourse_output_table
#     - contract_address
#     - output # Statistical features
#     - output_text # Textual features

# Record data where gigahorse run times out
# timeout_bytecode_table
#     - contract_address
#     - bytecode


class GigahorseOutputDB:
	def __init__(self, db_name):
		self.db_name = db_name
		self.conn = sqlite3.connect(db_name)
		self.cursor = self.conn.cursor()
		self.create_all_table()

	def create_all_table(self):
		self.create_gigahorse_output_table()
		self.create_timeout_bytecode_table()

	def create_gigahorse_output_table(self):
		create_table_query = """
            CREATE TABLE IF NOT EXISTS gigahorse_output_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                output TEXT NOT NULL,
                output_text TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def create_timeout_bytecode_table(self):
		create_table_query = """
            CREATE TABLE IF NOT EXISTS timeout_bytecode_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                bytecode TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def insert_data_gigahorse_output_table(self, data_to_insert):
		# data_to_insert = {
		#     'contract_address': contract_addr,
		#     'output': output,
		#     'output_text': output_text
		# }
		self.__insert_data("gigahorse_output_table", data_to_insert)

	def insert_timeout_bytecode_table(self, data_to_insert):
		# data_to_insert = {
		#     'contract_address': contract_addr,
		#     'bytecode': bytecode,
		# }
		self.__insert_data("timeout_bytecode_table", data_to_insert)

	def query_gigahorse_output_table(self, col_list=[]):
		return self.__query_table("gigahorse_output_table", col_list)

	def query_timeout_bytecode_table(self, col_list=[]):
		return self.__query_table("timeout_bytecode_table", col_list)

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


# Record gigahorse output (address replaced)
# tac_replaced_table
#     - contract_address
#     - output # Statistical features
#     - output_text_replaced # Textual features


class GigahorseOutputReplacedDB:
	def __init__(self, db_name):
		self.db_name = db_name
		self.conn = sqlite3.connect(db_name)
		self.cursor = self.conn.cursor()
		self.create_all_table()

	def create_all_table(self):
		self.create_tac_replaced_table()

	def create_tac_replaced_table(self):
		create_table_query = """
            CREATE TABLE IF NOT EXISTS tac_replaced_table (
                contract_address TEXT PRIMARY KEY NOT NULL,
                output TEXT NOT NULL,
                output_text_replaced TEXT NOT NULL
            )
        """
		self.__create_table(create_table_query)

	def insert_data_tac_replaced_table(self, data_to_insert):
		# data_to_insert = {
		#     'contract_address': contract_addr,
		#     'output': output,
		#     'output_text_replaced': output_text_replaced
		# }
		self.__insert_data("tac_replaced_table", data_to_insert)

	def query_tac_replaced_table(self, col_list=[]):
		return self.__query_table("tac_replaced_table", col_list)

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
