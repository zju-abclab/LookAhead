import pandas as pd
import db_tools as db_tools


def split_dataset(n, m):
	"""
	This function divides a dataset of size 'n' into'm' equal parts and returns the index ranges of each part.

	Args:
	    n (int): The size of the dataset.
	    m (int): The number of divisions.

	Returns:
	    list: A list containing tuples (a, b), representing the start index 'a' and end index 'b' of each division.
	"""
	result = []
	# Calculate the basic size of each part
	base_size = n // m
	# Calculate the number of remaining elements
	remainder = n % m
	start = 0
	for i in range(m):
		# The size of the current part, with an additional element if there are remainders
		current_size = base_size + (1 if i < remainder else 0)
		end = start + current_size
		# Add the index range of the current division as a tuple to the result list
		result.append((start, end))
		start = end
	return result


def read_db_to_dataframe(db_name, table_name):
	featute_db = db_tools.FeatureDB(db_name)
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

	evm_normal_db = db_tools.DataDB(evm_normal_db_file)
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


def split_df(df, split_num):
	n = len(df)
	split_ranges = split_dataset(n, split_num)
	split_dfs = []
	for start, end in split_ranges:
		split_dfs.append(df.iloc[start:end].reset_index(drop=True))
	return split_dfs


def getCrossDataSet():
	feature_db_file = r"../dataset/features.db"
	attack_timestamp_csv_file = r"../dataset/attack_timestamp.csv"
	evm_normal_db_file = r"../dataset/evm_data.db"

	attack_df_sorted = read_attack_df_sorted(feature_db_file, attack_timestamp_csv_file)
	normal_df_sorted = read_benign_df_sorted(feature_db_file, evm_normal_db_file)

	attack_df_sorted["tag"] = [1] * len(attack_df_sorted)
	normal_df_sorted["tag"] = [0] * len(normal_df_sorted)

	# Arrange in chronological order from oldest to newest
	attack_df_sorted = attack_df_sorted.iloc[::-1].reset_index(drop=True)
	normal_df_sorted = normal_df_sorted.iloc[::-1].reset_index(drop=True)

	split_num = 5
	attack_df_sorted_dfs = split_df(attack_df_sorted, split_num)
	normal_df_sorted_dfs = split_df(normal_df_sorted, split_num)

	merge_dfs = []
	for i in range(split_num):
		merge_dfs.append(
			pd.concat(
				[attack_df_sorted_dfs[i], normal_df_sorted_dfs[i]], ignore_index=True
			)
		)

	fold_dataset = [
		{"train": merge_dfs[0], "test": merge_dfs[1]},
		{
			"train": pd.concat([merge_dfs[0], merge_dfs[1]], ignore_index=True),
			"test": merge_dfs[2],
		},
		{
			"train": pd.concat(
				[merge_dfs[0], merge_dfs[1], merge_dfs[2]], ignore_index=True
			),
			"test": merge_dfs[3],
		},
		{
			"train": pd.concat(
				[merge_dfs[0], merge_dfs[1], merge_dfs[2], merge_dfs[3]],
				ignore_index=True,
			),
			"test": merge_dfs[4],
		},
	]

	return fold_dataset
