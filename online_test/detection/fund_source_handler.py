import os
import json
import requests
import pandas as pd

from dotenv import load_dotenv

load_dotenv("../../.env")


def getFundFromLabelDic(csvfile):
	"""
	Reads a CSV file containing fund source labels and returns a dictionary.
	The dictionary keys are fund source labels, and values are lists of addresses associated with those labels.
	All addresses are converted to lowercase.
	"""
	data_frame = pd.read_csv(csvfile)
	fund_from_label_dic = data_frame.to_dict(orient="list")
	for key in list(fund_from_label_dic.keys()):
		fund_from_label_dic[key] = [item.lower() for item in fund_from_label_dic[key]]
	return fund_from_label_dic


# Load the fund source labels from a CSV file
fund_from_label_csv_file = r"../../dataset/address_class_label.csv"
FUND_FROM_LABEL_DIC = getFundFromLabelDic(fund_from_label_csv_file)


class FundsourceHandler:
	def __init__(self, msg_sender, api_key="", one_session=None):
		"""
		Initializes the handler with the sender address, API key, and the requests session.
		Also, it fetches the fund source label and traces the fund source.
		"""
		self.api_key = api_key
		self.one_session = one_session
		self.t_array, self.label_result = self.getFundSourceLabel(msg_sender)
		hop_count = len(self.t_array)
		pass

	def getInternalTransByAddress(
		self, address, sort_flag="asc", page_flag=1, offset_flag=10
	):
		"""
		Fetches internal transactions for the specified address from the Etherscan API.
		"""
		params = {
			"module": "account",
			"action": "txlistinternal",
			"address": address,
			"startblock": 0,
			"endblock": 99999999,
			"page": page_flag,
			"offset": offset_flag,
			"sort": sort_flag,
			"apikey": self.api_key,
		}
		try:
			resp = self.one_session.get(
				url="https://api.etherscan.io/api", params=params, timeout=10
			)
			resp.encoding = "utf-8"
			return resp.text
		except requests.exceptions.RequestException as e:
			print(f"Request failed: {e}")
			return ""

	def getNormalTransByAddress(
		self, address, sort_flag="asc", page_flag=1, offset_flag=10
	):
		"""
		Fetches normal (external) transactions for the specified address from the Etherscan API.
		"""
		params = {
			"module": "account",
			"action": "txlist",
			"address": address,
			"startblock": 0,
			"endblock": 99999999,
			"page": page_flag,
			"offset": offset_flag,
			"sort": sort_flag,
			"apikey": self.api_key,
		}
		try:
			resp = self.one_session.get(
				url="https://api.etherscan.io/api", params=params, timeout=10
			)
			resp.encoding = "utf-8"
			return resp.text
		except requests.exceptions.RequestException as e:
			print(f"Request failed: {e}")
			return ""

	def getFirstFund(self, address):
		"""
		Fetches the first fund source address for a given address by querying both normal and internal transactions.
		It checks if the transaction value is greater than zero and returns the address that sent the funds.
		"""
		first_normal_transaction = json.loads(
			self.getNormalTransByAddress(address, "asc", 1, 1)
		)
		first_internal_transaction = json.loads(
			self.getInternalTransByAddress(address, "asc", 1, 1)
		)

		status1 = first_normal_transaction.get("message") in [
			"OK",
			"No transactions found",
		]
		status2 = first_internal_transaction.get("message") in [
			"OK",
			"No transactions found",
		]

		# Handle API response errors
		if not status1 or not status2:
			raise ValueError(
				f"Error in fetching transaction data: {first_normal_transaction}, {first_internal_transaction}"
			)

		status1 = False
		status2 = False
		# Check if the normal transaction has an incoming fund
		if (
			first_normal_transaction["status"] == "1"
			and first_normal_transaction["result"][0]["to"].lower() == address.lower()
			and int(first_normal_transaction["result"][0]["value"]) > 0
		):
			status1 = True
		# Check if the internal transaction has an incoming fund
		if (
			first_internal_transaction["status"] == "1"
			and first_internal_transaction["result"][0]["to"].lower() == address.lower()
			and int(first_internal_transaction["result"][0]["value"]) > 0
		):
			status2 = True

		# If both transactions are valid, return the sender address with the smallest block number
		if status1 and status2:
			block_num1 = int(first_normal_transaction["result"][0]["blockNumber"])
			block_num2 = int(first_internal_transaction["result"][0]["blockNumber"])
			if block_num1 < block_num2:
				return first_normal_transaction["result"][0]["from"]
			else:
				return first_internal_transaction["result"][0]["from"]
		# Return the sender address from the first valid transaction
		elif status1:
			return first_normal_transaction["result"][0]["from"]
		elif status2:
			return first_internal_transaction["result"][0]["from"]
		# If no valid transaction found, return '1' indicating an error or unknown source
		else:
			return "1"

	def getFundSourceLabel(self, address):
		"""
		Traces the fund source up to 5 hops to find the label of the fund source.
		If no known label is found, it returns 'unknown_low' or 'unknown_high' based on the transaction history.
		"""
		trace_array = []
		max_hop = 5
		while max_hop > 0:
			fund_source_addr = self.getFirstFund(address)
			# If the fund source is 'GENESIS' (initial block), return 'Safe'
			if fund_source_addr == "GENESIS":
				return trace_array, "Safe"
			# If no transaction found, return 'unknown_contract'
			if fund_source_addr == "1":
				return trace_array, "unknown_contract"

			trace_array.append(fund_source_addr)

			# Check if the fund source address matches any known label
			for key in FUND_FROM_LABEL_DIC.keys():
				if fund_source_addr.lower() in FUND_FROM_LABEL_DIC[key]:
					return trace_array, key

			# Fetch internal and normal transaction data for the fund source
			internal_response = json.loads(
				self.getInternalTransByAddress(fund_source_addr, "asc", 1, 10000)
			)
			normal_response = json.loads(
				self.getNormalTransByAddress(fund_source_addr, "asc", 1, 10000)
			)
			history_trans_num = len(internal_response["result"]) + len(
				normal_response["result"]
			)

			# If the fund source has more than 10,000 transactions, label it as 'unknown_high'
			if history_trans_num >= 10000:
				return trace_array, "unknown_high"

			# Reduce the hop count and continue tracing the fund source
			max_hop -= 1
			if max_hop == 0:
				return trace_array, "unknown_low"

			address = fund_source_addr


if __name__ == "__main__":
	# Example Ethereum address (can be replaced with any address)
	msg_sender = "0x946e9c780f3c79d80e51e68d259d0d7e794f2124"
	etherscan_apikey = os.getenv("ETHERSCAN_APIKEY")
	one_session = requests.session()

	# Initialize the FundsourceHandler with the address and API key
	fundsource_handler = FundsourceHandler(msg_sender, etherscan_apikey, one_session)

	# Print the trace array and fund source label
	print(fundsource_handler.t_array)
	print(fundsource_handler.label_result)
