import os
import time
import json
import requests

from dotenv import load_dotenv
from fund_source_handler import FundsourceHandler
from contract_label_handler import ContractLabelHandler

load_dotenv("../../.env")

# List of all possible features to be collected
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

# Specific features for Gigahorse analysis
GIGAHORSE_FEATURE_NAME = [
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

# Constants for APIs
ETHERSCAN_URL = "https://api.etherscan.io/api"
ETHERSCAN_APIKEY = os.getenv("ETHERSCAN_APIKEY")  # Etherscan API key
WEB3_PROVIDER_URL = "https://eth-mainnet.g.alchemy.com/v2/"
ALCHEMY_APIKEY = os.getenv("ALCHEMY_APIKEY")  # Alchemy API key
GIGAHORSE_WEBSERVER_URL = "http://127.0.0.1:5000/run"  # Gigahorse server URL


# Helper functions to interact with Etherscan and Web3 API
def getTxhashByContractAddress(one_session, contract_address):
	"""
	Get transaction hash associated with a given contract address.
	"""
	params = {
		"module": "contract",
		"action": "getcontractcreation",
		"contractaddresses": contract_address,
		"apikey": ETHERSCAN_APIKEY,
	}
	resp = one_session.get(ETHERSCAN_URL, params=params, timeout=5)
	return json.loads(resp.text)["result"][0]["txHash"]


def getVerifyTag(one_session, contract_address):
	"""
	Check if the contract has a verified ABI using Etherscan.
	"""
	params = {
		"module": "contract",
		"action": "getabi",
		"address": contract_address,
		"apikey": ETHERSCAN_APIKEY,
	}
	resp = one_session.get(ETHERSCAN_URL, params=params, timeout=5)
	data_json = json.loads(resp.text)
	return data_json["status"] == "1"


def getTxInfo(one_session, txhash):
	"""
	Get detailed transaction information using the transaction hash.
	"""
	data = {
		"jsonrpc": "2.0",
		"method": "eth_getTransactionByHash",
		"params": [txhash],
		"id": 1,
	}
	headers = {"Content-Type": "application/json"}
	resp = one_session.post(
		f"{WEB3_PROVIDER_URL}{ALCHEMY_APIKEY}", data=json.dumps(data), headers=headers
	)
	return json.loads(resp.text)["result"]


def getTxReceipt(one_session, txhash):
	"""
	Get the transaction receipt (including gas used) using the transaction hash.
	"""
	data = {
		"jsonrpc": "2.0",
		"method": "eth_getTransactionReceipt",
		"params": [txhash],
		"id": 1,
	}
	headers = {"Content-Type": "application/json"}
	resp = one_session.post(
		f"{WEB3_PROVIDER_URL}{ALCHEMY_APIKEY}", data=json.dumps(data), headers=headers
	)
	return json.loads(resp.text)["result"]


def getCode(one_session, contract_address, block_number):
	"""
	Get the contract bytecode at a specific block number.
	"""
	payload = {
		"id": 1,
		"jsonrpc": "2.0",
		"params": [contract_address, block_number],
		"method": "eth_getCode",
	}
	headers = {"accept": "application/json", "content-type": "application/json"}
	response = one_session.post(
		f"{WEB3_PROVIDER_URL}{ALCHEMY_APIKEY}", json=payload, headers=headers
	)
	return json.loads(response.text)["result"]


# Class to handle contract feature extraction
class ContractFeatureHandler:
	def debugPrint(self, *args, **kwargs):
		"""Print debug information if debug flag is enabled."""
		if self.debug_flag:
			print(*args, **kwargs)

	def getTxHashInfo(self):
		"""Fetch and store the transaction hash for the contract."""
		self.txHash = getTxhashByContractAddress(
			self.one_session, self.contract_address
		)

	def getVerifyTagInfo(self):
		"""Fetch and store the contract's verification tag."""
		self.features["verify_tag"] = getVerifyTag(
			self.one_session, self.contract_address
		)

	def getTxInfo(self):
		"""Fetch and store transaction details such as value, gas used, nonce, etc."""
		txInfo = getTxInfo(self.one_session, self.txHash)
		self.msg_sender = txInfo["from"].lower()
		self.features["value"] = txInfo["value"]
		self.features["gasUsed"] = txInfo["gas"]
		self.features["nonce"] = txInfo["nonce"]
		self.features["txDataLen"] = len(txInfo["input"])
		self.blocknumber = txInfo["blockNumber"]

	def getBytecodeInfo(self):
		"""Fetch and store the bytecode length of the contract."""
		self.bytecode = getCode(
			self.one_session, self.contract_address, self.blocknumber
		)
		self.features["contract_bytecodeLen"] = len(self.bytecode)

	def getGigahorseInfo(self):
		"""Fetch and store features extracted by Gigahorse analysis tool."""
		data = {
			"contract_address": self.contract_address,
			"bytecode": self.bytecode,
			"blocknumber": int(self.blocknumber, 16),
		}
		response = requests.post(GIGAHORSE_WEBSERVER_URL, json=data, timeout=70)
		self.gigahorse_features = response.json()["features"].split("\t")
		self.gigahorse_ori_tac = response.json()["output_text"]
		self.gigahorse_status = response.json()["status"]
		self.gighorseWebResponse = response.json()
		self.output_text_replaced = ContractLabelHandler(
			self.gigahorse_ori_tac, ETHERSCAN_APIKEY, self.one_session
		).output_text_replaced

		self.features["output_text_replaced"] = self.output_text_replaced

		# Map Gigahorse features to the feature names
		for idx, item in enumerate(self.gigahorse_features):
			self.features[GIGAHORSE_FEATURE_NAME[idx]] = item

	def getFundSourceInfo(self):
		"""Fetch and store the source of funds information."""
		fundsource_handler = FundsourceHandler(
			self.msg_sender, ETHERSCAN_APIKEY, self.one_session
		)
		self.features["fund_from_label"] = fundsource_handler.label_result

	def getGasUsedInfo(self):
		"""Fetch and store gas usage information."""
		txReceipt = getTxReceipt(self.one_session, self.txHash)
		self.features["gasUsed"] = txReceipt["gasUsed"]

	def __init__(self, contract_address, debug_flag=False):
		"""
		Initializes the ContractFeatureHandler and fetches the required information about the contract.

		Args:
		    contract_address (str): The address of the contract.
		    debug_flag (bool): Whether to enable debug prints or not.
		"""
		self.debug_flag = debug_flag
		self.contract_address = contract_address
		self.bytecode = None
		self.time_cal_dic = {}

		# Initialize a requests session and set up proxies if needed
		self.one_session = requests.session()
		proxy_params = {"https": "127.0.0.1:17890", "http": "127.0.0.1:17890"}
		self.one_session.proxies = proxy_params

		# Initialize features dictionary
		self.features = {feature_name: None for feature_name in ALL_FEATURE_NAME}
		self.features["contract_address"] = self.contract_address

		# Step-by-step feature extraction with time measurement
		start_time = time.time()
		self.getTxHashInfo()
		self.time_cal_dic["getTxHashInfo"] = time.time() - start_time
		self.debugPrint(
			f'getTxHashInfo done! Execution time: {self.time_cal_dic["getTxHashInfo"]} s'
		)

		# Repeat for other features
		start_time = time.time()
		self.getVerifyTagInfo()
		self.time_cal_dic["getVerifyTagInfo"] = time.time() - start_time
		self.debugPrint(
			f'getVerifyTagInfo done! Execution time: {self.time_cal_dic["getVerifyTagInfo"]} s'
		)

		start_time = time.time()
		self.getTxInfo()
		self.time_cal_dic["getTxInfo"] = time.time() - start_time
		self.debugPrint(
			f'getTxInfo done! Execution time: {self.time_cal_dic["getTxInfo"]} s'
		)

		start_time = time.time()
		self.getBytecodeInfo()
		self.time_cal_dic["getBytecodeInfo"] = time.time() - start_time
		self.debugPrint(
			f'getBytecodeInfo done! Execution time: {self.time_cal_dic["getBytecodeInfo"]} s'
		)

		start_time = time.time()
		self.getGasUsedInfo()
		self.time_cal_dic["getGasUsedInfo"] = time.time() - start_time
		self.debugPrint(
			f'getGasUsedInfo done! Execution time: {self.time_cal_dic["getGasUsedInfo"]} s'
		)

		start_time = time.time()
		self.getFundSourceInfo()
		self.time_cal_dic["getFundSourceInfo"] = time.time() - start_time
		self.debugPrint(
			f'getFundSourceInfo done! Execution time: {self.time_cal_dic["getFundSourceInfo"]} s'
		)

		start_time = time.time()
		self.getGigahorseInfo()
		self.time_cal_dic["getGigahorseInfo"] = time.time() - start_time
		self.debugPrint(
			f'getGigahorseInfo done! Execution time: {self.time_cal_dic["getGigahorseInfo"]} s'
		)

		# Sum up execution time
		total_time = sum(self.time_cal_dic.values())
		print(f"\033[32mTotal execution time: {total_time} s\033[0m")
		self.debugPrint(f"Gigahorse status: {self.gigahorse_status}")


if __name__ == "__main__":
	# Example contract address to test
	contract_address = "0x3a95fffeebdc113820e8b940254637c8477f59ef"
	contract_handler = ContractFeatureHandler(contract_address, True)
	print(contract_handler.features)
