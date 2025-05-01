import requests
import json
import re
import os

from dotenv import load_dotenv

load_dotenv("../../.env")


class ContractLabelHandler:
	def __init__(self, gigahorse_ori_tac, api_key="", one_session=None):
		self.api_key = api_key
		self.one_session = one_session
		self.output_text_replaced = self.replace_tac_text(gigahorse_ori_tac)

	def isToken(self, abi_string):
		abi_json = json.loads(abi_string)
		function_list = []
		for item in abi_json:
			if item["type"] == "function":
				function_list.append(item["name"])
		if "balanceOf" in function_list and "transfer" in function_list:
			return True
		else:
			return False

	def replace_addresses(self, match):
		address = match.group(0)
		address = address.lower()

		info = self.get_contract_info(address)

		if not info or info.get("ABI") == "Contract source code not verified":
			return "UnknownTarget"

		if self.isToken(info["ABI"]):
			return "Token"
		elif info["Proxy"] == "1":
			return "Proxy"
		else:
			return info["ContractName"]

	def replace_tac_text(self, tac_text):
		tac_text_replaced = ""
		eth_address_pattern = r"(?i)(?:0x)?[0-9a-f]{40}"
		tac_text_replaced = re.sub(
			eth_address_pattern, self.replace_addresses, tac_text
		)
		return tac_text_replaced

	# TODO: Integrate address labels and tags from blockchain explorers
	def get_contract_info(self, contract_address):
		try:
			params = {
				"module": "contract",
				"action": "getsourcecode",
				"address": contract_address,
				"apikey": self.api_key,
			}
			resp = self.one_session.get(
				url="https://api.etherscan.io/api", params=params, timeout=10
			)
			result = json.loads(resp.text)["result"][0]
			return result
		except Exception as e:
			print(f"Error querying address {contract_address}: {e}")
			return None


if __name__ == "__main__":
	etherscan_apikey = os.getenv("ETHERSCAN_APIKEY")
	one_session = requests.session()
	contract_address = """CONTRACT START
FUNCTION UnknownFunction0 public
    BB0:
        0xafc2f2d803479a2af3a72022d54cc0901a0ec0d6.clone
    EDGES: BB1
    BB1:
        0x9ec251401eafb7e98f37a1d911c0aea02cb63a80.initializeC
        0x359552deec849fd66e72743558d125a6a35991d9.test
    EDGES: BB2
    BB2:
        InternalFunction0
    EDGES: BB0
    BB0:
        0x2026001855172dc690233021e51a3cd1b54aa5ff.clone
        0x000000000000aaeb6d7670e522a718067333cd4e.registry
    EDGES: BB1
    FUNCTION END
"""
	label_handler = ContractLabelHandler(
		contract_address, etherscan_apikey, one_session=one_session
	)
	print(label_handler.output_text_replaced)
