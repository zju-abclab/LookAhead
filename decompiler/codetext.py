#!/usr/bin/env python3
import os
import sys

from web3 import Web3
from typing import Dict, Mapping, Set, TextIO
from dotenv import load_dotenv

# IT: Ugly hack; this can be avoided if we pull the script at the top level
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import Tuple, List, Union, Mapping, Set
from collections import namedtuple, defaultdict
from itertools import chain

load_dotenv("../.env")

Statement = namedtuple("Statement", ["ident", "op", "operands", "defs"])


class Block:
	def __init__(self, ident: str, statements: List[Statement]):
		self.ident = ident
		self.ident_copy = ident
		self.statements = statements
		self.predecessors: List[Block] = []
		self.successors: List[Block] = []


class Function:
	def __init__(self, ident: str, name: str, head_block: Block, is_public: bool):
		self.ident = ident
		self.name = name
		self.is_public = is_public
		self.head_block = head_block


def load_csv(path: str, seperator: str = "\t") -> List[Union[str, List[str]]]:
	with open(path) as f:
		return [line.split(seperator) for line in f.read().splitlines()]


def load_csv_map(
	path: str, seperator: str = "\t", reverse: bool = False
) -> Mapping[str, str]:
	return (
		{y: x for x, y in load_csv(path, seperator)}
		if reverse
		else {x: y for x, y in load_csv(path, seperator)}
	)


def load_csv_multimap(
	path: str, seperator: str = "\t", reverse: bool = False
) -> Mapping[str, List[str]]:
	ret = defaultdict(list)

	if reverse:
		for y, x in load_csv(path, seperator):
			ret[x].append(y)
	else:
		for x, y in load_csv(path, seperator):
			ret[x].append(y)

	return ret


def load_csv_specific(
	path: str, seperator: str = "\t", first: int = 0, second: int = 1
) -> Mapping[str, str]:
	with open(path) as f:
		return {
			line.split(seperator)[first]: line.split(seperator)[second]
			for line in f.read().splitlines()
		}


def load_target_slot_csv_map(
	path: str, separator: str = "\t"
) -> Dict[str, Tuple[str, int, int]]:
	stmt_to_storage = {}

	with open(path) as f:
		for line in f:
			columns = line.strip().split(separator)
			if len(columns) >= 4:
				stmt_id = columns[0]
				storage_slot = columns[1]
				byte_low = int(columns[2])
				byte_high = int(columns[3])

				stmt_to_storage[stmt_id] = (storage_slot, byte_low, byte_high)

	return stmt_to_storage


def emit(s: str, out: TextIO, indent: int = 0):
	# 4 spaces
	INDENT_BASE = "    "

	print(f"{indent*INDENT_BASE}{s}", file=out)


def remove_empty_blocks(function: Function):
	visited = set()

	def traverse_and_remove(block: Block):
		nonlocal function
		if block.ident in visited:
			return
		visited.add(block.ident)

		for succ in list(block.successors):
			traverse_and_remove(succ)

		if not block.statements:
			for pred in list(block.predecessors):
				pred.successors = [s for s in pred.successors if s != block]
				for succ in block.successors:
					if succ not in pred.successors:
						pred.successors.append(succ)
					succ.predecessors = [p for p in succ.predecessors if p != block]
					if pred not in succ.predecessors:
						succ.predecessors.append(pred)

			if block == function.head_block:
				if block.successors:
					function.head_block = block.successors[0]

			block.predecessors.clear()
			block.successors.clear()

	traverse_and_remove(function.head_block)
	renumber_blocks(function)


def renumber_blocks(function: Function) -> None:
	block_id_mapping = {}
	new_id = 0
	stack = [function.head_block]

	while stack:
		block = stack.pop()

		if block is None or block.ident_copy in block_id_mapping:
			continue

		block_id_mapping[block.ident] = f"BB{new_id}"
		block.ident = f"BB{new_id}"
		new_id += 1
		for succ in block.successors:
			stack.append(succ)


def emit_callprivate(stmt: Statement, out: TextIO):
	target_function = variable2value[stmt.operands[0]]
	emit(functions[target_function].name, out, 2)


def emit_call(stmt: Statement, out: TextIO):
	if stmt.operands[1] in variable2value:  # known const target
		if stmt.ident in stmt2resolvedExternalCall:  # resolved
			emit(
				f"{variable2value[stmt.operands[1]]}.{stmt2resolvedExternalCall[stmt.ident]}",
				out,
				2,
			)
		else:
			emit(f"{variable2value[stmt.operands[1]]}.UnknownExternalCall", out, 2)
	elif stmt.ident in stmt2targetAddress:  # known storage target
		if stmt.ident in stmt2resolvedExternalCall:  # resolved
			emit(
				f"{stmt2targetAddress[stmt.ident]}.{stmt2resolvedExternalCall[stmt.ident]}",
				out,
				2,
			)
		else:
			emit(f"{stmt2targetAddress[stmt.ident]}.UnknownExternalCall", out, 2)
	else:  # unknown target
		if stmt.ident in stmt2resolvedExternalCall:  # resolved
			emit(f"UnknownTarget.{stmt2resolvedExternalCall[stmt.ident]}", out, 2)
		else:  # know nothing
			emit("UnknownTarget.UnknownExternalCall", out, 2)


def pretty_print_block(block: Block, visited: Set[str], out: TextIO):
	emit(f"{block.ident}:", out, 1)

	succ = [s.ident for s in block.successors]

	for stmt in block.statements:
		if stmt.op == "CALLPRIVATE":
			emit_callprivate(stmt, out)
		else:
			emit_call(stmt, out)

	for block in block.successors:
		emit(f"EDGES: {', '.join(succ)}", out, 1)
		if block.ident not in visited:
			visited.add(block.ident)
			pretty_print_block(block, visited, out)


def construct_pruned_cfg() -> Tuple[Mapping[str, Block], Mapping[str, Function]]:
	# Load facts

	# function(also the first block/entry) maps to all blocks in this function
	function2blocks = load_csv_multimap(get_full_path("InFunction.csv"), reverse=True)

	# public function maps to its selector
	publicFunc2selector = load_csv_map(get_full_path("PublicFunction.csv"))

	# confused, but contain some info:
	# public function maps to its signature
	publicFunc2Signature = load_csv_map(get_full_path("HighLevelFunctionName.csv"))

	# function maps to its args and pos
	func2args: Mapping[str, List[Tuple[str, int]]] = defaultdict(list)
	for func_id, arg, pos in load_csv(get_full_path("FormalArgs.csv")):
		func2args[func_id].append((arg, int(pos)))

	# Inverse mapping
	# block belong to which function
	global block2function
	block2function = {}
	# block2function: Mapping[str, str] = {}
	for func_id, block_ids in function2blocks.items():
		for block in block_ids:
			block2function[block] = func_id

	# block maps to all stmts in it
	block2stmts = load_csv_multimap(get_full_path("TAC_Block.csv"), reverse=True)

	# stmt maps to its op
	stmt2op = load_csv_map(get_full_path("TAC_Op.csv"))

	# Load statement defs/uses
	stmt2defs: Mapping[str, List[Tuple[str, int]]] = defaultdict(list)
	for stmt_id, var, pos in load_csv(get_full_path("TAC_Def.csv")):
		stmt2defs[stmt_id].append((var, int(pos)))

	stmt2uses: Mapping[str, List[Tuple[str, int]]] = defaultdict(list)
	for stmt_id, var, pos in load_csv(get_full_path("TAC_Use.csv")):
		stmt2uses[stmt_id].append((var, int(pos)))

	# Load block edges
	block2succ = load_csv_multimap(get_full_path("LocalBlockEdge.csv"))
	block2pred: Mapping[str, List[str]] = defaultdict(list)
	for block, succs in block2succ.items():
		for succ in succs:
			block2pred[succ].append(block)

	def stmt_sort_key(stmt_id: str) -> int:
		return int(stmt_id.replace("S", "").split("0x")[1].split("_")[0], base=16)

	# Construct blocks
	blocks: Mapping[str, Block] = {}
	for block_id in chain(*function2blocks.values()):
		try:
			statements = [
				Statement(
					s_id,
					stmt2op[s_id],
					[var for var, _ in sorted(stmt2uses[s_id], key=lambda x: x[1])],
					[var for var, _ in sorted(stmt2defs[s_id], key=lambda x: x[1])],
				)
				for s_id in sorted(block2stmts[block_id], key=stmt_sort_key)
				if stmt2op[s_id]
				in {"CALL", "STATICCALL", "DELEGATECALL", "CALLPRIVATE"}
			]
			blocks[block_id] = Block(block_id, statements)
		except:
			__import__("pdb").set_trace()

	# Link blocks together
	for block in blocks.values():
		block.predecessors = [blocks[pred] for pred in block2pred[block.ident]]
		block.successors = [blocks[succ] for succ in block2succ[block.ident]]

	functions: Mapping[str, Function] = {}
	for (block_id,) in load_csv(get_full_path("IRFunctionEntry.csv")):
		func_id = block2function[block_id]

		selector = publicFunc2selector.get(func_id, "_")
		if selector == "0x00000000":
			high_level_name = "fallback()"
		elif selector == "0xeeeeeeee":
			high_level_name = "receive()"
		else:
			high_level_name = publicFunc2Signature[func_id]

		functions[func_id] = Function(
			func_id,
			high_level_name,
			blocks[block_id],
			func_id in publicFunc2selector or func_id == "0x0",
		)
	return blocks, functions


def print_codetext(out):
	emit("CONTRACT START", out)
	filtered_functions = [
		f for f in functions.values() if f.name.split("(")[0] != "__function_selector__"
	]
	public_functions = sorted(
		[f for f in filtered_functions if f.is_public], key=lambda x: x.ident
	)
	private_functions = sorted(
		[f for f in filtered_functions if not f.is_public], key=lambda x: x.ident
	)
	for index, function in enumerate(private_functions):
		functions[function.ident].name = f"InternalFunction{index}"

	resolved_functions = [f for f in public_functions if not f.name.startswith("0x")]
	non_resolved_functions = [f for f in public_functions if f.name.startswith("0x")]

	resolved_functions_sorted = sorted(resolved_functions, key=lambda x: x.name)
	non_resolved_functions_sorted = sorted(non_resolved_functions, key=lambda x: x.name)

	for function in resolved_functions_sorted:
		emit(f"FUNCTION {function.name.split('(')[0]} public", out)
		remove_empty_blocks(function)
		pretty_print_block(function.head_block, set(), out)
		emit("FUNCTION END", out, 1)

	for index, function in enumerate(non_resolved_functions_sorted):
		emit(f"FUNCTION UnknownFunction{index} public", out)
		remove_empty_blocks(function)
		pretty_print_block(function.head_block, set(), out)
		emit("FUNCTION END", out, 1)

	for index, function in enumerate(private_functions):
		emit(f"FUNCTION {function.name} private", out)
		remove_empty_blocks(function)
		pretty_print_block(function.head_block, set(), out)
		emit("FUNCTION END", out, 1)

	emit("CONTRACT END", out)


def get_storage_address(stmt2ExternalCallTargetStorage, address, chain, block_number):
	stmt2targetAddress = {}
	slot_cache = {}

	for stmt_id, (
		slot_index,
		byteLow,
		byteHigh,
	) in stmt2ExternalCallTargetStorage.items():
		if slot_index in slot_cache:
			target = slot_cache[slot_index]
		else:
			w3 = w3eth if chain == "eth" else w3bsc
			target = w3.eth.get_storage_at(
				w3.to_checksum_address(address),
				slot_index,
				block_identifier=block_number,
			).hex()
			# print(target,address,slot_index,block_number)
			target = target.replace("0x", "")
			if byteLow == 0:
				target = "0x" + target[-(byteHigh + 1) * 2 :]
			else:
				target = "0x" + target[-(byteHigh + 1) * 2 : -byteLow * 2]
			slot_cache[slot_index] = target
		if int(target, 16) != 0:
			stmt2targetAddress[stmt_id] = target
	return stmt2targetAddress


file = sys.argv[1]
blocknumber = int(sys.argv[2])
base_path = sys.argv[3]
base_path = f"{base_path}/{file}/out/"


def get_full_path(filename: str) -> str:
	return os.path.join(base_path, filename)


def main():
	if not os.path.exists(get_full_path("TAC_Variable_Value.csv")):
		print("TAC_Variable_Value.csv not found")
		exit(0)
	global variable2value
	variable2value = load_csv_map(get_full_path("TAC_Variable_Value.csv"))

	global stmt2resolvedExternalCall
	stmt2resolvedExternalCall = load_csv_specific(
		get_full_path("ExternalCallResolved.csv"), "\t", 1, 3
	)
	stmt2resolvedExternalCall = {
		stmt: func.split("(")[0] for stmt, func in stmt2resolvedExternalCall.items()
	}
	stmt2ExternalCallTargetStorage = load_target_slot_csv_map(
		get_full_path("ExternalCallTargetStorage.csv")
	)
	global stmt2targetAddress
	if len(stmt2ExternalCallTargetStorage) != 0:
		stmt2targetAddress = get_storage_address(
			stmt2ExternalCallTargetStorage, file, "eth", blocknumber
		)
	else:
		stmt2targetAddress = {}

	global functions
	(
		_,
		functions,
	) = construct_pruned_cfg()

	with open(f"{base_path}/codetext.tac", "w") as f:
		print_codetext(f)
	exit(111)


w3bsc_key = os.getenv("BSC_QUICKNODE_APIKEY")
w3eth_key = os.getenv("ALCHEMY_APIKEY")
w3bsc = Web3(
	Web3.HTTPProvider(f"https://fluent-late-replica.bsc.quiknode.pro/{w3bsc_key}")
)
w3eth = Web3(Web3.HTTPProvider(f"https://eth-mainnet.g.alchemy.com/v2/{w3eth_key}"))
if __name__ == "__main__":
	main()
