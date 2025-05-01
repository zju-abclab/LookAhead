import time
import subprocess
import os


class GigahorseHandler:
	def __init__(self, contract_address, bytecode, blocknumber):
		self.dataset_dir = "webcache/upload/"
		self.output_folder_path = "webcache/results/"

		self.contract_address = contract_address
		self.bytecode = bytecode
		self.blocknumber = blocknumber

		self.saveUpload()
		self.run()

	def saveUpload(self):
		with open(self.dataset_dir + self.contract_address + ".hex", "w") as f:
			f.write(self.bytecode)

	def run(self):
		start_time = time.time()

		process = subprocess.Popen(
			f"python3 gigahorse-toolchain/gigahorse.py -C gigahorse-toolchain/clients/features_client.dl {self.dataset_dir + self.contract_address + '.hex'} -w {self.output_folder_path} --restart -T 60",
			shell=True,
		)
		# process = subprocess.Popen(f"python3 gigahorse.py -C clients/features_client.dl {self.dataset_dir + self.contract_address + '.hex'} -w {self.output_folder_path} --restart -T 60 >> /dev/null 2>&1", shell=True)

		process.wait()
		end_time = time.time()
		gigahorse_executetime = end_time - start_time
		print(f"gigahorse execute time {gigahorse_executetime}s")

		feature_file = (
			f"{self.output_folder_path}/{self.contract_address}/out/AllFeature.csv"
		)
		tac_file = f"{self.output_folder_path}/{self.contract_address}/out/codetext.tac"

		if os.path.exists(feature_file):
			start_time = time.time()
			cmd = f"python3 codetext.py {self.contract_address} {self.blocknumber} {self.output_folder_path}"
			process = subprocess.Popen(cmd, shell=True)
			process.wait()
			end_time = time.time()
			codetext_executetime = end_time - start_time
			print(f"codetext execute time {codetext_executetime}s")

		print(
			f"feature_file => {os.path.exists(feature_file)} , tac_file => {os.path.exists(tac_file)}"
		)
		if os.path.exists(feature_file) and os.path.exists(tac_file):
			with open(feature_file, "r") as f:
				features = f.readline().strip()
			with open(tac_file, "r") as f:
				output_text = f.read()
			self.result = {
				"features": features,
				"output_text": output_text,
				"gigahorse_executetime": gigahorse_executetime,
				"codetext_executetime": codetext_executetime,
			}
		else:
			self.result = None
		pass
