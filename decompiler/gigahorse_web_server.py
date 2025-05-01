from flask import Flask, request, jsonify
from gigahorse_handler import GigahorseHandler

app = Flask(__name__)


@app.route("/run", methods=["POST"])
def gigahorse_run():
	try:
		data = request.get_json()
		contract_address = data["contract_address"]
		bytecode = data["bytecode"]
		blocknumber = data["blocknumber"]
		print(
			f"Get one request ==> {contract_address} , bytecodelength: {len(bytecode)}"
		)

		gigahorseHandler = GigahorseHandler(contract_address, bytecode, blocknumber)

		if gigahorseHandler.result is None:
			result = {"features": "", "output_text": "", "status": "ERROR"}
		else:
			result = {
				"features": gigahorseHandler.result["features"],
				"output_text": gigahorseHandler.result["output_text"],
				"gigahorse_executetime": gigahorseHandler.result[
					"gigahorse_executetime"
				],
				"codetext_executetime": gigahorseHandler.result["codetext_executetime"],
				"status": "OK",
			}

		return jsonify(result)
	except (TypeError, ValueError):
		return jsonify({"error": "Invalid input. Please provide valid numbers."}), 400


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000)
