# LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts
This repository contains the source code for **LookAhead**, an effective and efficient ML-based framework for DeFi attack detection *(based on adversarial contracts)*.

<br/>

## Code Structure
### `decompiler`
This folder contains custom-designed plugins for Gigahorse, a binary lifter and analysis framework for Ethereum smart contracts. We introduce various extra functional modules to be integrated with the Gigahorse toolkit.

The core functionality is to launch a web server (`gigahorse_web_server.py`), which is responsible for handling contract data uploads and running Gigahorse tools to perform contract data analysis.

### `model_eval`
This folder contains code for evaluating the classifier and transformer models employed by LookAhead.

We evaluate the datasets, generate prediction results, measure execution time, and assess the performance of the models.

### `online_test`
This folder contains the logic implemented for conducting online testing to measure the real-world performance of the LookAhead system.

- Before performing the detection, we first train local models (logic located under `training`).
- For detection, we use `contract_feature_handler.py` to extract contract features and `classification_evaluator.py` to load the trained models and produce predictions. An example usage is demonstrated in `evaluate.ipynb`.
(logic located under `detection`)

<br/>

## Dataset
Our dataset includes benign and adversarial contracts and their associated features and is available in the format of SQLite database files under `dataset` folder. It also includes a manually curated dataset of address labels that classify the sources of funds.

Benign contracts are collected from Google BigQuery based on unique user interactions during the period from June 1, 2022 to June 30, 2024. The SQL queries used for retrieving addresses and calculating unique user interactions are available under `dataset/sql`.

<br/>

## Usage
LookAhead has been fully tested on Ubuntu 20.04.

### Configure Environment
```bash
conda env create --name lookahead -f environment.yml
conda activate lookahead
```

You will also need to install Gigahorse. For information regarding the Gigahorse project, see: [https://github.com/nevillegrech/gigahorse-toolchain](https://github.com/nevillegrech/gigahorse-toolchain).

Make sure to install the Souffle addons required by Gigahorse:
```bash
# builds all, sets libfunctors.so as a link to libsoufflenum.so
cd decompiler/gigahorse-toolchain/souffle-addon && make WORD_SIZE=$(souffle --version | sed -n 3p | cut -c12,13)
```

### Environment Variables
Create a dotfile named `.env` and fill in the required API keys to be used as environment variables:
```
ETHERSCAN_APIKEY=YOUR_ETHERSCAN_API_KEY
ALCHEMY_APIKEY=YOUR_ALCHEMY_API_KEY
BSC_QUICKNODE_APIKEY=YOUR_BSC_QUICK_NODE_API_KEY
```

### Download Feature Dataset
Due to the file size limit of Git, we make our contract feature dataset files available via an alternative storage service, download them by running the following:
```bash
cd dataset
wget https://static.ouorz.com/features.db
```

### Train and Evaluate Classifiers
We propose the following classifiers and a transformer model to be trained and evaluated:
- XGBoost
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbor

```bash
python -W ignore evaluate_all.py
```

### Run Tests
To run `online_test`, follow these steps:

Train local classifier models (trained models are saved under `online_test/models`)

```bash
cd online_test/training
python -W ignore train.py
```

Start running a Gigahorse decompiler web server in the background:

```bash
cd decompiler
python gigahorse_web_server.py
```

Perform detection tests using the Jupyter Notebook `online_test/detection/evaluate_single_address.ipynb`.

<br/>

## Citation
Preprint available at [https://arxiv.org/abs/2401.07261](https://arxiv.org/abs/2401.07261).

```bibtex
@article{ren2025lookahead,
  title={LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts},
  author={Ren, Shoupeng and He, Lipeng and Tu, Tianyu and Wu, Di and Liu, Jian and Ren, Kui and Chen, Chun},
  journal={Proceedings of the ACM on Software Engineering},
  volume={2},
  number={FSE},
  year={2025},
  publisher={ACM New York, NY}
}
```
