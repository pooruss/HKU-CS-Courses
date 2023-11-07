# Framework for ML in Finance

## Project Introduction
### Background
Machine learning needs in the finance field: Machine learning algorithms can be used in many aspects such as risk management, asset management, market analysis and trading strategies, and have become a key tool in the field of finance and trading.

### Goals and Significance
#### Goals
* Develop a custom Python machine learning framework
* Intergrate basic machine learning algorithms implemented through packages such as numpy
* Evaluate the algorithms in a variety of financial scenarios
#### Significance
* Develop a machine learning framework for the financial field to provide solutions more suitable for financial problems instead of just relying on general machine learning libraries
* A framework for machine learning beginners to quickly get started with basic machine learning algorithms

### Main Content
✨Here is an overview of this framework.
<br>
<div align="center">
<img src="assets/overview.jpg" width="800px">
</div>
<br>

### Results and Evaluation


### Future Work
* Develop interactive UI
* Intergrate More algorithms
* Encapsulate the data preprocessing process to reduce the cost of getting started

## Main File Sturcture
```
├── main.py
├── pipeline.py
├── ...
├── /dataset/
│  ├── base.py
│  ├── risk_management.py
│  ├── investment_and_asset_management.py
│  ├── ...
│  └── finance_prediction.py
├── /algorithms/
│  ├── base.py
│  ├── svm.py
│  ├── linear_regression.py
│  ├── ...
│  └── pca.py
├── /evaluate/
│  └── utils.py
├── /visualization/
│  └── utils.py
```

## Setup
- Install
```bash
pip install -i requirements.txt
```

## Run
```bash
bash scripts/run_svm.sh
```
