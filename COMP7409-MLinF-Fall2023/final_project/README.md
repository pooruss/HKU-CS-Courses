# Framework for ML in Finance


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