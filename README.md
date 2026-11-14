# Online transaction fraud detection

Train logistic regression and decision tree classifiers on a labeled transaction CSV. The script performs a stratified train/test split, tunes hyperparameters with three-fold cross-validation using average precision, and reports precision, recall, F1 and confusion matrices on held-out data. It saves the better model by held-out average precision.

## Run

```bash
python -m venv .venv
# Activate the virtual environment, then:
pip install -r requirements.txt
python fraud.py path/to/transactions.csv --target isFraud --output-dir outputs
```

The input may use the PaySim-style `isFraud` label or another binary target supplied with `--target`. Both classes must be present, with at least five rows each; real evaluation needs far more fraud examples. `nameOrig`, `nameDest`, and `isFlaggedFraud` are excluded to reduce identifier leakage and avoid using a preexisting fraud flag. The script writes class balance and confusion matrix PNGs, a comparison CSV, and `model.joblib` under the output directory. Do not load model files from untrusted sources.

No transaction dataset or trained model is included; supply data that you have permission to use.

