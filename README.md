# Task 2
As part of this task I:
- created Python 3.9.13 venv for such task, venv was used both for analysis notebooks and `train.py` / `predict.py` testing.
- analysed provided dataset `train.csv` (process of analysis in `1.EDA.ipynb` and `2 Model selection.ipynb`)
- found out that target column is generated as `target = abs(var6)**2 + var7`, where var6 and var7 - columns '6' and '7' respectively.
- prepared files `train.py` to recreate linear regression model training and `predict.py` for model inference
- generated predictions for `hidden_test.csv` dataset - `predictions.csv`

## Files in repo
- `1.EDA.ipynb` - general analysis of `train.csv`
- `2 Model selection.ipynb` - additional research to find model that is the most effective in describing relationship target ~ data and see what info about this relationship I can extract from it
- `train.py` - script for model training and saving
- `predict.py` - script for generating predictions from saved model
- `.gitignore` - git exceptions
- `README.md` - this README
- `requirements.txt` - requirements for venv recreation
- `predictions.csv` - predictions for `hidden_test.csv`

## Usage
All elements (both notebooks and scripts) where created and tested in Python 3.9.13 venv with requirements as provided in `requirements.txt`.
### Training
To train model run in terminal:
```bash
$ python train.py
```

The script accepts two optional arguments:
1. `--train-file`: Path to the CSV file containing the training data. Default is `train.csv`.
2. `--model-file`: Path to save the trained model. Default is `model.pkl`.

so if there is need to set dataset other than `train.csv` and/or model's file name other than `model.pkl`, use this command:
```bash
$ python train.py --train-file custom_train_data.csv --model-file custom_model.pkl
```
### Prediction
To generate predictions from saved model run in terminal:
```bash
$ python predict.py
```

The script accepts three optional arguments:
1. `--model-file`: Path to the pre-trained model file. Default is `model.pkl`.
2. `--test-file`: Path to the CSV file containing the test data. Default is `hidden_test.csv`.
3. `--output-file`: Path to save the prediction results as a CSV file. Default is `predictions.csv`.

so if there is need to set dataset other than `hidden_test.csv`, and/or model's file name other than `model.pkl`, and/or predictions file name other than `predictions.csv` use this command:
```bash
$ python predict.py --model-file custom_model.pkl --test-file custom_test_data.csv --output-file custom_predictions.csv
```