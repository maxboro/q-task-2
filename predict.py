"""
Linear Regression model inference script.

This script loads a pre-trained linear regression model and makes predictions on new data provided in a CSV file.
By default, it expects the model to be in 'model.pkl' and the test data to be in 'hidden_test.csv'.
It saves the predictions to a CSV file, which is 'predictions.csv' by default.


Steps
-----
- parsing command-line arguments
- loading test data
- loading model from .pkl file
- preprocessing data
- inferencing model on preprocessed data
- saving predictions to .csv


Preprocessing
-------------
The script preprocesses the data by adding a new variable, `var6_power2`, which is the square of the absolute
value of column '6' in the input data.


Command-Line Arguments
----------------------
The script accepts three optional arguments:
1. `--model-file`: Path to the pre-trained model file. Default is 'model.pkl'.
2. `--test-file`: Path to the CSV file containing the test data. Default is 'hidden_test.csv'.
3. `--output-file`: Path to save the prediction results as a CSV file. Default is 'predictions.csv'.


Example
-------
$ python predict.py
or
$ python predict.py --model-file custom_model.pkl --test-file custom_test_data.csv --output-file custom_predictions.csv
"""
import numpy as np
import pandas as pd
import pickle
import argparse


def preprocess(data):
    """Add variable var6_power2 = abs(var6)**2."""
    data_prep = data\
        .assign(var6_power2=lambda df_: np.power(np.abs(df_['6']), 2))
    return data_prep


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions using a trained linear regression model.')
    parser.add_argument('--model-file', type=str, default='model.pkl', help='Path to the saved model file')
    parser.add_argument('--test-file', type=str, default='hidden_test.csv', help='Path to the test data CSV file')
    parser.add_argument('--output-file', type=str, default='predictions.csv', help='Path to save the predictions')
    return parser.parse_args()


def main():
    """Run all."""
    args = parse_arguments()

    # Load the trained model
    with open(args.model_file, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f'Model loaded from {args.model_file}')

    # Load the test data
    X_test = pd.read_csv(args.test_file)
    print(f'Test data loaded from {args.test_file}')

    # Preprocess the test data and make predictions
    predictions = pd.DataFrame(
        loaded_model.predict(preprocess(X_test)),
        columns=['prediction']
    )
    print('Predictions calculated')

    print('Predictions distribution:\n', predictions.describe())

    # Save the predictions to a CSV file
    predictions.to_csv(args.output_file, index=True)
    print(f'Predictions saved to {args.output_file}')

if __name__ == '__main__':
    main()
