"""
Trains linear regression model.

This script loads data, trains linear regression model and saves it to .pkl file (by default to 'model.pkl').
By default, it expects the data to be in 'train.csv'.

Steps
-----
- parsing command-line arguments
- loading train data
- preprocessing data
- training LinearRegression on preprocessed data
- saving model as .pkl

Preprocessing
-------------
The script preprocesses the data by adding a new variable, `var6_power2`, which is the square of the absolute
value of column '6' in the input data.


Command-Line Arguments
----------------------
The script accepts two optional arguments:
1. `--train-file`: Path to the CSV file containing the training data. Default is 'train.csv'.
2. `--model-file`: Path to save the trained model. Default is 'model.pkl'.

Example
-------
$ python train.py
or
$ python train.py --train-file custom_train_data.csv --model-file custom_model.pkl
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import argparse


def preprocess(data):
    """Add variable var6_power2 = abs(var6)**2."""
    data_prep = data\
        .assign(var6_power2=lambda df_: np.power(np.abs(df_['6']), 2))
    return data_prep


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a linear regression model.')
    parser.add_argument('--train-file', type=str, default='train.csv', help='Path to the training data file')
    parser.add_argument('--model-file', type=str, default='model.pkl', help='Path to save the trained model')
    return parser.parse_args()


def check_linear_model_weights(model, features_data):
    """Display model weights and intercept."""
    features = pd.DataFrame({
        'Variable': features_data.columns,
        'weight': model.coef_
    })

    features = pd.concat([
        features,
        pd.DataFrame({'Variable': 'intercept', 'weight': model.intercept_},
            index=[len(features)]
        )
    ])
    print(features.sort_values('weight', ascending=False))


def main():
    """Run all."""
    args = parse_arguments()

    # Load the training data
    df_train = pd.read_csv(args.train_file)
    X_train, y_train = df_train.drop(columns=['target']), df_train.target
    X_train_prep = preprocess(X_train)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_prep, y_train)
    print('Model is trained')

    # Check trained model's weights
    check_linear_model_weights(model, X_train_prep)

    # Save the trained model
    with open(args.model_file, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model is saved to {args.model_file}')


if __name__ == '__main__':
    main()
