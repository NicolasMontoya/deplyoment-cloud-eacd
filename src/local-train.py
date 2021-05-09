import os
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from train_helper import validate_data, split_data, train_model
from azureml.core import Run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default="./data/",
        help='Path to the training data'
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default="heart.csv",
        help='Filename'
    )
    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("FILE NAME: " + args.file_name)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    data = pd.read_csv(args.data_path + args.file_name)
    data = validate_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train, save=True)
    y_pred = model.predict(X_test)
    print(f"Accurancy: {accuracy_score(y_test, y_pred)}")