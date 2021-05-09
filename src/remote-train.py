import os
import pandas as pd
from sklearn.metrics import accuracy_score
from train_helper import validate_data, split_data, train_model
from azureml.core import Run, Dataset


if __name__ == "__main__":
    run = Run.get_context()
    ws = run.experiment.workspace

    datastore = ws.get_default_datastore()
    input_ds = Dataset.get_by_name(ws, 'cardio_ds')
    data = input_ds.to_pandas_dataframe()

    dataframe = validate_data(data)
    X_train, X_test, y_train, y_test = split_data(dataframe)
    model = train_model(X_train, y_train, save=True)
    y_pred = model.predict(X_test)
    print(f"Accurancy: {accuracy_score(y_test, y_pred)}")
    run.log('accurancy', accuracy_score(y_test, y_pred))