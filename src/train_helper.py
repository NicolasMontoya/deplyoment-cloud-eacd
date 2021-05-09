import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

def validate_data(dataset):
  if isinstance(dataset, pd.DataFrame):
    if set(
      [
        'age', 'sex', 'cp', 'trtbps','chol','fbs',
        'restecg','thalachh','exng','oldpeak','slp',
        'caa','thall','output'
      ]).issubset(dataset.columns):
      return dataset.drop_duplicates()
    else:
      raise ValueError("Wrong dataset")
  else:
    raise ValueError("New parameter should be a pandas dataframe")

def split_data(dataset):
  X = dataset.drop("output", axis=1)
  y = dataset["output"]
  return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, save=False, solver="liblinear", random_state=42):
  log_model = LogisticRegression(solver=solver, random_state=random_state)
  log_model.fit(X_train, y_train)
  if save:
    save_model(log_model, 'outputs/LogisticRegression.pkl')
  return log_model

def save_model(model, save_path='outputs/model.pkl'):
  with open(save_path, 'wb') as file:  
    pickle.dump(model, file)



