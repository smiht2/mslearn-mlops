# Import libraries

import argparse
import glob
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.start_run()
    mlflow.sklearn.autolog()

    # mlflow.autolog()
    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df, ratio=args.train_test_ratio)

    # train model
    model = train_model(args.reg_rate, X_train, y_train)

    # test model
    result = test_model(model, X_test, y_test)
    # register and save the model
    reg_save_model(
        model, reg=True, save=False, registered_model_name=args.registered_model_name
    )

    # stop logging
    mlflow.end_run()


def reg_save_model(model, reg=False, save=False, registered_model_name="no_name"):
    # Registering the model to the workspace
    ##########################
    # <save and register model>
    ##########################
    if reg:
        print("Registering the model via MLFlow")
        mlflow.sklearn.log_model(
            sk_model=model,
            registered_model_name=registered_model_name,
            artifact_path=registered_model_name,
        )

    # Saving the model to a file
    if save:
        mlflow.sklearn.save_model(
            sk_model=model,
            path=os.path.join(registered_model_name, "trained_model"),
        )
    ###########################
    # </save and register model>
    ###########################


def get_csvs_df(path):
    # path = '/home/azureuser/cloudfiles/code/Users/smibrahimhossain/mslearn-mlops/experimentation/data/'
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    # csv_files = path
    # print(csv_files)
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df, ratio):
    X, y = (
        df[
            [
                "Pregnancies",
                "PlasmaGlucose",
                "DiastolicBloodPressure",
                "TricepsThickness",
                "SerumInsulin",
                "BMI",
                "DiabetesPedigree",
                "Age",
            ]
        ].values,
        df["Diabetic"].values,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=0
    )
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, y_train):
    # train model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear").fit(X_train, y_train)
    return model


def test_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])

    return acc, auc


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument("--reg_rate", dest="reg_rate", type=float, default=0.01)
    parser.add_argument(
        "--registered_model_name",
        dest="registered_model_name",
        type=str,
        default="default-model",
    )
    parser.add_argument(
        "--train_test_ratio", dest="train_test_ratio", type=float, default=0.20
    )

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
