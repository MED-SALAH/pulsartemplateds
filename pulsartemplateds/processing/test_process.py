import numpy as np
from mlflow.pyfunc import load_pyfunc
import pandas as pd


def typecolumns(df):
    df.debit[df.debit == ""] = np.NaN
    df.credit[df.credit == ""] = np.NaN
    df['debit'] = df.debit.astype(float)
    df['credit'] = df.credit.astype(float)
    df['credit_o_n'] = df.credit.isnull()
    df['lib_transaction'] = df.lib_transaction.astype(str)
    return df


def read_data(data):
    df = pd.DataFrame.from_dict(data, orient='columns')
    # df.columns = ['credit', 'date', 'debit', 'libilisation']
    # typecolumns
    df = typecolumns(df)
    return df


data = [{
    "id_operation": 199,
    "lib_transaction": "PRLV SEPA SPB DEB-SPB-876-20181228-000026657097 ASSURANCE MOBILE SPB BOUYGUES TELECOM",
    "credit": "",
    "debit": -13.99
}]

import pandas as pd

data = pd.read_csv("../data/operation_bancaire.csv")
df_to_score = read_data(data)

import pandas as pd

data = pd.read_csv("../data/operation_bancaire.csv")

loaded_model = load_pyfunc('/BigApps/DS/Indus/Models/mlruns/1/b16360d615164921811ac30d313cb02a/artifacts/model')

predtect_categorie = loaded_model["model_catego"].predict(df_to_score)

# add predict categorie to df
df_to_score["lib_categorie"] = predtect_categorie

# predict under category
pred_sous_categorie = loaded_model["model_sous_catego"].predict(df_to_score)

# add predict under category to df
df_to_score["lib_sous_categorie"] = pred_sous_categorie
df_to_score = df_to_score.drop(['credit_o_n'], axis='columns')
print(df_to_score.head())
