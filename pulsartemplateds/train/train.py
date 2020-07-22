import dsflow
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath(os.curdir))
from pulsartemplateds.train.outil import *

def train():
    file_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/operation_bancaire.csv')
    df_t = pd.read_csv(file_csv, sep=',')
    df = df_t[['lib_transaction', 'credit', 'debit', 'lib_categorie']]
    df = df.dropna(subset=['lib_categorie'])
    print(df)

    # typecolumns

    df = typecolumns(df)

    # train test split
    y = df['lib_categorie']
    x = df[['lib_transaction', 'credit_o_n']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    Y = y_train.values
    del x

    dic = {}
    print("__________________________________train categorie ____________________________________")

    model = MyPipeline(class_weight='balanced', C=50, max_iter=100)
    model.fit(X_train, Y)
    dic["model_catego"] = model

    # calculate f1
    y_preds = model.predict(X_test)
    f1 = f1_score(y_test, y_preds, average='weighted')
    accuracy = accuracy_score(y_test, y_preds)
    y_preds = model.predict(X_test)
    f1 = f1_score(y_test, y_preds, average='weighted')
    accuracy = accuracy_score(y_test, y_preds)
    print(f1)
    print(accuracy)

    print("_______________________________train sous categorie_________________________")

    df = df_t[['lib_transaction', 'credit', 'debit', 'lib_categorie', 'lib_sous_categorie']]
    df = df.dropna(subset=['lib_sous_categorie'])

    # typecolumns
    df = typecolumns(df)

    # train test split
    y = df['lib_sous_categorie']
    x = df[['lib_transaction', 'credit_o_n', 'lib_categorie']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    Y = y_train.values
    del x
    # param
    class_weight = "balanced"
    c = 10
    max_iter = 200

    model = MyPipeline_sous_categorie(class_weight=class_weight, C=c, max_iter=max_iter)
    model.fit(X_train, Y)
    dic["model_sous_catego"] = model

    # calculate f1
    y_preds = model.predict(X_test)
    f1 = f1_score(y_test, y_preds, average='weighted')
    accuracy = accuracy_score(y_test, y_preds)
    dsflow.log_param("C", c)
    dsflow.log_param("class_weight", class_weight)
    dsflow.log_param("max_iter", max_iter)
    dsflow.log_metric("accuracy", accuracy)
    dsflow.log_metric("f1", f1)
    dsflow.sklearn.log_model(dic, "model", serialization_format='pickle')

    print(f1)
    print(accuracy)


if __name__ == '__main__':
    os.path.abspath(os.curdir)
    dsflow.train(name_experiment='pulsar_template_ds', train_method=train)

