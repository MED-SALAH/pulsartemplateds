import dsflow as pulsar
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys

from sklearn.preprocessing import label_binarize

sys.path.append(os.path.abspath(os.curdir))
from {{cookiecutter.project_name}}.train.outils import *

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

    print("f1_score for model categories: ", f1)
    print("accuracy for model categories: ", accuracy)
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
    pulsar.log_param("C", c)
    pulsar.log_param("class_weight", class_weight)
    pulsar.log_param("max_iter", max_iter)
    pulsar.log_metric("accuracy", accuracy)
    pulsar.log_metric("f1", f1)
    pulsar.sklearn.log_model(dic, "model", serialization_format='pickle')

    print("f1_score for model sous categories: ", f1)
    print("accuracy for model sous categories: ", accuracy)


    # Compute ROC curve and ROC area for each class
    y_score = model.predict_proba(X_test)
    classes = model.classes_
    # Binarize the output
    y = label_binarize(y_test, classes=classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fig = plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC curve')
    plt.legend(loc="lower right")

    # Save figures
    fig.savefig("roc-paths.png")
    # Close plot
    plt.close(fig)

    pulsar.log_artifact("roc-paths.png")

if __name__ == '__main__':
    os.path.abspath(os.curdir)
    pulsar.train(name_experiment='{{cookiecutter.project_name}}', train_method=train)

