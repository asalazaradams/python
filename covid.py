#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:03:03 2021

@author: elcolegiodesonora
"""
import statistics as st
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt


covid_df =pd.read_csv("/Users/elcolegiodesonora/Documents/ESTADISTICAS MEXICO/COVID/200724COVID19MEXICO.csv")

covid_df.head()
covid_df.describe()
covid_df.shape
list(covid_df.columns.values)

#Conservamos solo casos positivos:
covid_df = covid_df.drop(covid_df[covid_df.RESULTADO==2].index)
covid_df.shape

#Seleccionamos variables independientes de interes
covid = covid_df[['EDAD','SEXO', 'DIABETES', 'OBESIDAD','INTUBADO','TIPO_PACIENTE','FECHA_DEF']]
covid.head()
covid.shape
#REcodificamos variables:
covid=pd.get_dummies(covid, columns=["SEXO","DIABETES","OBESIDAD","TIPO_PACIENTE", "INTUBADO","FECHA_DEF"])
covid.shape
covid.head()

#Conservamos solo una dummy por variable
covid = covid[['EDAD','SEXO_1', 'DIABETES_1','OBESIDAD_1','INTUBADO_1','TIPO_PACIENTE_1','FECHA_DEF_9999-99-99']]
covid=pd.DataFrame(covid)
covid.head()
covid.mean()
covid.shape

#SELECCIONAMOS VARIABLE DEPENDIENDTE
covid = covid.rename(columns = {'FECHA_DEF_9999-99-99': 'DEFUN'})
covid.min()
covid.max()
covid.mean()
covid['DEFUN'] = 1-covid['DEFUN']
covid.shape
covid.mean()
covid.sum()

#PREPARA TRAIN y TEST SETs

X = np.asarray(covid[['EDAD','SEXO_1', 'DIABETES_1','OBESIDAD_1','TIPO_PACIENTE_1','INTUBADO_1']])
X[0:5]

y = np.asarray(covid['DEFUN'])
y [0:5]

#from sklearn import preprocessing
#X = preprocessing.StandardScaler().fit(X).transform(X)
#X[0:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#AJUSTA MODELO LOGIT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

#PREDICE VALORES DEL MODELO SOBRE LOS VALORES DE X DEL TEST SET
yhat = LR.predict(X_test)
yhat

yhat_prob = LR.predict_proba(X_test)
yhat_prob

#EVALUACION
from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat,pos_label=0)


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['DEF=1','DEF=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


#SVM







