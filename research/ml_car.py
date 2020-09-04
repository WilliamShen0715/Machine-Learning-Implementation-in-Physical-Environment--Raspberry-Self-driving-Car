# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:58:57 2019

@author: William
"""
import pandas as pd
import numpy as np
#%% load data & preprocessing
import pickle
data=[];
f=pd.read_csv("label_hand.csv")

x=f.iloc[:,1:4].values
lastx=f.iloc[:-1,1:4].values
y=f.iloc[1:,5].values
x=np.hstack((x[1:],lastx))
# using the "NAN" label to exclude appropriate data
infor =f.iloc[1:,6]
dele=[]
for i in range (1,len(infor)):
    if infor[i]=="NAN": dele.append(i)
y=np.delete(y,dele,axis=0)
x=np.delete(x,dele,axis=0)
#save
a_dict = {'s':x}
file = open('s.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()
#%% call svm
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
x=scale(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=0)

param_grid = {'C': [0.1, 0.5, 1, 5, 10, 15, 20, 100, 150, 175, 200, 250, 1000],
              'gamma': [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 10], }
gclf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
gclf.fit(x_train, y_train.ravel())
best_C=gclf.best_estimator_.C
best_gamma=gclf.best_estimator_.gamma
clf = svm.SVC(gamma=best_gamma, C=best_C)

clf.fit(x_train,y_train.ravel())
y_pdt = clf.predict(x_test)
# check the acc to see how well you've trained the model
acc_svm=accuracy_score(y_pdt,y_test)
#%% call random forest classifier
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',n_estimators=10,n_jobs=1)
forest.fit(x_train, y_train.ravel())
y_pdt2 = forest.predict(x_test)
acc_forest=accuracy_score(y_pdt2,y_test)
#%% call gradient boost dicision tree
from sklearn.ensemble import GradientBoostingClassifier
gb =  GradientBoostingClassifier(n_estimators=100,random_state=0)
gb.fit(x_train, y_train.ravel())
y_pdt3 = gb.predict(x_test)
acc_gb=accuracy_score(y_pdt3,y_test)
#%% transform the trained svm model into C language
from sklearn_porter import Porter
# export:
porter = Porter(clf, language='C')
output = porter.export(embed_data=True)