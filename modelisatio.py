from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import random as rnd
import scipy as sp
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('/Volumes/ECO/Python/df.csv', sep=',')

col = df.columns
df.drop(columns = ['Unnamed: 0','installment'],inplace=True)

mask = (df['issue_d'] >= '2017-01-01')
df_test = df.loc[mask]

mask2 = (df['issue_d'] < '2017-01-01')
df_train = df.loc[mask2]


var_cate = [col for col in df.columns if df[col].dtype=='O']

for col in var_cate : 
    df.drop(columns = col,inplace=True)
    df_test.drop(columns = col,inplace=True)
    df_train.drop(columns = col,inplace=True)
 

df_test['Target'].value_counts(normalize=True)
df_train['Target'].value_counts(normalize=True)

#C'est bien équilibré ! 

#On isole nos target : 
x_train = df_train.drop(['Target'],axis = 1 ) 
x_test = df_test.drop(['Target'],axis = 1 ) 

y_train = df_train['Target'].values
y_test = df_test['Target'].values

#Nombre de Y=0 dans le test 
y_0_test = df_train['Target'].value_counts(normalize=True)[1]*len(y_test)
y_1_test = df_train['Target'].value_counts(normalize=True)[0]*len(y_test)



#Test Xgboost 
xgb_clf = xgb.XGBClassifier().fit(x_train, y_train)
xgb_clf.score(x_test, y_test)
xgb_pred = xgb_clf.predict(x_test)
print(confusion_matrix(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

#Logistic 1er tunning
pipeline_sgdlogreg = Pipeline([
    ('imputer', SimpleImputer(copy=False)), # Mean imputation by default
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])

param_grid_sgdlogreg = {
    'model__alpha': [10**-5, 10**-2, 10**1],
    'model__penalty': ['l1', 'l2']
}

grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)

grid_sgdlogreg.fit(x_train, y_train)

grid_sgdlogreg.best_score_
grid_sgdlogreg.best_params_

print(grid_sgdlogreg.best_score_, '- Logistic regression')

#K neighbors
pipeline_knn = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('scaler', StandardScaler(copy=False)),
    ('lda', LinearDiscriminantAnalysis()),
    ('model', KNeighborsClassifier(n_jobs=-1))
])

#Tunning
param_grid_knn = {
    'model__n_neighbors': [5, 25, 125] # The 'k' in k-nearest neighbors
}

grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)

grid_knn.fit(x_train, y_train)

grid_knn.best_score_
grid_knn.best_params_
y_pred_KNN = grid_knn.predict(x_test)

#Random Forest
pipeline_rfc = Pipeline([
    ('imputer', SimpleImputer(copy=False)),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])
param_grid_rfc = {
    'model__n_estimators': [50] # The number of randomized trees to build
}
grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)

grid_rfc.fit(x_train, y_train)
grid_rfc.best_score_
y_pred_RF = grid_rfc.predict(x_test)

print(confusion_matrix(y_test, ))
print(grid_rfc.best_score_, '- Random forest')
# Comparaison 

ar = np.array([['RegressionLogistic',grid_sgdlogreg.best_score_],['RandomForest',grid_rfc.best_score_],['K neighbors',grid_knn.best_score_],["Xgb",xgb_clf.score(x_test, y_test)]])
df_graph = pd.DataFrame(ar, columns = ['Model_Name','Best_Score'])
df_graph['Best_Score']= df_graph['Best_Score'].astype(float)
plt.figure(figsize=(17,15))
sns.barplot(x='Model_Name',y='Best_Score', data = df_graph, palette="Set3")

#Lsecond tunning de la logistique 
param_grid_sgdlogreg = {
    'model__alpha': np.logspace(-4.5, 0.5, 11), # Fills in the gaps between 10^-5 and 10^1
    'model__penalty': ['l1', 'l2']
}

print(param_grid_sgdlogreg)
#pip install RandomState
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=1, pre_dispatch=1, cv=5, verbose=1, return_train_score=False)
grid_sgdlogreg.fit(x_train, y_train)
grid_sgdlogreg.best_score_
grid_sgdlogreg.best_params_
y_score = grid_sgdlogreg.predict_proba(x_test)[:,1]
y_pred =grid_sgdlogreg.predict(x_test)
print(y_score)
auc_roc = roc_auc_score(y_test,y_score)

from sklearn.metrics import roc_curve, auc

def plot_roc_auc(actual, preds):
    fpr, tpr, thresholds = roc_curve(actual, preds[:,1])
    plt.plot(fpr, tpr,'r')
    plt.plot([0,1],[0,1],'b')
    plt.title('AUC: {}'.format(auc(fpr,tpr)))
    plt.show()

#LOG
plot_roc_auc(y_test, grid_sgdlogreg.predict_proba(x_test))
print(auc_roc)

#RF
plot_roc_auc(y_test, grid_rfc.predict_proba(x_test))

#Xgboost
plot_roc_auc(y_test, xgb_clf.predict_proba(x_test))

#KNN
plot_roc_auc(y_test, grid_knn.predict_proba(x_test))


import numpy as np
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, y_score))

#LOGIS
confusion_matrix(y_test, y_pred)


#KNN
confusion_matrix(y_test, y_pred_KNN)

#RF
conf_RF = confusion_matrix(y_test, y_pred_RF)
conf_RF.sum()
#Xbgoost
print(confusion_matrix(y_test, xgb_pred))

#Recall
Recall_logist = 24177/(24177+22711)
print(Recall_logist)
Recall_KNN =  21985/(21985+24903)
print(Recall_KNN)
Recall_RF = 23936/(23936+22952)
print(Recall_RF)
Recall_Xgboost = 23943/(23943+22945)
print(Recall_Xgboost)

#precision

pre_logist = 24177/(24177+1453)
print(pre_logist)
pre_KNN =  21985/(21985+8778)
print(pre_KNN)
pre_RF = 23936/(23936+497)
print(pre_RF)
pre_Xgboost = 23943/(23943+595)
print(pre_Xgboost)

#F1-Score
def F1(Recall,precision):
    F1 = (2*(Recall*precision))/(Recall+precision)
    print(F1)
F1(Recall_logist,pre_logist)
F1(Recall_KNN,pre_KNN)
F1(Recall_RF,pre_RF)
F1(Recall_Xgboost,pre_Xgboost)

#F_Beta:
def F1B(Recall,precision,beta):
    F1B = (1+beta**2)*((precision*Recall)/((beta*precision)+Recall))
    print(F1B)

F1B(Recall_logist,pre_logist,1.5)
F1B(Recall_KNN,pre_KNN,1.5)
F1B(Recall_RF,pre_RF,1.5)
F1B(Recall_Xgboost,pre_Xgboost,1.5)
 

#Comparaion
#Recall
ar = np.array([['RegressionLogistic',Recall_logist],['RandomForest',Recall_RF],['K neighbors',Recall_KNN],['Xgboost',Recall_Xgboost]])
df_graph = pd.DataFrame(ar, columns = ['Model_Name','Recall'])
df_graph['Recall']= df_graph['Recall'].astype(float)
plt.figure(figsize=(20,20))
sns.barplot(x='Model_Name',y='Recall', data = df_graph, palette="Set3")

#precision
ar = np.array([['RegressionLogistic',pre_logist],['RandomForest',pre_RF],['K neighbors',pre_KNN],['Xgboost',pre_Xgboost]])
df_graph = pd.DataFrame(ar, columns = ['Model_Name','precision'])
df_graph['precision']= df_graph['precision'].astype(float)
plt.figure(figsize=(20,20))
sns.barplot(x='Model_Name',y='precision', data = df_graph, palette="Set3")

#F1_Score
ar = np.array([['RegressionLogistic',0.8188163022863217],['RandomForest',0.8208937899013348],['K neighbors',0.7067733320144419],['Xgboost',0.8202250447981447]])
df_graph = pd.DataFrame(ar, columns = ['Model_Name','F1'])
df_graph['F1']= df_graph['F1'].astype(float)
plt.figure(figsize=(20,20))
sns.barplot(x='Model_Name',y='F1', data = df_graph, palette="Set3")


#Ramdom forest >>>


#calcul profit du modèle en espérance :
int_rate_gp=df.groupby(['Target']).agg({'int_rate':'mean'})
loan_gp=df.groupby(['Target']).agg({'loan_amnt':'mean'})
tt =df['Target'].value_counts()

#Profit moyen estimé sans modèle : 
    #Hypothèse : le pire coût est le montant du crédit et l'intêret du prêt
Profit_withoutmod = y_0_test*((int_rate_gp["int_rate"][0]/100)*loan_gp["loan_amnt"][0])-(y_1_test*(int_rate_gp["int_rate"][1]/100))
Proba_y_0 = y_0_test/(y_0_test+y_1_test)
Proba_y_1 = y_1_test/(y_0_test+y_1_test)

profit_nM_proba = Proba_y_0*((int_rate_gp["int_rate"][0]/100)*loan_gp["loan_amnt"][0])-(Proba_y_1*(int_rate_gp["int_rate"][1]/100))
#Profit avec le modèle :

Profit_Mod = ((conf_RF[0,0]-conf_RF[1,0])*(int_rate_gp["int_rate"][0]/100)*loan_gp["loan_amnt"][0])-(conf_RF[0,1]*(int_rate_gp["int_rate"][1]/100)*loan_gp["loan_amnt"][1])
Profit_Mod_proba = ((0.807-0.093)*(int_rate_gp["int_rate"][0]/100)*loan_gp["loan_amnt"][0])-(0.002*(int_rate_gp["int_rate"][1]/100)*loan_gp["loan_amnt"][1])

Profit = Profit_Mod-Profit_withoutmod

Profit_capita = Profit/len(y_test)

Profit_capita_prob=Profit_Mod_proba-profit_nM_proba
print(Profit_capita_prob)
print(Profit_capita)
