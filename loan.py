import pandas as pd
import numpy as np

import re
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor

#Visualization


train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
train_len = len(train)
dataset = pd.concat([train, test], axis=0).reset_index(drop=True)
dataset.info()
dataset["Married"] = dataset["Married"].fillna("Yes")

dataset.loc[dataset["Gender"].isnull(),"Gender"] =dataset["Married"][dataset["Gender"].isnull()].apply(lambda x: "Male" if x == "Yes" else "Female")

Dependents_choice = list(dataset["Dependents"].value_counts().index)
dataset.loc[dataset["Dependents"].isnull(),"Dependents"] = dataset["Married"][dataset["Dependents"].isnull()].apply(
lambda x: '0' if x == "No" else np.random.choice(Dependents_choice))

dataset["Loan_Amount_Term"] = dataset["Loan_Amount_Term"].fillna(dataset["Loan_Amount_Term"].mode()[0])

dataset["Self_Employed"] = dataset["Self_Employed"].fillna(dataset["Self_Employed"].mode()[0])

dataset["Credit_History"] = dataset["Credit_History"].fillna("nohis")



'''
train["Loan_Status"] = train["Loan_Status"].map({"Y":1,"N":0})

train["Gender"] = train["Gender"].map({"Male":1,
                                          "Female":0})
train["Married"] = train["Married"].map({"Yes":1,
                                            "No":0})
train["Dependents"] = train["Dependents"].map({"1":1,
                                                  "2":2,
                                                  "3+":3,
                                                  "0":0})
train["Education"] = train["Education"].map({"Graduate":1,
                                                "Not Graduate":0})
train["Self_Employed"] = train["Self_Employed"].map({"Yes":1,
                                                        "No":0})
train["Property_Area"] = train["Property_Area"].map({"Urban":0,
                                                        "Semiurban":1,
                                                        "Rural":2})
train["Loan_Status"] = train["Loan_Status"].map({"Y":1,"N":0})
'''



def fill_missing_feature(data, missing_feature, corr_feature):
    known = data[data[missing_feature].notnull()]
    unknown = data[data[missing_feature].isnull()]

    y = known[missing_feature]

    x = known[corr_feature]

    clf = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    clf.fit(x,y)

    data.loc[data[missing_feature].isnull(),missing_feature] = clf.predict(unknown[corr_feature])
    print(clf.predict(unknown[corr_feature]))
    return data

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset["Credit_History"] = dataset["Credit_History"].replace('nohis', 2.0)
dataset["Dependents"] = encoder.fit_transform(dataset["Dependents"])
dataset["Gender"] = encoder.fit_transform(dataset["Gender"])
dataset["Married"] = encoder.fit_transform(dataset["Married"])
dataset["Property_Area"] = encoder.fit_transform(dataset["Property_Area"])
dataset["Education"] = encoder.fit_transform(dataset["Education"])
dataset["Self_Employed"] = encoder.fit_transform(dataset["Self_Employed"])
dataset = fill_missing_feature(dataset,"LoanAmount",["ApplicantIncome","CoapplicantIncome","Dependents","Education",
                                                    "Married","Self_Employed"])
train = dataset[:train_len]
train["Loan_Status"] = train["Loan_Status"].map({"Y":1,"N":0})
loan_status = train["Loan_Status"]
train.drop(["Loan_ID","Loan_Status","Credit_History"], axis=1, inplace=True)


train_feature = list(train.columns)
df_plus = pd.DataFrame()
for i in range(len(train_feature)):
  for j in range(i+1, len(train_feature)):
    featurn_name = train_feature[i] + '+' + train_feature[j]
    df_plus[featurn_name] =train[train_feature[i]] + train[train_feature[j]]

df_subtract = pd.DataFrame()
for i in range(len(train_feature)):
  for j in range(i+1, len(train_feature)):
    featurn_name = train_feature[i] + '-' + train_feature[j]
    df_subtract[featurn_name] =train[train_feature[i]] - train[train_feature[j]]


df_multi = pd.DataFrame()
for i in range(len(train_feature)):
  for j in range(i+1, len(train_feature)):
    featurn_name = train_feature[i] + '*' + train_feature[j]
    df_multi[featurn_name] =train[train_feature[i]] - train[train_feature[j]]


df_div = pd.DataFrame()
for i in range(len(train_feature)):
  for j in range(i+1, len(train_feature)):
    featurn_name = train_feature[i] + '/' + train_feature[j]
    df_div[featurn_name] =train[train_feature[i]] - train[train_feature[j]]


df_div = pd.DataFrame()
for i in range(len(train_feature)):
  for j in range(i+1, len(train_feature)):
    featurn_name = train_feature[i] + '/' + train_feature[j]
    df_div[featurn_name] =train[train_feature[i]] - train[train_feature[j]]


df_conpound = pd.DataFrame()
for i in range(len(train_feature)):
  for j in range(i+1, len(train_feature)):
    for k in range(j+1,len(train_feature)):
      feature_name = "(" + train_feature[i] + "-" + train_feature[j] + ")" + "*" + train_feature[k]
      df_conpound[feature_name] = (train[train_feature[i]] + train[train_feature[j]]) * train[train_feature[k]]
      feature_name = "(" + train_feature[i] + "-" + train_feature[k] + ")" + "*" + train_feature[j]
      df_conpound[feature_name] = (train[train_feature[i]] - train[train_feature[k]]) * train[train_feature[j]]
      feature_name = "(" + train_feature[j] + "-" + train_feature[k] + ")" + "*" + train_feature[i]
      df_conpound[feature_name] = (train[train_feature[j]] - train[train_feature[k]]) * train[train_feature[i]]




df = pd.concat([df_conpound,df_div,df_multi,df_plus,df_subtract],axis = 1)
df["Loan_Status"] = loan_status

correlation = df.corr()["Loan_Status"]
print(correlation[correlation > 0.1])



from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score
clf_name = []
cv_precision = []
cv_recall = []
cv_f1score = []
for clf in classifiers:
  y_train_pred = cross_val_predict(clf, x_tarin, y_train, cv=kfold, n_jobs=-1)
  clf_name.append(clf.__class__.__name__)
  cv_precision.append(precision_score(y_train, y_train_pred))
  cv_recall.append(recall_score(y_train, y_train_pred))
  cv_f1score.append(f1_score(y_train, y_train_pred))
cv_res = pd.DataFrame({"clf_name":clf_name,"precision_score":cv_precision,
                      "recall_score":cv_recall,"f1_score":cv_f1score})
cv_res.sort_values(by="f1_score",ascending=False,inplace=True)


