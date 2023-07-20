import sqlalchemy as db
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import sys

predict_col = sys.argv[1]
print(predict_col)

engine = create_engine('sqlite:///data/failure.db', echo=False)
connection = engine.connect()
metadata = db.MetaData()
failure = db.Table('failure', metadata, autoload=True, autoload_with=engine)

query = db.select([failure]) 
ResultProxy = connection.execute(query)
ResultSet = ResultProxy.fetchall()


df = pd.DataFrame(ResultSet)
df.columns = ResultSet[0].keys()

df_farenheit = df[df.loc[:, ('Temperature')].str.contains("°F")]
df_degree = df[df.loc[:, ('Temperature')].str.contains("°C")]

df_farenheit['Temperature'] = df_farenheit.loc[:, ('Temperature')].str.replace(" °F","")
df_farenheit['Temperature'] = df_farenheit['Temperature'].astype(float).astype(int)
df_farenheit['Temperature'] = df_farenheit['Temperature'].apply(lambda x: ((x-32 )* 5/9))

df_degree['Temperature'] = df_degree['Temperature'].str.replace(" °C","")
df_degree['Temperature'] = df_degree['Temperature'].astype(float).astype(int)

df = pd.concat([df_farenheit, df_degree]).sort_index()
model_df = df[predict_col]


lab = preprocessing.LabelEncoder()
model_df = lab.fit_transform(model_df)

failure_df = df[['Failure A','Failure B','Failure C','Failure D','Failure E']]

logisticRegr = LogisticRegression(max_iter=1000)
x_train, x_test, y_train, y_test = train_test_split(failure_df, model_df, test_size=0.25, random_state=0)
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)