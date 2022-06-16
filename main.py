#IMPORT REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

#READ THE DATASET
Data = pd.read_csv('/home/anushka/Downloads/heart_2020_cleaned.csv')

#CONVERT TEXT VALUES TO NUMERIC FOR BETTER ANALYSIS
le = LabelEncoder()
col = Data[['HeartDisease', 'Smoking', 'AlcoholDrinking','AgeCategory', 'Stroke', 'DiffWalking','Race', 'Sex','PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer','GenHealth' ,'Diabetic']]
for i in col:
    Data[i] = le.fit_transform(Data[i])
#print(Data.head())

#SCALE THE NUMERIC DATA WRT THE GIVEN RANGE FOR EACH FIELD
Scaler = StandardScaler()
num_cols = ['MentalHealth', 'BMI', 'PhysicalHealth', 'SleepTime']
Data[num_cols] = Scaler.fit_transform(Data[num_cols])

#PLOT CORRELATION HEATMAP
import plotly.express as px
fig = px.imshow(Data.corr(), color_continuous_scale="Blues")
fig.update_layout(height=800)
fig.show()

#DECLARE TARGET VARIABLES
X = Data.drop(columns=['HeartDisease'], axis=1)
Y = Data['HeartDisease']

#SPLIT X AND Y DATASETS AS TRAIN AND TEST IN 80:20 RATIO
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.3, random_state=40)
print(Y_train.value_counts())

#TO SOLVE THE PROBLEM OF UNEVEN DATA DISTRIBUTION, WE OVERSAMPLE THE TRAINING DATASETS
rs = RandomOverSampler(random_state=40)
X_train_resample, Y_train_resample = rs.fit_resample(X_train, Y_train)
print(Y_train_resample.value_counts())

#TRAINING THE DATA WITH RANDOM FOREST
rf = RandomForestClassifier(n_estimators=9)
rf.fit(X_train_resample, Y_train_resample)
Y_prediction = rf.predict(X_test)
print(classification_report(Y_test, Y_prediction))

#TRAINING THE DATA WITH NAIVE BAYES CLASSIFIER
bnb = BernoulliNB()
bnb.fit(X_train_resample, Y_train_resample)
Y_pred_bnb = bnb.predict(X_test)
print(classification_report(Y_test, Y_pred_bnb))

#UPON CONDUCTING ANALYSIS FROM THE TWO METHODS MENTIONED ABOVE, IT IS CONCLUDED THAT RANDOM FOREST IS
#MORE EFFECTIVE IN THIS CASE




