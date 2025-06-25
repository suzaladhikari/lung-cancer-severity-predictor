import joblib
import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, recall_score,precision_score,f1_score
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

##-------------------------------------------Model Preparation----------------------------------

data = pd.read_csv('cleaned.csv') ## This is the data that we will be using throughout the model preparation 
label = LabelEncoder()## For Level column we will be using the Label Encoder 
data['severityLevel'] = label.fit_transform(data['Level']) ## This will change each of the category(Low, Medium, and High) to the numerical value
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0}) ## This change the Gender column to binary values 
targetedColumn = 'severityLevel' ## This is the column that we will predict
features = ['Age', 'Gender', 'Smoking', 'Alcohol use', 'Obesity', 'Balanced Diet','Fatigue', 'Coughing of Blood', 'Chest Pain', 'Air Pollution','Genetic Risk', 'chronic Lung Disease'] ##This are the features on which the level will be predicted 
X = data[features]
y  = data[targetedColumn]
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42) ## The data has been split into parts 
steps = [("impute", SimpleImputer()) ## First it will check the missing values
         ,("scalar",StandardScaler()) ## Then it will transform each feature to have a mean of 0 and std of 1 
         ,("logit",LogisticRegression(C = 10))] ## This is the model we will be using to predict the values 
pipeline = Pipeline(steps) ## This is the pipeline that will execute the steps
pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

##-----------------------------Metrics to deterime model's performance-------------------------------

precision = precision_score(y_pred, y_test , average='macro') ## This evaluates the TruePositives
recall = recall_score(y_pred, y_test , average='macro') ## This evaluates the number of true positives 
f1Score = f1_score(y_pred, y_test , average='macro') 

#Cross Validation Score 
scores = cross_val_score(pipeline, X_train,y_train, cv = 5,scoring='accuracy')
meanAccuracy = scores.mean() ## This gives the model's mean accuracy ! 
modelStability = scores.std() ## This gives how stable the model is, if the stability is less, the model is more stable

##Hyperparametre Tuning 
## Its always better if the model has good hyperparametrs to avoid the overfitting for that we use GridSearchCV

params = {
    'logit__C':[0.01, 0.1, 1, 10, 100],
    'logit__class_weight':[None,'balanced']
}
grid = GridSearchCV(pipeline, param_grid= params, cv = 5) ##Here we determine which parameter is the best to use?!
grid.fit(X_train,y_train)
bestParameters = grid.best_params_ ## We got the C value to be 10 so we replace it with C = 1 which is default 


###This is for the deploying the file for the webapp!

joblib.dump(pipeline, 'lungCancerPredictor.pkl') 


