import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
import logging, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN

# Configure the logger
logging.basicConfig(filename='model_retraining.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("model_retraining")
df = pd.read_csv("Application_Data.csv")
df.drop(df.columns[df.nunique() == 1][0],axis=1,inplace=True)
df.drop('Total_Bad_Debt',axis=1,inplace=True)
df.columns = df.columns.str.strip()

for col in df.select_dtypes(object).columns:
    df[col] = df[col].str.strip()

def onehotencode(data: pd.DataFrame,col: str) -> pd.DataFrame:
    encoder = OneHotEncoder(drop='first',sparse_output=False,max_categories=10)
    encoded_data = encoder.fit_transform(data[[col]])
    encoded_data = pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out())
    return encoded_data

onehotencode_cols = ['Applicant_Gender','Income_Type','Family_Status','Housing_Type','Job_Title']

for col in onehotencode_cols:
    encoded_data = onehotencode(df,col)
    df = pd.concat([df,encoded_data],axis=1)
    df.drop(col,axis=1,inplace=True)

def ordinal_encode(data: pd.DataFrame, col: str) -> pd.Series:
    encoder = OrdinalEncoder(categories=[['Lower secondary','Secondary / secondary special','Incomplete higher','Higher education','Academic degree']])
    data[col] = encoder.fit_transform(data[[col]])
    data[col] = data[col].astype(np.int64)
    return data[col]

df['Education_Type'] = ordinal_encode(df,'Education_Type')

df = df[['Total_Good_Debt', 'Applicant_Gender_M', 'Income_Type_Working',
       'Job_Title_Laborers', 'Family_Status_Married', 'Total_Income',
       'Total_Children', 'Years_of_Working', 'Applicant_Age',
       'Income_Type_State servant','Status']]

X = df.drop('Status',axis=1)
y = df['Status']

smote = ADASYN()
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True,random_state=75)

transformer = ColumnTransformer(transformers=[
    ('log_transform',FunctionTransformer(np.log1p),['Years_of_Working']),
    ('sqrt_transform',FunctionTransformer(np.sqrt),['Total_Children']),
    ('power_transform',PowerTransformer(),['Total_Income'])
],remainder='passthrough')

features = X_train.columns
X_train = transformer.fit_transform(X_train)
X_train = pd.DataFrame(X_train,columns=features)
X_test = transformer.transform(X_test)
X_test = pd.DataFrame(X_test,columns=features)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_model(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_pred)
    logger.info("Evaluation metrics - Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f, ROC-AUC Score: %.2f", accuracy, precision, recall, f1, roc_auc)
    return model, accuracy

model, baseline_acc = train_and_evaluate_model(ExtraTreesClassifier())

param_grid = {'n_estimators': [100,300,600,1000],
             'criterion': ['gini','entropy','log_loss'],
             'max_features': ['auto','sqrt','log2'],
             'bootstrap': [True,False],
             'class_weight': ['balanced','balanced_subsample'],
             'oob_score': [True,False],
             'warm_start': [True,False],
             'max_samples': [0.2,0.4,0.7,1]
             }

grid_et = RandomizedSearchCV(ExtraTreesClassifier(),param_grid,verbose=4)
train_and_evaluate_model(grid_et)

grid_et = RandomizedSearchCV(model,param_grid,cv=5,verbose=0)
optimized_model, optimized_acc = train_and_evaluate_model(grid_et)

if baseline_acc < optimized_acc:
    model = optimized_model

avg_cv_scores = cross_val_score(model,X_test,y_test,scoring='accuracy',cv=5,verbose=2)
mean_score = round(np.mean(avg_cv_scores),4) * 100
logger.info("Mean Cross Validation Performance of Extra Trees Classifier: %.2f%",mean_score)

pipeline = Pipeline(steps=[
    ('transformer',transformer),
    ('scaler',scaler),
    ('model',model)
])

logging.shutdown()
joblib.dump(pipeline,'pipeline.pkl')