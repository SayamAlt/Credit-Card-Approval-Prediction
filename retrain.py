import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
import logging, joblib, warnings, sqlite3
warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN

# Configure the logger
logging.basicConfig(filename='model_retraining.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("model_retraining")

conn = sqlite3.connect("credit_card_application.db")
cursor = conn.cursor()

# Fetch all rows from the table in the database
cursor.execute("SELECT * FROM CREDIT_APPLICATION")
records = cursor.fetchall()

# Extract the column names from the database table
table_name = "CREDIT_APPLICATION"
cursor.execute(f"PRAGMA table_info({table_name})")
column_names = [info[1] for info in cursor.fetchall()]
conn.close()

df = pd.DataFrame(records,columns=column_names)

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

grid_et = RandomizedSearchCV(ExtraTreesClassifier(),param_grid,verbose=0)
train_and_evaluate_model(grid_et)

grid_et = RandomizedSearchCV(model,param_grid,cv=5,verbose=0)
optimized_model, optimized_acc = train_and_evaluate_model(grid_et)

if baseline_acc < optimized_acc:
    model = optimized_model

avg_cv_scores = cross_val_score(model,X_test,y_test,scoring='accuracy',cv=5,verbose=2)
mean_score = round(np.mean(avg_cv_scores),4) * 100
logger.info(f"Mean Cross Validation Performance of Extra Trees Classifier: {round(mean_score,2)}%")

pipeline = Pipeline(steps=[
    ('transformer',transformer),
    ('scaler',scaler),
    ('model',model)
])

logging.shutdown()
joblib.dump(pipeline,'backup_pipeline.pkl')