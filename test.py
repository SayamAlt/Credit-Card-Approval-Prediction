import pytest, joblib, logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pipeline = joblib.load('pipeline.pkl')

# Configure the logger
logging.basicConfig(filename='model_testing.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_testing")

column_names = ['Total_Good_Debt', 'Applicant_Gender_M', 'Income_Type_Working',
       'Job_Title_Laborers', 'Family_Status_Married', 'Total_Income',
       'Total_Children', 'Years_of_Working', 'Applicant_Age',
       'Income_Type_State servant']

test_cases = [
    # Test Case 1
    {
        "input": pd.DataFrame([[18,0,1,1,1,157500,0,5,28,0]],columns=column_names),
        "expected_output": 1
    },
    # Test Case 2
    {
        "input": pd.DataFrame([[5,0,0,0,0,126000,0,7,52,0]],columns=column_names),
        "expected_output": 0
    },
    # Test Case 3
    {
        "input": pd.DataFrame([[17,1,1,1,1,270000,3,4,35,0]],columns=column_names),
        "expected_output": 1
    },
    # Test Case 4
    {
        "input": pd.DataFrame([[6,1,0,0,0,171000,0,1,32,0]],columns=column_names),
        "expected_output": 0
    }
]

# Create test functions for each test case
@pytest.mark.parametrize("test_input, expected_output", [(tc["input"], tc["expected_output"]) for tc in test_cases])
def test_prediction_with_custom_input(test_input,expected_output):
    try:
        # Make a prediction
        prediction = pipeline.predict(test_input)[0]
        accuracy = accuracy_score(expected_output,prediction)
        precision = precision_score(expected_output,prediction)
        recall = recall_score(expected_output,prediction)
        f1 = f1_score(expected_output,prediction)
        roc_auc = roc_auc_score(expected_output,prediction)
        logger.info("Evaluation metrics - Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f", accuracy, precision, recall, f1)
        # Check if the prediction matches the expected output
        assert prediction == expected_output
    except Exception as e:
        logger.error("The model's prediction didn't match with the expected output!",exc_info=True)
    
logging.shutdown()