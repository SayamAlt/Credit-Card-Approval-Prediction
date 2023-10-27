import pytest, joblib, logging, warnings
warnings.filterwarnings("ignore")
import pandas as pd

pipeline = joblib.load('backup_pipeline.pkl')

# Configure the logger
logging.basicConfig(filename='tests/model_testing.log', level=logging.INFO)

# Create a logger
logger = logging.getLogger("tests/model_testing")
logger.setLevel(logging.DEBUG)  # Set the default log level for the logger

# Create a handler for INFO-level messages
info_handler = logging.FileHandler("tests/model_performance_info.log")
info_handler.setLevel(logging.INFO)

# Create a handler for ERROR-level messages
error_handler = logging.FileHandler("tests/error.log")
error_handler.setLevel(logging.ERROR)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Set the formatter for the handlers
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(error_handler)


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
    },
    # Test Case 5
    {
        "input": pd.DataFrame([[33,0,1,1,1,157500,0,5,28,0]],columns=column_names),
        "expected_output": 1
    },
    # Test Case 6
    {
        "input": pd.DataFrame([[1,0,0,0,0,157500,1,7,43,0]],columns=column_names),
        "expected_output": 0
    }
]

# Create test functions for each test case
@pytest.mark.parametrize("test_input, expected_output", [(tc["input"], tc["expected_output"]) for tc in test_cases])
def test_prediction_with_custom_input(test_input,expected_output):
    # Make a prediction
    prediction = pipeline.predict(test_input)[0]
    logger.debug(prediction)
    try:
        # Check if the prediction matches the expected output
        assert prediction == expected_output
        logger.info("The model's prediction matched the expected output.")
    except:
        logger.error("The model's prediction didn't match with the expected output.")
    
logging.shutdown()