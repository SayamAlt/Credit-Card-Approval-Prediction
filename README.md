# Credit Card Approval Prediction

![Credit Card Approval Prediction](https://th.bing.com/th/id/OIP.6HyOVBe3qnbodDgymUuTVgHaDW?pid=ImgDet&rs=1)
![Credit Card Approved](https://cdn.gobankingrates.com/wp-content/uploads/2016/07/istock_21970466_large-2.jpg)
![Credit Card Issued](https://miro.medium.com/max/1200/1*oLsdDr8xP3h2j66YFXyMXQ.jpeg)

This project is a Credit Card Approval Prediction system that uses machine learning techniques to evaluate credit card applications and predict whether an applicant is likely to be approved or rejected. The project includes a Flask web application for making predictions, Continuous Integration/Continuous Deployment (CI/CD) setup using Docker and GitHub Actions, model testing with the pytest module, and automatic model retraining on every push to the repository.

## Table of Contents

- [Context](#context)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [CI/CD Workflow](#cicd-workflow)
- [Testing](#testing)
- [Model Retraining](#model-retraining)
- [Database Integration](#database-integration)
- [Contributing](#contributing)
- [License](#license)

## Context

Credit score cards are a common risk control method in the financial industry. It uses personal information and data submitted by credit card applicants to predict the probability of future defaults and credit card borrowings. The bank is able to decide whether to issue a credit card to the applicant. Credit scores can objectively quantify the magnitude of risk.

Machine learning plays a significant role in credit card approval prediction due to its ability to analyze vast amounts of data, make complex decisions, and identify patterns that may not be evident through traditional methods. Here are some key reasons why machine learning is important in this context:

1. **Risk Assessment**: Machine learning models can assess an applicant's creditworthiness by analyzing their credit history, financial behavior, and various other factors. These models can calculate credit risk more accurately, leading to better decision-making.

2. **Data-Driven Decisions**: Credit card approval decisions benefit from data-driven insights. Machine learning models use historical data to make predictions about a new applicant's likelihood of defaulting on payments, helping issuers approve or decline applications.

3. **Scalability**: Machine learning can handle large volumes of applications efficiently, making it suitable for institutions that process thousands or millions of credit card applications.

4. **Customization**: ML models can be customized to the specific needs of a credit card company. This allows for a tailored approach in assessing credit risk, setting credit limits, and offering personalized card options.

5. **Fraud Detection**: Machine learning helps in real-time fraud detection by analyzing transaction data for unusual patterns or anomalies. This enhances card security and reduces the risk of unauthorized charges.

6. **Continuous Learning**: Machine learning models can adapt to changing circumstances and trends. They can continuously learn from new data and adjust their decision-making processes, improving accuracy over time.

7. **Automation**: Automation of the credit approval process reduces the need for manual review, making it faster and more efficient. This benefits both credit card issuers and applicants.

8. **Reduced Human Bias**: Machine learning models are less prone to human bias, such as discrimination based on gender, race, or age. This leads to fairer lending practices.

9. **Improved Customer Experience**: Faster approval processes and personalized credit offerings can enhance the customer experience, making credit card companies more competitive in the market.

10. **Regulatory Compliance**: Machine learning models can be designed to comply with relevant regulations and laws governing the credit industry, ensuring that lending practices meet legal requirements.

11. **Predictive Insights**: ML models can provide insights into an applicant's future credit behavior, helping card issuers make more informed decisions about credit limits, interest rates, and promotional offers.

12. **Default Prediction**: ML models can predict the likelihood of an applicant defaulting on payments. This helps issuers take proactive measures to minimize losses.

While machine learning offers significant advantages, it's important to note that the accuracy and fairness of these models are critical considerations. Proper data handling, model validation, and adherence to ethical guidelines are essential to ensure that credit approval predictions are both accurate and fair to all applicants.
 
Generally speaking, credit score cards are based on historical data. Once encountering large economic fluctuations. Past models may lose their original predictive power. Logistic model is a common method for credit scoring. Because Logistic is suitable for binary classification tasks and can calculate the coefficients of each feature. In order to facilitate understanding and operation, the score card will multiply the logistic regression coefficient by a certain value (such as 100) and round it.
 
At present, with the development of machine learning algorithms. More predictive methods such as Boosting, Random Forest, and Support Vector Machines have been introduced into credit card scoring. However, these methods often do not have good transparency. It may be difficult to provide customers and regulators with a reason for rejection or acceptance.

## Project Structure

The project has the following structure:

bash

credit-card-approval-prediction/
│
├── app/
│   ├── app.py                # Flask web application
│   ├── templates/            # HTML templates
│   └── ...
│
├── data/
│   ├── Application_Data.csv       # Historical credit application data
│   └── ...
│
├── model/
│   ├── model.pkl             # Pre-trained Extra Trees Classifier model
│   └── ...
│
├── tests/
│   ├── test_model.py         # Unit tests using pytest
│   └── ...
│
├── requirements.txt         # Python dependencies
│
├── Dockerfile               # Docker containerization
│
├── retrain_script.py        # Model retraining script
│
├── README.md                # Project documentation
│
└── ...

## Prerequisites

Before you begin, ensure you have met the following requirements:

<ul>
    <li>Python 3.7+</li>
    <li>Docker</li>
    <li>Git</li>
    <li>GitHub account</li>
    <li>SQLite database (create database.db for storing data during prediction)</li>
</ul>

## Installation

Clone this repository:

bash

git clone https://github.com/SayamAlt/credit-card-approval-prediction.git

Navigate to the project directory:

bash

cd credit-card-approval-prediction

## Install Python dependencies:

bash

pip install -r requirements.txt

## Usage

Run the Flask web application:

bash

python app/app.py

Access the web application at http://localhost:8000 in your web browser.

## CI/CD Workflow

The CI/CD pipeline is set up with GitHub Actions. It automatically builds and deploys the Docker container whenever changes are pushed to the repository. You can find the GitHub Actions workflow in .github/workflows/main.yaml.

## Testing

To run unit tests using pytest, execute the following command:

bash

pytest tests/test_model.py

## Model Retraining

The model retraining script, retrain.py, is automatically triggered on every push to the repository. It fetches newly obtained data from the SQLite database (credit_application.db) and re-trains the model. Additionally, it performs hyperparameter tuning and cross validation to fine-tune the model and enhance its performance.

## Database Integration

The SQLite database (credit_application.db) is used for storing newly obtained data during prediction and for model retraining.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to this repository.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE.md file for details.

