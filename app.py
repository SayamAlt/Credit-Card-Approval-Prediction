from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load('backup_pipeline.pkl')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        yes_no_mapping = {"Yes": 1, "No": 0}
        total_good_debt = float(request.form['total_good_debt'])
        applicant_gender_male = float(yes_no_mapping[request.form['applicant_gender_male']])
        income_type_working = float(yes_no_mapping[request.form['income_type_working']])
        job_title_laborers = float(yes_no_mapping[request.form['job_title_laborers']])
        family_status_married = float(yes_no_mapping[request.form['family_status_married']])
        total_income = float(request.form['total_income'])
        total_children = float(request.form['total_children'])
        years_of_working = float(request.form['years_of_working'])
        applicant_age = float(request.form['applicant_age'])
        income_type_state_servant = float(yes_no_mapping[request.form['income_type_state_servant']])
        data = pd.DataFrame([[total_good_debt,
                              applicant_gender_male,
                              income_type_working,
                              job_title_laborers,
                              family_status_married,
                              total_income,
                              total_children,
                              years_of_working,
                              applicant_age,
                              income_type_state_servant]],columns=['Total_Good_Debt', 
                                                                   'Applicant_Gender_M', 
                                                                   'Income_Type_Working',
                                                                   'Job_Title_Laborers', 
                                                                   'Family_Status_Married', 
                                                                   'Total_Income',
                                                                   'Total_Children', 
                                                                   'Years_of_Working', 
                                                                   'Applicant_Age',
                                                                   'Income_Type_State servant'])
        pred = pipeline.predict(data)[0]
        
        if pred == 0:
            return render_template('index.html',prediction_text="Your credit card application won't be approved.")
        elif pred == 1:
            return render_template('index.html',prediction_text="Bravo! Your credit card application will be approved.")
        
if __name__ == '__main__':
    app.run(port=8000)