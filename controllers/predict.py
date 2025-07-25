import pandas as pd
import numpy as np
from  controllers import input_value
from controllers import column_map
from controllers import transform
from controllers import biasmitigate
import joblib
xgboost= joblib.load('models/XGBoost.pkl')

def predict(data):
    df1= pd.DataFrame([data.dict()])
    df2= column_map.map_columns(df1)
    df3 = input_value.replace_outliers(df2)
    df4 = input_value.feature_engineering(df3)
    df5= transform.normalize_data(df3)
    df6=transform.encode_data(df4)
    prediction=xgboost.predict(df6) 
    disparate_impact = 0.808
    statistical_parity_diff = -0.098
    equal_opportunity_diff = -0.027 
    if prediction[0] == 1: 
       title = "Low Risk"
       message = "The applicant is likely to repay the loan."
    else:
        title = "High Risk"
        message = "The applicant is likely to default on the loan."
    print(title, message, disparate_impact, statistical_parity_diff, equal_opportunity_diff)
    
    return {
        'title': title,
        'message': message,
        'disparate_impact': disparate_impact,
        'statistical_parity_diff': statistical_parity_diff,
        'equal_opportunity_diff': equal_opportunity_diff
    }
    

