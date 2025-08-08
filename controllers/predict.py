import pandas as pd
import numpy as np
from controllers import input_value
from controllers import column_map
from controllers import transform
from controllers import biasmitigate
import joblib
import shap


# Load models
xgboost = joblib.load('models/XGBoost.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Feature explanation templates (used to generate human-readable insights)
explanation_templates = {
    'maritalstatus_Married': {
        'low_risk': "Being married increased the chance of paying back the loan.",
        'high_risk': "Being married increased the chance of not paying back the loan."
    },
    'maritalstatus_Single': {
        'low_risk': "Being single increased the chance of paying back the loan(unexpected).",
        'high_risk': "Being single increased the chance of not paying back the loan."
    },
    'region_North': {
        'low_risk': "Living in the North region increased the chance of paying back the loan.",
        'high_risk': "Living in the North region increased the chance of not paying back the loan."
    },
    'region_South': {
        'low_risk': "Living in the South region increased the chance of paying back the loan.",
        'high_risk': "Living in the South region increased the chance of not paying back the loan."
    },
    'employmentstatus_Self-employed': {
        'low_risk': "Being self-employed increased the chance of paying back the loan.",
        'high_risk': "Being self-employed increased the chance of not paying back the loan."
    },
    'employmentstatus_Unemployed': {
        'low_risk': "Being unemployed increased the chance of paying back the loan (unexpected).",
        'high_risk': "Being unemployed increased the chance of not paying back the loan."
    },
    'level_Diploma': {
        'low_risk': "Having a diploma increased the chance of paying back the loan(unexpected).",
        'high_risk': "Having a diploma increased the chance of not paying back the loan."
    },
    'repaymentplan_Voluntary': {
        'low_risk': "A voluntary repayment plan increased the chance of paying back the loan(unexpected).",
        'high_risk': "A voluntary repayment plan increased the chance of not paying back the loan."
    },
    'collectionefforts_No Collectioneffort': {
        'low_risk': "Having no collection efforts improved the chance of paying back the loan(unexpected).",
        'high_risk': "Having no collection efforts reduced the perceived reliability."
    },
    'collectionefforts_Notices': {
        'low_risk': "Having received notices increased the chance of paying back the loan (unexpected).",
        'high_risk': "Receiving notices from collections increased the chance of not paying back the loan."
    },
    'collectionefforts_Reminders': {
        'low_risk': "Receiving payment reminders increased the chance of paying back the loan (unexpected).",
        'high_risk': "Receiving payment reminders increased the chance of not paying back the loan."
    },
    'borrowerage': {
        'low_risk': "The applicant's age decreased the risk of not paying back the loan.",
        'high_risk': "The applicant's age increased the  risk of not paying back the loan."
    },
    'householdsize': {
        'low_risk': "A smaller household size decreased the chance of not paying back the loan.",
        'high_risk': "A larger household size increased the chance of not paying back the loan."
    },
    'loanamount': {
        'low_risk': "A lower loan amount decreased the risk of not paying back.",
        'high_risk': "A higher loan amount increased the risk of not paying back."
    },
    'repaidamount': {
        'low_risk': "A higher repaid amount improved the credit profile and lowered the risk.",
        'high_risk': "A lower repaid amount worsened the credit profile and increased the risk."
    },
    'remainingamount': {
        'low_risk': "A lower remaining loan amount decreased the risk of not paying back.",
        'high_risk': "A higher remaining loan amount increased the risk of not paying back."
    },
    'originalloanintrate': {
        'low_risk': "A lower original loan interest rate decreased the risk of paying back the money.",
        'high_risk': "A higher original loan interest rate increased the risk of the risk of paying back the money."
    },
    'currentloanintrate': {
        'low_risk': "A lower current loan interest rate decreased the risk of paying back the loan.",
        'high_risk': "A higher current loan interest rate increased the risk of paying back the loan."
    },
    'interestratechange': {
        'low_risk': "A decrease in interest rate lower the risk of paying.",
        'high_risk': "An increase in interest rate raised the risk of not paying."
    }
}

friendly_labels = {
    'maritalstatus_Married': "Marital Status: Married",
    'maritalstatus_Single': "Marital Status: Single",
    'region_North': "Region: North",
    'region_South': "Region: South",
    'employmentstatus_Self-employed': "Employment Status: Self-employed",
    'employmentstatus_Unemployed': "Employment Status: Unemployed",
    'level_Diploma': "Education Level: Diploma",
    'repaymentplan_Voluntary': "Voluntary Repayment Plan",
    'collectionefforts_No Collectioneffort': "No Collection Efforts",
    'collectionefforts_Notices': "Collection Notices",
    'collectionefforts_Reminders': "Collection Reminders",
    'borrowerage': "Borrower Age",
    'householdsize': "Household Size",
    'loanamount': "Loan Amount",
    'repaidamount': "Amount Repaid",
    'remainingamount': "Remaining Loan Amount",
    'originalloanintrate': "Original Loan Interest Rate",
    'currentloanintrate': "Current Loan Interest Rate",
    'interestratechange': "Interest Rate Change"
}

def predict(data):
    # 1. Data preprocessing pipeline
    df1 = pd.DataFrame([data.dict()])
    df2 = column_map.map_columns(df1)
    df3 = input_value.replace_outliers(df2)
    df4 = input_value.feature_engineering(df3)
    df5 = transform.normalize_data(df3)
    df6 = transform.encode_data(df4)

    # 2. Model prediction
    prediction = xgboost.predict(df6)  # 0 = High Risk, 1 = Low Risk

    # 3. SHAP explanation
    explainer = shap.TreeExplainer(xgboost)
    shap_values = explainer.shap_values(df6)
    feature_names = preprocessor.get_feature_names_out()

    df7 = pd.DataFrame(df6.toarray() if hasattr(df6, 'toarray') else df6,
                       columns=feature_names)

    def clean_feature_name(name):
        return name.replace('onehot__', '').replace('remainder__', '')

    excluded_features = ['remainder__inflationrate']

    top_features = sorted(
        [
            (feature, value)
            for feature, value in zip(df7.columns, shap_values[0])
            if feature not in excluded_features
        ],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    # 4. Build explanations relative to predicted class
    top_features_output = []
    predicted_class = int(prediction[0])
    prediction_label = 'low_risk' if predicted_class == 1 else 'high_risk'

    for feature, contribution in top_features:
        base_name = clean_feature_name(feature)

        # Get friendly label, fallback to base_name if not found
        label = friendly_labels.get(base_name, base_name)

        # Get explanation template
        template = explanation_templates.get(base_name, {})
        explanation = template.get(
            prediction_label,
            f"{label} contributed to the predicted {prediction_label.replace('_', ' ')}."
        )

        top_features_output.append({
            "feature": base_name,          # Keep original feature key if needed
            "label": label,                # Friendly name for frontend display
            "explanation": explanation
        })
    # 5. Prediction summary
    if predicted_class == 1:
        title = "Low Risk"
        message = "The applicant is likely to repay the loan."
    else:
        title = "High Risk"
        message = "The applicant is likely to default on the loan."

    # 6. Fairness metrics (static here, update if dynamic)
    return {
        'title': title,
        'message': message,
        'disparate_impact': 0.808,
        'statistical_parity_diff': -0.098,
        'equal_opportunity_diff': -0.027,
        'top_features': top_features_output,
    }
