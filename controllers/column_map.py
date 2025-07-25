
import pandas as pd

def map_columns(data):
    column_mapping = {
        'age': 'borrowerage',
        'region': 'region',
        'maritalStatus': 'maritalstatus',
        'householdSize': 'householdsize',
        'employmentStatus': 'employmentstatus',
        'educationLevel': 'level',
        'loanAmount': 'loanamount',
        'repaidAmount': 'repaidamount',
        'collectionEfforts': 'collectionefforts',
        'originalInterestRate': 'originalloanintrate',
        'currentInterestRate': 'currentloanintrate',
        'repaymentPlan': 'repaymentplan'
    }
    data1 = data.rename(columns=column_mapping)
    data1['inflationrate']=9

    return data1


