import joblib
scaler= joblib.load('models/scaler.pkl')
processor= joblib.load('models/preprocessor.pkl')

def normalize_data(dfI):
    numerical_columns= ['borrowerage', 'householdsize', 'loanamount', 'repaidamount',
       'remainingamount', 'originalloanintrate', 'currentloanintrate',
       'interestratechange', 'inflationrate']
    dfI[numerical_columns] = scaler.transform(dfI[numerical_columns])


    
    return dfI

   
def encode_data(dfI):
   dfI = processor.transform(dfI)

   return dfI
    