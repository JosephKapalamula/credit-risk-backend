import numpy as np
import pandas as pd
def replace_outliers(dfI):
    numeric_cols = [col for col in dfI.select_dtypes(include=[np.number]) ]
    for col in numeric_cols:
        Q1 = dfI[col].quantile(0.25)
        Q3 = dfI[col].quantile(0.75)
        IQR = Q3 - Q1

        # Outlier bounds 
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers instead of removing them
        dfI[col] = np.where(dfI[col] < lower_bound, lower_bound, dfI[col])
        dfI[col] = np.where(dfI[col] > upper_bound, upper_bound, dfI[col])

    return dfI

def feature_engineering(dfI):
    dfI['interestratechange'] = dfI['currentloanintrate'] - dfI['originalloanintrate']
    dfI['remainingamount'] = dfI['loanamount'] - dfI['repaidamount']
    return dfI 
