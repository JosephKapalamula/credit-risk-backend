def mapping(dfI,df_mapp):
    region_map = {
    'North': 1,    # Privileged
    'South': 1,
    'Central':0
    }

    marital_map = {
    'Single': 1,     # Privileged
    'Married': 1,
    'Divorced': 0
    }
    # dfI['region_binary'] = df_mapp['region'].map(region_map)
    # dfI['marital_binary'] = df_mapp['maritalstatus'].map(marital_map)
    
    return dfI



