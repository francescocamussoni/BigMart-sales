def StandarScaler(x_train, x_test):
    '''
    Standar scaler
  
    Parameters
    ----------
    x_train : pandas.dataframe
      The training dataframe to scale.
    x_test : pandas.dataframe
      The validaiton/test dataframe to scale

    Returns
    -------
    x_train : pandas.dataframe
      Scaled x_train
    data_t : pandas.dataframe
      Scaled x_test
     '''
        
    u = x_train.mean()  # escaleo con los datos de train
    s = x_train.std()  # escaleo con los datos de train
    x_train = (x_train-u) / s
    x_test = (x_test-u) / s
    return x_train, x_test

def RobustScaler(x_train, x_test):
    '''
    Robust scaler
  
    Parameters
    ----------
    x_train : pandas.dataframe
      The training dataframe to scale.
    x_test : pandas.dataframe
      The validaiton/test dataframe to scale.

    Returns
    -------
    x_train : pandas.dataframe
      Scaled x_train.
    data_t : pandas.dataframe
      Scaled x_test.
     '''
    
    m = x_train.median() #escaleo con los datos de train
    p25 = x_train.quantile(0.25) #escaleo con los datos de train
    p75 = x_train.quantile(0.75) #escaleo con los datos de train
    x_train = (x_train-m)/(p75-p25)
    x_test = (x_test-m)/(p75-p25)
    return x_train, x_test

def MeanImputer_NA(data, data_t, atr, atr_filter):
    '''
    Mean Imputer for NA values
  
    Parameters
    ----------
    data : pandas.dataframe
      The training dataframe to process.
    data_t : pandas.dataframe
      The validaiton/test dataframe to process.
    atr: str
      Attribute that contains NA values.
    atr_filter : str
      Attribute used to filter data and compute mean value.

    Returns
    -------
    data : pandas.dataframe
      Preprocessed data.
    data_t : pandas.dataframe
      Preprocessed data_t.
    '''
    key_filter = data[atr_filter].unique()
    for key in key_filter:
        mean = np.mean(data[atr][data[atr_filter] == key])
        data[atr][data[atr_filter] == key] = data[atr][data[atr_filter] == key].fillna(mean)
        data_t[atr][data_t[atr_filter] == key] = data_t[atr][data_t[atr_filter] == key].fillna(mean)
    return data, data_t

def MeanImputer_0(data, data_t, atr, atr_filter):
    '''
    Mean Imputer for 0 (missing) values 
  
    Parameters
    ----------
    data : pandas.dataframe
      The training dataframe to process.
    data_t : pandas.dataframe
      The validaiton/test dataframe to process.
    atr: str
      Attribute that contains 0 values.
    atr_filter : str
      Attribute used to filter data and compute mean value.

    Returns
    -------
    data : pandas.dataframe
      Preprocessed data.
    data_t : pandas.dataframe
      Preprocessed data_t.
    '''
    key_filter = data[atr_filter].unique()
    for key in key_filter:
        mean = np.mean(data[atr][data[atr_filter] == key])
        data[atr][data[atr_filter] == key] = data[atr][data[atr_filter] == key].replace(0, mean)
        data_t[atr][data_t[atr_filter] == key] = data_t[atr][data_t[atr_filter] == key].replace(0, mean)
    return data, data_t

def MeanImputer_Size(data, data_t, atr, atr_filter):
    '''
    Mean Imputer for size missing values
  
    Parameters
    ----------
    data : pandas.dataframe
      The training dataframe to process.
    data_t : pandas.dataframe
      The validaiton/test dataframe to process.
    atr: str
      Attribute that contains 0 values.
    atr_filter : str
      Attribute used to filter data and replace NaN with correct values
    Returns
    -------
    data : pandas.dataframe
      Preprocessed data.
    data_t : pandas.dataframe
      Preprocessed data_t.
    '''
    data[atr][data[atr_filter]=='010'] = 'Small'
    data[atr][data[atr_filter]=='045'] = 'Medium'
    data[atr][data[atr_filter]=='017'] = 'Medium'
    data_t[atr][data_t[atr_filter]=='010'] = 'Small'
    data_t[atr][data_t[atr_filter]=='045'] = 'Medium'
    data_t[atr][data_t[atr_filter]=='017'] = 'Medium'
    return data, data_t

def ValueGrouper(Serie, group, value):
    for g in group:
        Serie = Serie.replace(g, value)

def LogTransform(x):
    '''
    Logarithmic space transform function. 
    '''
    return np.log(x+1)

def LogAntitransform(x):
    '''
    Anti-logarithmic space transform function. 
    '''
    return np.exp(x)-1

def predict(model, antitransform, x):
    return antitransform(model.predict(x))

def preprocessing(dataframe, dataframe_t, droppable_columns, integer_columns, categorical_columns, objective, transform=None, one_hot=False):
    '''
    Preprocessing function for the proposed dataframe.

    This functions modifies the dataframe from of the training set and the dataframe of the validation/test set to get categorical columns from strings,
    dropping non-informative attributes, and rescaling continuous attributes.

    Parameters
    ----------
    dataframe : pandas.dataframe
      The training dataframe to process.
    dataframe_t : pandas.dataframe
      The validaiton/test dataframe to process.
    droppable_columns : list
      List of droppable column strings.
    continuous_columns : list
      List of continuous column strings.
    categorical_columns : list
      List of categorical column strings.

    Returns
    -------
    dataframe : pandas.dataframe
      Preprocessed dataframe.
    dataframe_t : pandas.dataframe
      Preprocessed dataframe_t.
    '''

    for c in droppable_columns:
        dataframe = dataframe.drop(c, axis="columns")
        dataframe_t = dataframe_t.drop(c, axis="columns")
    for c in categorical_columns:
        if c == 'Item_Fat_Content':
            dataframe[c] = dataframe[c].replace('Low Fat', 'LF').replace('low fat', 'LF').replace('reg', 'R').replace('Regular', 'R')
            dataframe_t[c] = dataframe_t[c].replace('Low Fat', 'LF').replace('low fat', 'LF').replace('reg', 'R').replace('Regular', 'R')
        if c == 'Outlet_Size':
            dataframe[c], dataframe_t[c] = MeanImputer_Size(dataframe, dataframe_t, 'Outlet_Size', 'Outlet_Identifier')            
        dataframe[c] = dataframe[c].astype("category", copy=False)
        dataframe_t[c] = dataframe_t[c].astype("category", copy=False)
    for c in continuous_columns:
        if c == 'Item_Weight':
            dataframe[c], dataframe_t[c] = MeanImputer_NA(dataframe, dataframe_t, 'Item_Weight', 'Item_Type')
        if c == 'Item_Visibility':
            dataframe[c], dataframe_t[c] = MeanImputer_0(dataframe, dataframe_t, 'Item_Visibility', 'Item_Type')
        dataframe[c], dataframe_t[c] = StandarScaler(dataframe[c].astype('float64', copy=False), dataframe_t[c].astype('float64', copy=False))
    if one_hot:
        dataframe = pd.get_dummies(dataframe, categorical_columns)
        dataframe_t = pd.get_dummies(dataframe_t, categorical_columns)
    if transform:
        return dataframe.drop(objective, axis="columns"), dataframe_t.drop(objective, axis="columns"), transform(dataframe[objective]), transform(dataframe_t[objective])
    return dataframe.drop(objective, axis="columns"), dataframe_t.drop(objective, axis="columns"), dataframe[objective], dataframe_t[objective]
