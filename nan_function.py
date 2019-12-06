
from scipy.io import arff
import pandas as pd

def nan_check(dataframe):
    #the dataframe has to be a pandas dataframe
    #find all nans in dataframe
    dataframe_nans = dataframe.isnull().sum().sum()
    
    if (dataframe_nans/len(dataframe.index) < 0.1):
        dataframe = dataframe.dropna()
    
    return dataframe

#test run

#data_adolescent = (arff.loadarff('Autism-Adolescent-Data.arff'))
#data_child = (arff.loadarff('Autism-Child-Data.arff'))
#dataframe_adolescent = pd.DataFrame(data_adolescent[0])
#dataframe_child = pd.DataFrame(data_child[0])
#print (dataframe_child)
#
#dataframe_nans = nan_check(dataframe_child)
#print (dataframe_nans)
#    