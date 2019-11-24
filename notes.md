# Notes
## Preprocessing
* See `dataset.preprocessing()` function
    * Renamed misspelled column names
    * Used sklearn LabelEncoder for converting nominative values to numerical values
    * Converted boolean and columns with only two values to 0 and 1
* *As of now, includes '?' in encoding*
* Add continent column

## Imputation
* For ethnicity (assume mean == mode):
    * Try mode of country_of_res
    * Otherwise, mode of continent
    
* Try with assumption that we can ignore missing data, otherwise,
* Possible methods if necessary:
    * Mode
    * Learn/predict
    
## Models
 Based on (find Hsu's paper on number of models), we'll use 3 models for Model Fusion:
 * Knn
 * Random Forest
 * SVM