import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_variables =['Age','DailyRate','DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating','TotalWorkingYears','TotalWorkingYears','TrainingTimesLastYear','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
nonnum_variables = ['BusinessTravel','Department','EducationField','JobRole','MaritalStatus','Education','EnvironmentSatisfaction','JobLevel','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance','Gender','OverTime']
del_variables = ['EmployeeCount','EmployeeNumber','StandardHours','Over18']

preprocess_data = ColumnTransformer([
    ('std_scaler', StandardScaler(),num_variables),
    ('onehot_encoder', OneHotEncoder(drop:'first'),nonnum_variables),
    ('del_cols', 'drop',del_variables)
])

def loadpreprocess_dataset(path):
    data=pd.read_csv(path)
    preprocessed_data = preprocess_data.fit_transform(data)
    # log dataset parameters
    # mlflow.log_param("dataset_path", path)
    # mlflow.log_param("dataset_shape", data.shape)
    # mlflow.log_param("one_hot_encoding", True)
    return [preprocessed_data,data['Attrition']]
     
if __name__ == '__main__':
     print(loadpreprocess_dataset('employee-attrition.csv'))