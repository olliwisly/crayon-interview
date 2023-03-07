from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

rf_grid = {
 'n_estimators': [ 10,100],
 'max_depth': [1, 5],
'criterion': ['gini', 'entropy'],
 'bootstrap': [True, False],
}

if __name__=='__main__':
    # mlflow.set_experiment('crayon-pipeline')
    # with mlflow.start_run(run_name='crayon-pipeline') as run:
    #     mlflow.log_metric("auc", auc)
    #     mlflow.rfc.log_model(rfc, "crayon-pipeline")
    #     mlflow.end_run()

    cv = 5
    RF = RandomForestClassifier(class_weight='balanced')
    [X,Y]= loadpreprocess_dataset('employee-attrition.csv')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    grid_rf_search = GridSearchCV(RF, param_grid = rf_grid, cv = cv, n_jobs = 8, verbose = 2)
    grid_rf_search.fit(X_train,Y_train)

    RF2 = RandomForestClassifier(class_weight='balanced', bootstrap=bool(grid_rf_search.best_params_['bootstrap']), criterion=grid_rf_search.best_params_['criterion'], max_depth=grid_rf_search.best_params_['max_depth'] , n_estimators=grid_rf_search.best_params_['n_estimators'] )
    
    
    
    # mlflow.log_param("cross_validation", cv)
    # mlflow.log_param("rf_gridsearch", rf_grid)
    # mlflow.log_param("dataset_shape", scoring)
    
    
    print(crossval_scores)