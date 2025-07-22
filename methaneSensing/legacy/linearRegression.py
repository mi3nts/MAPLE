import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Config ---
experiment_names = ["exp1", "exp2", "exp3"]
nodeID = "001e064a872f"
output_folder = "filteredExperiments"

# --- Set pandas display options ---
pd.set_option('display.max_columns', None)


# --- Loop through experiments ---
for experiment_name in experiment_names:
    
    save_path = f"{output_folder}/{experiment_name}_{nodeID}_filtered_train_test_split_data.pkl"
    
    if not os.path.exists(save_path):
        print(f"[ERROR] File not found: {save_path}")
        continue  # Skip to next experiment

    loaded_data = joblib.load(save_path)
    print(f"[loaded] {save_path}")

    X_train = loaded_data['X_train']
    y_train = loaded_data['y_train']
    X_test  = loaded_data['X_test']
    y_test  = loaded_data['y_test']
    X   = loaded_data['X']
    y   = loaded_data['y']


    print(X_train.head())



    # # Normalize the features
    # scaler         = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled  = scaler.transform(X_test)
    # X_all_scaled   = scaler.fit_transform(X_all)

    # # Define the Linear Regression model
    # print("Training Linear Regression")
    # linear_reg_model = LinearRegression()

    # # Train the Linear Regression model
    # linear_reg_model.fit(X_train_scaled, y_train)

    # # Predict on training and test data

    # y_pred_train = linear_reg_model.predict(X_train_scaled)
    # y_pred_test  = linear_reg_model.predict(X_test_scaled)
    # y_pred_all   = linear_reg_model.predict(X_all_scaled)

    # # Evaluate the training data set
    # r2Train  = r2_score(y_train, y_pred_train)
    # mseTrain = mean_squared_error(y_train, y_pred_train)


    # # print("Best Hyperparameters:", random_search.best_params_)
    # print("R^2 Score Train         :", r2Train)
    # print("Mean Squared Error Train:", mseTrain)


    # # Evaluate the optimized model
    # r2Test  = r2_score(y_test, y_pred_test)
    # mseTest = mean_squared_error(y_test, y_pred_test)


    # # print("Best Hyperparameters:", random_search.best_params_)
    # print("R^2 Score Test         :", r2Test)
    # print("Mean Squared Error Test:", mseTest)


    # # Evaluate for the complete data set 
    # r2All = r2_score(y_all, y_pred_all)
    # mseAll = mean_squared_error(y_all, y_pred_all)


    # # print("Best Hyperparameters:", random_search.best_params_)
    # print("R^2 Score All         :", r2All)
    # print("Mean Squared Error All:", mseAll)

    # # Save results and model
    # data_to_save_with_ml = {
    # 'r2Train': r2Train,
    # 'rmseTrain':mseTrain,
    # 'r2Test': r2Test,
    # 'rmseTest': mseTest,
    # 'r2All': r2All,
    # 'rmseAll': mseAll,
    # 'best_model': linear_reg_model
    # }

    # import datetime
    # current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # # Generate the filename with the datetime
    # filename = f"dataSetsWithML_LR_{fileID}_{current_datetime}.pkl"

    # joblib.dump(data_to_save_with_ml, filename)
    # print(f"Data with ML saved successfully as {filename}!")