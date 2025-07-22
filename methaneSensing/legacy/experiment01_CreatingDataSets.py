# normalize_calibrated.py
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Config ---
experiment_name = "exp1"
nodeID = "001e064a872f"
output_folder = "filteredExperiments"
filtered_file = f"{output_folder}/{experiment_name}_{nodeID}_filtered.pkl"

# --- Load Data ---
if not os.path.exists(filtered_file):
    print(f"[ERROR] File not found: {filtered_file}")
    exit()

df = pd.read_pickle(filtered_file)

# --- Define methane sensor columns ---
ref_col = 'methane_GSR001ACON'
low_cost_cols = [
    'methane_INIR2ME5',
    'methane_SJH5',
    'methaneEQBusVoltage_TGS2611C00'
]

# Drop rows with NaNs
df_clean = df[[ref_col] + low_cost_cols].dropna()

# Initialize output DataFrame
df_calibrated = pd.DataFrame(index=df_clean.index)
df_calibrated['reference'] = df_clean[ref_col]


# Creating Data Sets for Linear Regression

# # Check for NaNs in the entire DataFrame
# X_selected = BAMWithCorrectedCleaned[['temperature', 'pressure', 'humidity', 'dewPoint',
#                                        'pc0_1HC', 'pc0_3HC', 'pc0_5HC', 
#                                        'pc1_0HC', 'pc2_5HC', 'pc5_0HC', 
#                                        'pc10_0HC','pm0_1HC', 'pm0_3HC',
#                                        'pm0_5HC', 'pm1_0HC', 'pm2_5HC',
#                                        'pm5_0HC', 'pm10_0HC','pm2_5BAM']]

# X = X_selected.drop(columns=['pm2_5BAM'])  # Replace 'target_column' with your actual target column name
# y = X_selected['pm2_5BAM']  # Target column

# X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)

# # Sort the resulting DataFrames and Series by index (timestamp)
# X_train = X_train.sort_index()
# X_test  = X_test.sort_index()
# y_train = y_train.sort_index()
# y_test  = y_test.sort_index()

# print("Are X indices unique?",       X.index.is_unique)
# print("Are X_train indices unique?", X_train.index.is_unique)
# print("Are X_test indices unique?",  X_test.index.is_unique)

# print("Index alignment in train:", set(X_train.index).issubset(X.index))
# print("Index alignment in test:", set(X_test.index).issubset(X.index))

# train_indices = np.where(X.index.isin(X_train.index))[0]
# test_indices = np.where(X.index.isin(X_test.index))[0]

# BAMWithCorrectedCleanedTrain = BAMWithCorrectedCleaned.iloc[train_indices] 
# BAMWithCorrectedCleanedTest  = BAMWithCorrectedCleaned.iloc[test_indices] 

# # Training Just one model
# print("Training the best model")
# best_model = RandomForestRegressor(n_estimators=100, max_features=1.0)
# best_model.fit(X_train, y_train)

# # Set up the parameter grid for random search
# param_distributions = {
#     'n_estimators': randint(50, 200),        # Number of trees
#     'max_depth': randint(5, 30),             # Maximum depth of each tree
#     'min_samples_split': randint(2, 20),     # Minimum number of samples required to split a node
#     'min_samples_leaf': randint(1, 10),      # Minimum number of samples required to be at a leaf node
#     'max_features': ['auto', 'sqrt', 'log2'] # Number of features to consider at each split
# }

# # Make predictions on the test set
# y_pred_all   = best_model.predict(X)
# y_pred_test  = best_model.predict(X_test)
# y_pred_train = best_model.predict(X_train)

# # Evaluate for the complete data set 
# mseAll = mean_squared_error(y, y_pred_all)
# r2All = r2_score(y, y_pred_all)

# # print("Best Hyperparameters:", random_search.best_params_)
# print("Mean Squared Error All:", mseAll)
# print("R^2 Score All         :", r2All)

# # Evaluate the training data set
# mseTrain = mean_squared_error(y_train, y_pred_train)
# r2Train  = r2_score(y_train, y_pred_train)

# # print("Best Hyperparameters:", random_search.best_params_)
# print("Mean Squared Error Train:", mseTrain)
# print("R^2 Score Train         :", r2Train)

# # Evaluate the optimized model
# mseTest = mean_squared_error(y_test, y_pred_test)
# r2Test  = r2_score(y_test, y_pred_test)

# # print("Best Hyperparameters:", random_search.best_params_)
# print("Mean Squared Error Test:", mseTest)
# print("R^2 Score Test         :", r2Test)

# data_to_save = {
#     'X_train': X_train,
#     'X_test': X_test,
#     'y_train': y_train,
#     'y_test': y_test,
#     'train_indices': train_indices ,
#     'test_indices': test_indices,
# }

# joblib.dump(data_to_save, 'train_test_split_data.pkl')
# print("Data saved successfully!")

# joblib.dump(best_model, 'best_random_forest_model.joblib')
# print("Model saved successfully!")

# BAMWithCorrectedCleanedTest.loc[:, 'pm2_5ML']  = y_pred_test
# BAMWithCorrectedCleanedTrain.loc[:, 'pm2_5ML'] = y_pred_train
# BAMWithCorrectedCleaned.loc[:, 'pm2_5ML']      = y_pred_all

# data_to_save_with_ml = {
#     'BAMWithCorrectedCleanedWithMLTest': BAMWithCorrectedCleanedTest,
#     'BAMWithCorrectedCleanedWithMLTrain': BAMWithCorrectedCleanedTrain,
#     'BAMWithCorrectedCleanedWithMLAll': BAMWithCorrectedCleaned
# }

# joblib.dump(data_to_save_with_ml, 'dataSetsWithML.pkl')
# print("Data with ML saved successfully!")















# # Fit each low-cost sensor to the reference using linear regression
# for col in low_cost_cols:
#     X = df_clean[[col]]
#     y = df_clean[ref_col]

#     model = LinearRegression()
#     model.fit(X, y)
#     predicted = model.predict(X)

#     calibrated_col = col.replace('methane_', '') + '_calibrated'
#     df_calibrated[calibrated_col] = predicted

# # --- Save calibrated output ---
# output_path = os.path.join(output_folder, f"{experiment_name}_{nodeID}_filtered_calibrated.pkl")
# df_calibrated.to_pickle(output_path)

# print(f"[SAVED] Linear-regression calibrated data to: {output_path}")
