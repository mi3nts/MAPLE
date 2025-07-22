
from matplotlib import font_manager
import matplotlib.font_manager as fm
import matplotlib as mpl
from cycler import cycler
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#Define plotting style
montserrat_path = "Montserrat,Sankofa_Display/Montserrat/static"
font_files = font_manager.findSystemFonts(fontpaths=montserrat_path)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)


mpl.rcParams.update({
    'font.family': 'Montserrat',
    'font.size': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.minor.size': 1,
    'ytick.minor.size': 1,
    'grid.alpha': 0.5,
    'axes.grid': True,
    'grid.linewidth': 2,
    'axes.grid.which': 'both',
    'axes.titleweight': 'bold',
    'axes.titlesize': 18,
    'axes.prop_cycle': cycler('color', ['#3cd184', '#f97171', '#1e81b0', '#66beb2', '#f99192', '#8ad6cc', '#3d6647', '#000080']),
    'image.cmap': 'viridis',
})

# Scatter Plot 

fileID = "20250129_175241"

# Load the data for scatter plots
loaded_data = joblib.load("dataSetsWithML" + "_" + fileID+ ".pkl")

fileIDModel = "ORF_20250129_175241_20250129_190840"

# Load the data for scatter plots
loadedMdl_data = joblib.load("dataSetsWithML" + "_" + fileIDModel+ ".pkl")

# Get information on the model 
best_model = loadedMdl_data['best_model']

# print(best_model.get_params())

X_train = loaded_data['X_train']
y_train = loaded_data['y_train']

X_test  = loaded_data['X_test']
y_test  = loaded_data['y_test']

X_all   = loaded_data['X_all']
y_all   = loaded_data['y_all']

X       = loaded_data['X']
y       = loaded_data['y']

print("MIN MAX")
print(min(y_all))
print(max(y_all))

# Normalize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_all_scaled   = scaler.fit_transform(X_all)

# Predict on training, test, and complete data
y_pred_train = best_model.predict(X_train_scaled)
y_pred_test  = best_model.predict(X_test_scaled)
y_pred_all   = best_model.predict(X_all_scaled)

# Evaluate for the training dataset
r2Train = r2_score(y_train, y_pred_train)
mseTrain = mean_squared_error(y_train, y_pred_train)

print("R^2 Score Train         :", r2Train)
print("Mean Squared Error Train:", mseTrain)

# Evaluate the test dataset
r2Test = r2_score(y_test, y_pred_test)
mseTest = mean_squared_error(y_test, y_pred_test)

print("R^2 Score Test         :", r2Test)
print("Mean Squared Error Test:", mseTest)

# Evaluate for the complete dataset
r2All = r2_score(y_all, y_pred_all)
mseAll = mean_squared_error(y_all, y_pred_all)

print("R^2 Score All         :", r2All)
print("Mean Squared Error All:", mseAll)


n_train   = len(y_pred_train)
n_test    = len(y_pred_test)

## FOR GRAPHING 
# BAMWithCorrectedCleaned = loaded_data['BAMWithCorrectedCleaned']

# print("Data with ML loaded successfully!")

# # Sort the resulting DataFrames and Series by index (timestamp)
# X_train = X_train.sort_index()
# y_train = y_train.sort_index()

# X_test  = X_test.sort_index()
# y_test  = y_test.sort_index()

# X_all  = X_all.sort_index()
# y_all  = y_all.sort_index()

# train_indices = np.where(X.index.isin(X_train.index))[0]
# test_indices  = np.where(X.index.isin(X_test.index))[0]
# all_indices   = np.where(X.index.isin(X_all.index))[0]


# BAMWithCorrectedCleanedTrain = BAMWithCorrectedCleaned.iloc[train_indices] 
# BAMWithCorrectedCleanedTest  = BAMWithCorrectedCleaned.iloc[test_indices] 
# BAMWithCorrectedCleanedAll   = BAMWithCorrectedCleaned.iloc[all_indices] 

# BAMWithCorrectedCleanedTrain.loc[:, 'pm2_5ML'] = y_pred_train
# BAMWithCorrectedCleanedTest.loc[:, 'pm2_5ML']  = y_pred_test
# BAMWithCorrectedCleanedAll.loc[:, 'pm2_5ML']   = y_pred_all


# r2_train = r2_score(BAMWithCorrectedCleanedTrain.pm2_5BAM, BAMWithCorrectedCleanedTrain.pm2_5ML)
# r2_test  = r2_score(BAMWithCorrectedCleanedTest.pm2_5BAM, BAMWithCorrectedCleanedTest.pm2_5ML)

# n_train   = len(BAMWithCorrectedCleanedTrain)
# n_test    = len(BAMWithCorrectedCleanedTest)

# mseTrain  = mean_squared_error(BAMWithCorrectedCleanedTrain.pm2_5BAM, BAMWithCorrectedCleanedTrain.pm2_5ML)
# mseTest   = mean_squared_error(BAMWithCorrectedCleanedTest.pm2_5BAM, BAMWithCorrectedCleanedTest.pm2_5ML)


# print("R2 Train")
# print(r2_train)

# print("R2 Test")
# print(r2_test)

# print("RMSE Train")
# print(mseTrain)

# print("RMSE Test")
# print(mseTest)




plt.figure(figsize=(10, 10))

plt.xlim(0, 100)
plt.ylim(0, 100)

# Set log scale for both axes
# plt.xscale('log')
# plt.yscale('log')

# # Set axis limits
# plt.xlim(0.1, 100)
# plt.ylim(0.1, 100)

# Plot training set scatter plot with R² and number of points
plt.scatter(y_train, y_pred_train,
            color='#1e81b0', label=f'Training (n={n_train}, R²={r2Train:.2f})')

# Plot testing set scatter plot with R² and number of points
plt.scatter(y_test, y_pred_test,
     marker='+',label=f'Independent Validation (n={n_test}, R²={r2Test:.2f})')

# Plot y=x reference line
x_vals = np.linspace(0.1, 100, 1000)  # Adjusted to avoid log scale issues at zero
plt.plot(x_vals, x_vals, color='#f97171', linestyle='--', linewidth=2, label='y=x Line')



# Add labels and title for scatter plots
plt.xlabel('True PM$_{2.5}$ ($\mu g/m^3$)', fontsize=25)
plt.ylabel('Estimated PM$_{2.5}$ ($\mu g/m^3$)', fontsize=25)
plt.title('PM$_{2.5}$ Scatter Plot', fontsize=25, pad=20)
# Configure grid and ticks
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Add legend
plt.legend(fontsize=20)

# Save the plot
plt.savefig('scatterPlotsV3_' + fileIDModel +'.png', dpi=300)
plt.close()

####################
# QQ Plot 

# Calculate quantiles
quantiles = [0.25, 0.5, 0.75]
quantile_values_var1 = np.quantile(y_test, quantiles)
quantile_values_var2 = np.quantile(y_pred_test, quantiles)

# Q-Q plot Log Scale 
plt.figure(figsize=(10, 10))
# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

plt.scatter(np.sort(y_test),
            np.sort(y_pred_test),
            alpha=0.75)

# Plot y=x line on log scale
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(),y_pred_test.max())

# Ensure that min_val is not zero for log scale
min_val = max(min_val,0.0000001)  # Adjust this value as needed to avoid log(0)

plt.plot([min_val, max_val], [min_val, max_val], color='#f97171', linestyle='--',linewidth=2)


# plt.plot([BAMWithCorrectedCleanedTest['pm2_5BAM'].min(), BAMWithCorrectedCleanedTest['pm2_5BAM'].max()],
#          [BAMWithCorrectedCleanedTest['pm2_5ML'].min(), BAMWithCorrectedCleanedTest['pm2_5ML'].max()],
#          color='#f97171', linestyle='--')

# Mark quantiles
for q, val1, val2 in zip(quantiles, quantile_values_var1, quantile_values_var2):
    plt.scatter(val1, val2, color='#f97171')
    plt.text(val1, val2, f'Q{int(q*100)}', fontsize=20, color='#f97171', verticalalignment='bottom', horizontalalignment='right')

# Set x and y limits between 1 and 100
plt.xlim(1, 100)
plt.ylim(1, 100)

plt.xlabel('True PM$_{2.5}$ ($\mu g/m^3$)', fontsize=25)
plt.ylabel('Estimated PM$_{2.5}$ ($\mu g/m^3$)', fontsize=25)
plt.title('PM$_{2.5}$ Quantile-Quantile Plot', fontsize=25, pad=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('QQPlotLog_' + fileIDModel +'.png', dpi=300)
plt.close()

quantiles = [0, 0.25, 0.5, 0.75,1]
quantile_values_var1 = np.quantile(y_test, quantiles)
quantile_values_var2 = np.quantile(y_pred_test, quantiles)


############
# Q-Q plot
plt.figure(figsize=(10, 10))
plt.scatter(np.sort(y_test),
            np.sort(y_pred_test),
            alpha=0.75)
plt.plot([y_test.min(), y_test.max()],
         [y_pred_test.min(), y_pred_test.max()],
         color='#f97171', linestyle='--')
# plt.xlim(0, 95)
# plt.ylim(0, 95)
# Mark quantiles
for q, val1, val2 in zip(quantiles, quantile_values_var1, quantile_values_var2):
    plt.scatter(val1, val2, color='#f97171',linewidth=2)
    plt.text(val1, val2, f'Q{int(q*100)}', fontsize=20, color='#f97171', verticalalignment='bottom', horizontalalignment='right')

plt.xlabel('True PM$_{2.5}$ ($\mu g/m^3$)', fontsize=25)
plt.ylabel('Estimated PM$_{2.5}$ ($\mu g/m^3$)', fontsize=25)
plt.title('PM$_{2.5}$ Quantile-Quantile Plot', fontsize=25, pad=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('QQPlot_' + fileIDModel +'.png', dpi=300)
plt.close()


##### Predictor Importance 
# Load the train/test split data
loaded_data = joblib.load('train_test_split_data.pkl')
X_train = loaded_data['X_train']
X_test = loaded_data['X_test']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']
train_indices = loaded_data['train_indices']
test_indices = loaded_data['test_indices']

# Load the best model
best_model = joblib.load('best_random_forest_model.joblib')
print("Model loaded successfully!")

## Get feature importances
feature_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)
feature_importances = feature_importances.sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 10))
# ax = feature_importances.plot(kind='barh')
ax = feature_importances.plot(kind='barh', color=[ '#1e81b0' if i > 2 else '#3cd184' for i in range(len(feature_importances))])

plt.title('PM$_{2.5}$ Predictor Importance Estimates', fontsize=25, pad=20)
plt.xlabel('Estimated Importance', fontsize=25)
plt.ylabel('Predictors', fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, .3)

# Invert y-axis to have the most important features at the top
plt.gca().invert_yaxis()

# Annotate the bars with their importance values and feature labels at the end of each bar
# for i, (value, name) in enumerate(zip(feature_importances, feature_importances.index)):
    # ax.text(value + 0.01, i, f'{name} ({value:.4f})', va='center', ha='left', fontsize=20, color='black')


def replace_particle_counts(name):
    # Replace 'pc' with 'PC' and '_2_5' with '2.5 µm', also adjust other particle sizes
    name = name.replace('pc', 'PC ')
    name = name.replace('_', '.')  # Replace underscore with period for readability
    name = name.replace('HC', ' µm')  # Optional: add space after 'HC' for readability
    name = name.replace('temperature', 'Temperature')
    name = name.replace('pressure', 'Pressure')  # Optional: add space after 'HC' for readability
    name = name.replace('humidity', 'Humidity')  # Optional: add space after 'HC' for readability
    name = name.replace('dewPoint', 'Dew Point')  # Optional: add space after 'HC' for readability

    return name

num_labels = len(feature_importances)

feature_importances.index = feature_importances.index.map(replace_particle_counts)

print(feature_importances)

for i, value in enumerate(feature_importances):
    name = feature_importances.index[i]  # Access the feature name using the index
    ax.text(value + 0.001, i, f'{name}', va='center', ha='left', fontsize=20, color='black')

plt.yticks(ticks=range(num_labels), labels=[str(i+1) for i in range(num_labels)], fontsize=20)

plt.tight_layout()  # Adjust layout to prevent clipping


# Save and close the plot
plt.savefig('predImp_' + fileIDModel +'.png', dpi=300)
plt.close()



# Time Series Plot 
# # Load the DataFrame from the pickle file
withLikelyHoodIPSBME5MinWithHCPCPMML= pd.read_pickle('withLikelyHoodIPSBME5MinWithHCPCPMML.pkl')

print(withLikelyHoodIPSBME5MinWithHCPCPMML.head())


# Plot the time series
plt.figure(figsize=(16, 9))
plt.plot(withLikelyHoodIPSBME5MinWithHCPCPMML.index, withLikelyHoodIPSBME5MinWithHCPCPMML['pm2_5'], label='IPS7100 PM$_{2.5}$')
plt.plot(withLikelyHoodIPSBME5MinWithHCPCPMML.index, withLikelyHoodIPSBME5MinWithHCPCPMML['pm2_5HC'], label='Humidity Corrected PM$_{2.5}$',color='#1e81b0')
plt.plot(withLikelyHoodIPSBME5MinWithHCPCPMML.index, withLikelyHoodIPSBME5MinWithHCPCPMML['pm2_5ML'], label='Machine Learning Corrected PM$_{2.5}$',color='#f97171')

plt.xlim(pd.Timestamp('2024-10-01'), pd.Timestamp('2024-10-31'))  # Example x-limit
plt.ylim(0, 100)  # Example y-limit

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Rotate x-axis labels
plt.xticks(rotation=45)

plt.title('PM$_{2.5}$ Time Series Plot',fontsize=25, pad=20)
plt.xlabel('Date Time (UTC)', fontsize=25)
plt.ylabel(r'PM$_{2.5}$ ($\mu g/m^3$)',fontsize=25)
plt.legend()
plt.tight_layout()
plt.savefig('timeSeries_' + fileIDModel +'.png', dpi=300)
plt.close()