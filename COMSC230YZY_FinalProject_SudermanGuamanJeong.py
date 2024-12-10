
"""
Created on Fri Nov 29 13:18:13 2024

@author: Napoleon Suderman, Gavin Jeong, Matt Guaman
"""

# Merge interest data into one csv----------------------------------------------
"""
# List of file paths
file_paths = [
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\nov17.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\nov18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\oct17.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\oct18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\sept17.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\sept18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\april18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\august18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\dec17.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\dec18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\feb18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\feb19.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\jan18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\jan19.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\july18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\june18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\march18.csv",
    "C:\\Users\\guama\\Desktop\\DataScience_FinalProject\\interest data csvs\\may18.csv"
]

# Read CSV files into dataframes
dfs = [pd.read_csv(file) for file in file_paths]

# Merge dataframes
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.fillna(value=0, inplace=True)# Filling missing values with zeros

# Save Merged Dataframe
merged_df.to_csv('merged_interestdates.csv', index=False)

"""





# Read, clean, merge interest csv and original csv------------------------------
"""
import pandas as pd
import numpy as np


# Read stockX data
df = pd.read_csv('StockX_Data.csv', delimiter=",")

# Convert Order Date and release date to datetime type
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Release Date'] = pd.to_datetime(df['Release Date'])

# New column Days since release
df['Days_since_release'] = (df['Order Date'] - df['Release Date']).dt.days

# Filter rows where dates are only yeezy
yeezydf = df[df['Brand'] == ' Yeezy']

# Convert 'Sale Price' column to numerical values
yeezydf.loc[:, 'Sale Price'] = yeezydf['Sale Price'].replace('[\$,]', '', regex=True).astype(float)





# Read merged interest data
interest_data = pd.read_csv("merged_interestdates.csv")

# Convert 'Day' column to datetime dtype
interest_data['Day'] = pd.to_datetime(interest_data['Day'])

# Merge DataFrames based on Order Date and day
merged_data = pd.merge(yeezydf, interest_data, left_on='Order Date', right_on='Day', how='left')

# Fill missing values in prenamed column with 0
merged_data['adidas Originals YEEZY Boost 350: (United States)'] = merged_data['adidas Originals YEEZY Boost 350: (United States)'].fillna(0)

# Rename prenamed column to Interest
merged_data.rename(columns={'adidas Originals YEEZY Boost 350: (United States)': 'Interest'}, inplace=True)

# Convert Interest column to integer
merged_data['Interest'] = merged_data['Interest'].astype(int)

# Drop the 'day' column
merged_data.drop('Day', axis=1, inplace=True)



# Remove some outliers
merged_data = merged_data[(merged_data['Sale Price'] >= 220) & (merged_data['Sale Price'] <= 700)]
merged_data = merged_data[(merged_data['Interest'] <= 90) & (merged_data['Interest'] > 10)]
merged_data = merged_data[(merged_data['Shoe Size'] <= 13) & (merged_data['Shoe Size'] > 5)]
merged_data = merged_data[merged_data['Days_since_release'] >= 1]

# Round down the shoe sizes to the nearest whole number
merged_data['Shoe Size'] = np.floor(merged_data['Shoe Size'])


# Display the resulting DataFrame
print(merged_data)

# Save as csv
#merged_data.to_csv('smerged_data.csv', index=False)
"""


import pandas as pd


# Read new smerged_data csv
#should be in bridges
merged_data = pd.read_csv('smerged_data.csv', delimiter=",")


# Plot histograms to find outliers----------------------------------------------
import matplotlib.pyplot as plt

# Plot histograms for numerical features
plt.figure(figsize=(12, 8))

# Sale Price histogram
plt.subplot(2, 2, 1)
plt.hist(merged_data['Sale Price'], bins=20, color='blue', alpha=0.7)
plt.title('Sale Price Distribution')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')

# Retail Price histogram
plt.subplot(2, 2, 2)
plt.hist(merged_data['Days_since_release'], bins=20, color='green', alpha=0.7)
plt.title('Days since release Distribution')
plt.xlabel('Days_since_release')
plt.ylabel('Frequency')

# Shoe Size histogram
plt.subplot(2, 2, 3)
plt.hist(merged_data['Shoe Size'], bins=20, color='orange', alpha=0.7)
plt.title('Shoe Size Distribution')
plt.xlabel('Shoe Size')
plt.ylabel('Frequency')

# Interest histogram
plt.subplot(2, 2, 4)
plt.hist(merged_data['Interest'], bins=20, color='red', alpha=0.7)
plt.title('Interest Distribution')
plt.xlabel('Interest')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()






# Regression methods------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Feature Selection
selected_features = ['Interest', 'Shoe Size', 'Days_since_release']
X = merged_data[selected_features]
y = merged_data['Sale Price']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)





#linear regression---------------------------------------------
from sklearn.linear_model import LinearRegression

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred)

# Plot actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (LinearRegression)')
plt.legend()
plt.grid(True)
plt.show()




# Decision Tree Regression--------------------------------
from sklearn.tree import DecisionTreeRegressor

# Model Training
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Model Evaluation
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)

# Plot actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_dt, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Decision Tree Regression)')
plt.legend()
plt.grid(True)
plt.show()






# Random Forest Regression------------------------------------------

from sklearn.ensemble import RandomForestRegressor

# Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# Model Evaluation
y_pred = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred)


# Plot actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()



# Print mse's
print("\n\n")
print("LinearRegression Mean Squared Error:", mse_lr)
print("Decision Tree Mean Squared Error:", mse_dt)
print("Random Forest Mean Squared Error:", mse_rf)



