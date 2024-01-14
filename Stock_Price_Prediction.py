import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Load data
df = pd.read_csv("NFLX.csv")
viz = df.copy()
print(df.head())

# Split data into training and testing sets
train, test = train_test_split(df, test_size=0.2)
test_pred = test.copy()

# Prepare features and target variables
x_train = train[['Open', 'High', 'Low', 'Volume']].values
x_test = test[['Open', 'High', 'Low', 'Volume']].values
y_train = train['Close'].values
y_test = test['Close'].values

# Train linear regression model
model_lnr = LinearRegression()
model_lnr.fit(x_train, y_train)

# Make predictions
y_pred = model_lnr.predict(x_test)

# Evaluate the model
print("MSE", round(mean_squared_error(y_test, y_pred), 3))
print("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
print("MAE", round(mean_absolute_error(y_test, y_pred), 3))
print("MAPE", round(mean_absolute_percentage_error(y_test, y_pred), 3))
print("R2 Score : ", round(r2_score(y_test, y_pred), 3))

# Plot style
def style():
    plt.figure(facecolor='black', figsize=(15, 10))
    ax = plt.axes()
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.set_facecolor("black")

# Convert 'Date' to datetime
viz['Date'] = pd.to_datetime(viz['Date'], format='%Y-%m-%d')

# Plot closing stock price
data = pd.DataFrame(viz[['Date', 'Close']])
data = data.reset_index()
data = data.drop('index', axis=1)
data.set_index('Date', inplace=True)
data = data.asfreq('D')

style()
plt.title('Closing Stock Price', color="white")
plt.plot(viz.Date, viz.Close, color="#94F008")
plt.legend(["Close"], loc="lower right", facecolor='black', labelcolor='white')
plt.show()

# Scatter plot
style()
plt.scatter(y_pred, y_test, color='red', marker='o')
plt.scatter(y_test, y_test, color='blue')
plt.plot(y_test, y_test, color='lime')
plt.show()




