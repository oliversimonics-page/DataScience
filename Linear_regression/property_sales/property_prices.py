import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('property_sales.csv')
df = df.drop(columns=['ID','Num_Bathrooms','Num_Floors','Year_Built','Has_Garden','Has_Pool','Garage_Size','Location_Score','Distance_to_Center'])
df['Area_Bedrooms'] = df['Square_Feet'] + df['Num_Bedrooms']*50

"""
corr = df.corr()
plt.figure(figsize=(10,6))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)

plt.title("Correlation matrix")
plt.show()
print(df.head())
"""

plt.figure(figsize=(8, 6))
plt.scatter(df['Area_Bedrooms'], df['Price'])
plt.xlabel("Area+ 50*Num_Bedrooms")
plt.ylabel("Price")
plt.title("Area vs Price scatter plot")
plt.show()


X = df[["Area_Bedrooms"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8,6))

# pontok
plt.scatter(X, y, alpha=0.4, label="Real datas")

# regressziós egyenes
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression line")

plt.xlabel("House+50*Num_Bedrooms")
plt.ylabel("Price")

plt.title("Linear regression")
plt.legend()
plt.show()
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Average deviation (MAPE): {mape:.2f}%")
