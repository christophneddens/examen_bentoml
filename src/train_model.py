import numpy as np
import pandas as pd
import bentoml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Load the prepared data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = np.ravel(pd.read_csv("data/processed/y_train.csv"))
y_test = np.ravel(pd.read_csv("data/processed/y_test.csv"))

# 2. Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Test the model on the test set
y_pred = model.predict(X_test)
print(f"R2   : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")

# 4. Save the model in the BentoML Model Store
model_ref = bentoml.sklearn.save_model("admission_lr", model)
print(f"Model saved as: {model_ref}")
