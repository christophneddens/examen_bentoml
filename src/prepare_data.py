import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the raw data
df = pd.read_csv("data/raw/admission.csv")

# 2. Clean the data
df.columns = df.columns.str.strip()        # remove spaces around the column names
df = df.drop(columns=["Serial No."])       # an ID number is not useful for the model
df = df.dropna()                           # remove rows with missing values

# 3. Features (X) and target (y)
X = df.drop(columns=["Chance of Admit"])
y = df["Chance of Admit"]

# 4. Split: 80 % training, 20 % test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Save the four files in data/processed
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Columns used:", list(X.columns))
print("X_train:", X_train.shape, " X_test:", X_test.shape)
