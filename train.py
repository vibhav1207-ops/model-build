import os
import pickle
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
os.makedirs("saved_model", exist_ok=True)
with open("saved_model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved at saved_model/model.pkl")