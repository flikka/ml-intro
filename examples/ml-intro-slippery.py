import pandas as pd

df = pd.read_parquet("weather.parquet")

#%%
# Split the data into training and test data
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, train_size=.5)

# Train a simple decision tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(train[["temperature", "wind", "precipitation"]], train["slippery"])

# Predict on the test data
predictions = model.predict(test[["temperature", "wind", "precipitation"]])

# Evaluate the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test["slippery"], predictions)
print(f"Achieved accuracy: {accuracy}")

# Plot the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=["temperature", "wind", "precipitation"], filled=True)