# Read the data into pandas dataframe
import pandas as pd
data = pd.read_parquet("live/weather.parquet")

#%%

X = data[["temperature", "wind", "precipitation"]]
y = data[["slippery"]]

# Make train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)


# Use decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prediction)
print(f"Accuracy: {accuracy}")

# Plot the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 20))
plot_tree(model, feature_names = X.columns, class_names = ["Not slippery", "Slippery"], filled=True)
plt.show()