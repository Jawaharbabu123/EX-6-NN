<H3>ENTER YOUR NAME : JAWAHAR BABU S</H3>
<H3>ENTER YOUR REGISTER NO.:212224220041</H3>
<H3>EX. NO.6</H3>
<H3>DATE:25.05.2026</H3>
<H1 ALIGN =CENTER>Heart attack prediction using MLP</H1>
<H3>Aim:</H3>  To construct a  Multi-Layer Perceptron to predict heart attack using Python
<H3>Algorithm:</H3>
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.<BR>
Step 2:Load the heart disease dataset from a file using pd.read_csv().<BR>
Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).<BR>
Step 4:Split the dataset into training and testing sets using train_test_split().<BR>
Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.<BR>
Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.<BR>
Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.<BR>
Step 8:Make predictions on the testing set using mlp.predict(X_test).<BR>
Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().<BR>
Step 10:Print the accuracy of the model.<BR>
Step 11:Plot the error convergence during training using plt.plot() and plt.show().<BR>
<H3>Program: </H3>

``` python
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("heart.csv")

# Display first few rows
print(data.head())

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    max_iter=1000,
    random_state=42
)

# Train the model
mlp.fit(X_train, y_train)

# Predict using test data
y_pred = mlp.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy of MLP Model:", accuracy * 100, "%")

# Plot loss curve
plt.plot(mlp.loss_curve_)
plt.title("Error Convergence during Training")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```

<H3>Output:</H3>

<img width="902" height="301" alt="image" src="https://github.com/user-attachments/assets/c0070d9f-d4cf-4b39-8224-f42ba9b7986f" />
<img width="993" height="552" alt="image" src="https://github.com/user-attachments/assets/1a576c54-6958-4ac6-8322-1582aea6a85a" />

<H3>Results:</H3>
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
