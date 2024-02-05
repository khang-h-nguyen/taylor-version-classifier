from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the CSV file from the current directory
file_path = os.path.join(current_dir, 'taylor_swift_tracks.csv')
df = pd.read_csv(file_path)

label_encoder = LabelEncoder()
df['key'] = label_encoder.fit_transform(df['key'])
df = df.drop(['track_id', 'album_name', 'album_id', 'artist_name', 'artist_id', 'release_date', 'mode', 'time_signature'], axis=1)

# Normalize numerical features
numerical_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

# Create a new column 'label' based on the track name
df['label'] = df['track_name'].apply(lambda x: 1 if 'Taylor\'s Version' in x else 0)

# Features (X) and Labels (y)
X = df.drop(['label', 'track_name'], axis=1)  # Drop 'label' and 'track_name' columns
y = df['label']

# Split the data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Normalize numerical features in the tensors using the mean and std from the training set
X_train_tensor[:, :9] = (X_train_tensor[:, :9] - X_train_tensor[:, :9].mean(dim=0)) / X_train_tensor[:, :9].std(dim=0)
X_test_tensor[:, :9] = (X_test_tensor[:, :9] - X_train_tensor[:, :9].mean(dim=0)) / X_train_tensor[:, :9].std(dim=0)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the Logistic Regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Adjust based on the number of features
model = LogisticRegressionModel(input_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
num_epochs = 400
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    predictions = (model(X_test_tensor) > 0.5).float()

# Convert predictions to NumPy arrays for sklearn metrics
y_pred = predictions.numpy()
y_test_np = y_test_tensor.numpy()

# Calculate accuracy
accuracy = accuracy_score(y_test_np, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display additional metrics
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred))

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_np, y_pred))

# Extract the track names from the original DataFrame
track_names = df['track_name'].values

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'Track Name': track_names[X_test.index],  # Use X_test.index to get the corresponding track names,
    'Actual': y_test_np,
    'Predicted': y_pred.flatten()
})
print(results_df)

# Calculate precision, recall, and F1 score for each class
precision_recall_f1 = pd.DataFrame(classification_report(y_test_np, y_pred, output_dict=True)).T
print("\nPrecision, Recall, and F1 Score for Each Class:")
print(precision_recall_f1)

# Explore feature importance (coefficients for logistic regression)
if hasattr(model, 'linear') and hasattr(model.linear, 'weight'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.linear.weight.detach().numpy().flatten()
    })
    print("\nFeature Importance:")
    print(feature_importance)
