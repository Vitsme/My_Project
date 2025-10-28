import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

#Loading Data from Sample Dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print(df.head())
print(f"\nShape of the dataset: {df.shape}")
print(f"Species counts:\n{df['species'].value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42)
}
results = {}
for name, model in models.items():
    if name == "Random Forest":
        model.fit(X_train, y_train)  # Use unscaled for Random Forest
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)  # Use scaled for Linear models/Distance-based
        y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

#classification report
print(classification_report(y_test, y_pred_final, target_names=target_names))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_final)
print(conf_mat)

#Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names
)
plt.title(f'Confusion Matrix for {best_model_name}')
plt.ylabel('True Species')
plt.xlabel('Predicted Species')
plt.show()