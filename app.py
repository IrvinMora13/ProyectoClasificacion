import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import streamlit as st

# Título de la aplicación
st.title("Clasificación con ExtraTrees Irvin Morales 343423")

df = pd.read_csv('/Wholesale customers data.csv')
df.columns = df.columns.str.strip()

st.write("Datos cargados:")
st.dataframe(df.head())

# Preparar los datos
X = df.drop(columns=['Region'])
y = df['Region']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = X.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

extra_trees = ExtraTreesClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
}
grid_search = GridSearchCV(extra_trees, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_trees = grid_search.best_estimator_
y_pred = best_trees.predict(X_test)

accuracyET = accuracy_score(y_test, y_pred)
precisionET = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recallET = recall_score(y_test, y_pred, average='weighted')
f1ET = f1_score(y_test, y_pred, average='weighted')

y_probs = best_trees.predict_proba(X_test)
auc = roc_auc_score(y_test, y_probs, multi_class='ovr')


fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1], pos_label=1)

st.write(f"**Accuracy:** {accuracyET:.2f}")
st.write(f"**Precision:** {precisionET:.2f}")
st.write(f"**Recall:** {recallET:.2f}")
st.write(f"**F1 Score:** {f1ET:.2f}")
st.write(f"**AUC:** {auc:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()

# Mostrar la gráfica en Streamlit
st.pyplot(plt)
