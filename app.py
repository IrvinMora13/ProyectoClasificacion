import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Función para cargar los datos con cacheo
@st.cache_data
def load_data():
    df = pd.read_csv('Wholesale customers data.csv')
    df.columns = df.columns.str.strip()  
    return df

df = load_data()

# Título de la aplicación
st.title("Clasificación con ExtraTreesClassifier")

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

with st.spinner("Entrenando el modelo..."):
    grid_search = GridSearchCV(extra_trees, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

best_trees = grid_search.best_estimator_  
y_pred = best_trees.predict(X_test)       

# Calcular métricas
accuracyET = accuracy_score(y_test, y_pred)
precisionET = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recallET = recall_score(y_test, y_pred, average='weighted')
f1ET = f1_score(y_test, y_pred, average='weighted')

# Mostrar métricas en Streamlit
st.subheader("Métricas del Modelo")
st.write(f"**Accuracy:** {accuracyET:.4f}")
st.write(f"**Precision:** {precisionET:.4f}")
st.write(f"**Recall:** {recallET:.4f}")
st.write(f"**F1 Score:** {f1ET:.4f}")

# Probabilidades y curva ROC
y_probs = best_trees.predict_proba(X_test)
auc_score = roc_auc_score(y_test, y_probs, multi_class='ovr')
fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1], pos_label=1)

st.write(f"**AUC Score:** {auc_score:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - ExtraTreesClassifier')
ax.legend(loc='lower right')
ax.grid()

st.pyplot(fig)
