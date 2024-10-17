import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import streamlit as st

# Título de la aplicación
st.title("Clasificación con ExtraTrees - Ajuste de Hiperparámetros")

# Cargar datos
df = pd.read_csv('Wholesale customers data.csv')
df.columns = df.columns.str.strip()

st.write("Datos cargados:")
st.dataframe(df.head())

features = st.multiselect("Selecciona las columnas para las características (X)", df.columns.tolist(), default=df.columns.tolist()[2:])
target = st.selectbox("Selecciona la columna objetivo (y)", df.columns.tolist(), index=1)

X = df[features]
y = df[target]

if y.dtype == 'object' or df[target].nunique() <= 10:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

X = X.apply(pd.to_numeric, errors='coerce')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.title("Ajuste de Hiperparámetros del Modelo")

n_estimators = st.sidebar.slider("Número de árboles (n_estimators)", 50, 500, 100, step=50)
max_depth = st.sidebar.slider("Profundidad máxima (max_depth)", 5, 50, None, step=5)
min_samples_split = st.sidebar.slider("Mín. muestras por división (min_samples_split)", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Mín. muestras por hoja (min_samples_leaf)", 1, 5, 1)
bootstrap = st.sidebar.checkbox("¿Usar Bootstrap?", value=True)

extra_trees = ExtraTreesClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    bootstrap=bootstrap,
)


extra_trees.fit(X_train, y_train)
y_pred = extra_trees.predict(X_test)


accuracyET = accuracy_score(y_test, y_pred)
precisionET = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recallET = recall_score(y_test, y_pred, average='weighted')
f1ET = f1_score(y_test, y_pred, average='weighted')

y_probs = extra_trees.predict_proba(X_test)

if len(set(y_test)) == 2:
    auc = roc_auc_score(y_test, y_probs[:, 1])
    fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
else:
    auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
    fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1], pos_label=1)

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

st.pyplot(plt)
