import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------

st.set_page_config(page_title="Clasificador MNIST", layout="wide")
st.title("🧠 Clasificador MNIST - Dígitos 0-9")

st.write("Selecciona el modelo y configura los parámetros para clasificar los dígitos escritos a mano.")

# ---------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------

@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(int)
    return X, y

X, y = load_data()

# Para que no sea absurdamente pesado
subset_size = st.sidebar.slider("Cantidad de datos a usar", 1000, 20000, 5000, step=1000)

X = X[:subset_size]
y = y[:subset_size]

# ---------------------------------------------------
# SIDEBAR - CONFIGURACIÓN
# ---------------------------------------------------

st.sidebar.header("⚙️ Configuración del Modelo")

model_option = st.sidebar.selectbox(
    "Selecciona el modelo",
    (
        "Logistic Regression",
        "SVM",
        "KNN",
        "Random Forest"
    )
)

test_size = st.sidebar.slider("Tamaño del test (%)", 10, 40, 20) / 100

use_scaling = st.sidebar.checkbox("Aplicar normalización", True)

# ---------------------------------------------------
# SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

if use_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# ---------------------------------------------------
# MODELOS
# ---------------------------------------------------

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_option == "SVM":
    model = SVC()
elif model_option == "KNN":
    k = st.sidebar.slider("Número de vecinos (K)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)
elif model_option == "Random Forest":
    n_trees = st.sidebar.slider("Número de árboles", 10, 200, 100, step=10)
    model = RandomForestClassifier(n_estimators=n_trees)

# ---------------------------------------------------
# ENTRENAMIENTO
# ---------------------------------------------------

with st.spinner("Entrenando modelo..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------------------------------------------
# MÉTRICAS
# ---------------------------------------------------

st.subheader("📊 Métricas de desempeño")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")

# ---------------------------------------------------
# MATRIZ DE CONFUSIÓN
# ---------------------------------------------------

st.subheader("🧩 Matriz de Confusión")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# ---------------------------------------------------
# VISUALIZACIÓN DE PREDICCIONES
# ---------------------------------------------------

st.subheader("🔍 Ejemplos de Predicciones")

num_examples = st.slider("Cantidad de ejemplos a mostrar", 5, 20, 10)

indices = np.random.choice(len(X_test), num_examples)

fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))

for i, idx in enumerate(indices):
    axes[i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
    axes[i].set_title(f"Pred: {y_pred[idx]}")
    axes[i].axis("off")

st.pyplot(fig)

# ---------------------------------------------------
# VISUALIZACIÓN CON PCA (2D)
# ---------------------------------------------------

st.subheader("📈 Visualización 2D con PCA")

use_pca = st.checkbox("Reducir dimensionalidad a 2D (PCA)")

if use_pca:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    df_pca = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Clase": y_test
    })

    st.scatter_chart(df_pca, x="PC1", y="PC2", color="Clase")
