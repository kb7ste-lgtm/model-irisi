import streamlit as st
import pandas as pd
import pickle
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Iris Prediction", layout="wide")

# --- 2. CHARGEMENT DU MODÈLE (PICKLE) ---
MODEL_PATH = r"modeliris6.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Fichier modèle introuvable : {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Erreur chargement pickle : {e}")
        return None

model = load_model()

# --- 3. SIDEBAR (Image + Sliders) ---
with st.sidebar:
    st.image("https://cdn.britannica.com/39/91239-004-44353E32/Diagram-flowering-plant.jpg", caption="Parts of a Flower")
    
    st.header("User Input Parameters")
    s_length = st.slider('Sepal length', 4.0, 8.0, 5.4)
    s_width  = st.slider('Sepal width', 2.0, 5.0, 3.4)
    p_length = st.slider('Petal length', 1.0, 7.0, 1.3)
    p_width  = st.slider('Petal width', 0.1, 3.0, 0.2)

# --- 4. MAIN CONTENT ---
col_logo, col_titre = st.columns([1, 4])
with col_logo:
    st.image("https://www.ehtp.ac.ma/images/logo-ehtp.png")
with col_titre:
    st.title("Iris Flower Prediction App")
    st.write("MSDE6 : Machine Learning Course")

# Choix de la méthode d'entrée
option = st.selectbox("How would you like to use the prediction model?", 
                      ["input parameters directly", "Load a file of data"])

if option == "input parameters directly":
    input_df = pd.DataFrame({
        'sepal_length': [s_length],
        'sepal_width': [s_width],
        'petal_length': [p_length],
        'petal_width': [p_width]
    })
else:
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        st.info("Veuillez uploader un fichier CSV pour continuer.")
        st.stop()

# --- 5. RÉSULTATS ---
st.subheader("User Input Parameters:")
st.write(input_df)

st.subheader("Class labels and their corresponding index number")
st.table(pd.DataFrame({'species': ['setosa', 'versicolor', 'virginica']}))

if st.button("Predict"):
    if model is not None:
        try:
            # Prédiction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            st.subheader("Prediction")
            st.write(prediction[0])
            
            st.subheader("Prediction Probability")
            st.write(pd.DataFrame(prediction_proba, columns=['0', '1', '2']))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.write("Vérifie que les noms des colonnes du fichier CSV correspondent à ceux du modèle.")
    else:
        st.error("Le modèle n'est pas chargé.")
