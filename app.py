import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import requests

# Configuration de la page
st.set_page_config(
    page_title="Gradient Descent ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le design
st.markdown("""
<style>
/* Fonts and body reset */
body, .stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #0d1b2a;
    color: #e0e1dd;
}

/* Header */
.main-header {
    background: linear-gradient(90deg, #0d1b2a 0%, #1b263b 100%);
    padding: 2rem;
    border-radius: 10px;
    color: #e0e1dd;
    text-align: center;
    margin: 2rem 0 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* Metric Cards */
.metric-container {
    background: linear-gradient(135deg, #415a77 0%, #778da9 100%);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    color: #e0e1dd;
    transition: all 0.3s ease-in-out;
}
.metric-container:hover {
    background: #1b263b;
    transform: scale(1.02);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #778da9;
    border-radius: 10px 10px 0 0;
    color: #0d1b2a;
    transition: background-color 0.3s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #415a77;
    color: #e0e1dd;
}
.stTabs [aria-selected="true"] {
    background-color: #1b263b;
    color: #e0e1dd;
}

/* Primary button */
button[kind="primary"] {
    background-color: #415a77 !important;
    color: #e0e1dd !important;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease-in-out;
}
button[kind="primary"]:hover {
    background-color: #1b263b !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #0d1b2a;
}
::-webkit-scrollbar-thumb {
    background: #415a77;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #1b263b;
}
</style>
""", unsafe_allow_html=True)


class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000, task_type='regression'):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.task_type = task_type
        self.weights = None
        self.bias = None
        self.costs = []
        
    def sigmoid(self, z):
        """Fonction sigmoïde pour la classification"""
        z = np.clip(z, -500, 500)  # Éviter l'overflow
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """Entraîner le modèle"""
        n_samples, n_features = X.shape
        
        # Initialiser les paramètres
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        self.costs = []
        
        for i in range(self.max_iterations):
            # Prédiction
            linear_pred = np.dot(X, self.weights) + self.bias
            
            if self.task_type == 'classification':
                y_pred = self.sigmoid(linear_pred)
                # Coût (log-loss)
                cost = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
                # Gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
            else:  # regression
                y_pred = linear_pred
                # Coût (MSE)
                cost = np.mean((y_pred - y) ** 2)
                # Gradients
                dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (2 / n_samples) * np.sum(y_pred - y)
            
            self.costs.append(cost)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        """Faire des prédictions"""
        linear_pred = np.dot(X, self.weights) + self.bias
        
        if self.task_type == 'classification':
            return (self.sigmoid(linear_pred) >= 0.5).astype(int)
        else:
            return linear_pred
    
    def predict_proba(self, X):
        """Probabilités pour la classification"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
 

    

def download_dataset(url):
    """Télécharger un dataset depuis une URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        st.error(f"Erreur lors du téléchargement: {e}")
        return None

def preprocess_data(df, target_column, task_type):
    """Prétraitement des données"""
    df_processed = df.copy()
    
    # Traiter les valeurs manquantes
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown', inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Encoder les variables catégorielles (sauf la variable cible)
    label_encoders = {}
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' and col != target_column:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Traiter la variable cible
    if df_processed[target_column].dtype == 'object':
        # Pour les variables catégorielles, toujours encoder
        le_target = LabelEncoder()
        df_processed[target_column] = le_target.fit_transform(df_processed[target_column].astype(str))
        label_encoders['target'] = le_target
    else:
        # Pour les variables numériques, s'assurer qu'elles sont bien numériques
        df_processed[target_column] = pd.to_numeric(df_processed[target_column], errors='coerce')
        # Remplacer les NaN par la médiane
        df_processed[target_column].fillna(df_processed[target_column].median(), inplace=True)
    
    # Convertir toutes les colonnes en float pour éviter les erreurs de type
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    return df_processed, label_encoders

def create_dashboard(df):
    """Créer un dashboard des statistiques"""
    st.markdown('<div class="main-header"><h1>📊 Dashboard des Statistiques</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Nombre de lignes", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Nombre de colonnes", len(df.columns))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Valeurs manquantes", df.isnull().sum().sum())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Variables numériques", len(df.select_dtypes(include=[np.number]).columns))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualisations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Distribution des variables numériques")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Choisir une variable:", numeric_cols)
            fig = px.histogram(df, x=selected_col, nbins=30, 
                             title=f"Distribution de {selected_col}")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔗 Matrice de corrélation")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, 
                          title="Matrice de corrélation",
                          color_continuous_scale="RdBu_r",
                          aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques descriptives
    st.subheader("📋 Statistiques descriptives")
    st.dataframe(df.describe(), use_container_width=True)

def main():
    st.markdown('<div class="main-header"><h1>🧠 Gradient Descent ML </h1><p>Application  pour la régression et la classification</p></div>', unsafe_allow_html=True)
    
    # Sidebar pour les paramètres
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Données", "📊 Dashboard", "🔧 Prétraitement", "🤖 Modèle", "📈 Résultats"])
    
    # Initialiser les variables de session
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    with tab1:
        st.header("📁 Chargement des données")
        
        # Option de chargement
        data_option = st.radio("Choisir une option:", 
                              ["Télécharger un fichier",  "URL personnalisée"])
        
        if data_option == "Télécharger un fichier":
            uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
            if uploaded_file is not None:
                try:
                    st.session_state.df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Dataset chargé avec succès! ({len(st.session_state.df)} lignes, {len(st.session_state.df.columns)} colonnes)")
                    st.dataframe(st.session_state.df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement: {e}")
        
        
        else:  # URL personnalisée
            url = st.text_input("Entrer l'URL du dataset CSV:")
            if url and st.button("Télécharger"):
                with st.spinner("Téléchargement en cours..."):
                    st.session_state.df = download_dataset(url)
                    if st.session_state.df is not None:
                        st.success("✅ Dataset téléchargé avec succès!")
                        st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    with tab2:
        if st.session_state.df is not None:
            create_dashboard(st.session_state.df)
        else:
            st.info("📁 Veuillez d'abord charger un dataset dans l'onglet 'Données'")
    
    with tab3:
        st.header("🔧 Prétraitement des données")
        
        if st.session_state.df is not None:
            st.subheader("Configuration du prétraitement")
            
            # Choisir la variable cible
            target_column = st.selectbox("Choisir la variable cible (y):", 
                                       st.session_state.df.columns)
            
            # Choisir le type de tâche
            task_type = st.radio("Type de tâche:", ["regression", "classification"])
            
            # Variables explicatives
            feature_columns = st.multiselect("Choisir les variables explicatives (X):",
                                           [col for col in st.session_state.df.columns if col != target_column],
                                           default=[col for col in st.session_state.df.columns if col != target_column][:5])
            
            if st.button("Appliquer le prétraitement"):
                if feature_columns:
                    # Sélectionner les colonnes
                    df_selected = st.session_state.df[feature_columns + [target_column]].copy()
                    
                    # Prétraitement
                    st.session_state.df_processed, label_encoders = preprocess_data(
                        df_selected, target_column, task_type
                    )
                    
                    st.session_state.target_column = target_column
                    st.session_state.feature_columns = feature_columns
                    st.session_state.task_type = task_type
                    
                    st.success("✅ Prétraitement terminé!")
                    
                    # Afficher les statistiques après prétraitement
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Avant prétraitement")
                        st.dataframe(df_selected.head(), use_container_width=True)
                        st.write("**Types de données:**")
                        st.write(df_selected.dtypes)
                    with col2:
                        st.subheader("Après prétraitement")
                        st.dataframe(st.session_state.df_processed.head(), use_container_width=True)
                        st.write("**Types de données:**")
                        st.write(st.session_state.df_processed.dtypes)
                        
                        # Vérifications
                        if st.session_state.df_processed.isnull().sum().sum() > 0:
                            st.warning("⚠️ Il reste des valeurs manquantes après prétraitement")
                        else:
                            st.success("✅ Aucune valeur manquante")
                        
                        # Vérifier les types numériques
                        non_numeric = st.session_state.df_processed.select_dtypes(exclude=[np.number]).columns
                        if len(non_numeric) > 0:
                            st.warning(f"⚠️ Colonnes non-numériques détectées: {list(non_numeric)}")
                        else:
                            st.success("✅ Toutes les colonnes sont numériques")
                else:
                    st.warning("⚠️ Veuillez sélectionner au moins une variable explicative")
        else:
            st.info("📁 Veuillez d'abord charger un dataset")
    
    with tab4:
        st.header("🧠 Configuration et entraînement du modèle")
        
        if st.session_state.df_processed is not None:
            # Paramètres du modèle dans la sidebar
            st.sidebar.markdown("### 🎛️ Paramètres du Gradient Descent")
            learning_rate = st.sidebar.slider("Learning Rate (α)", 0.001, 1.0, 0.01, 0.001)
            max_iterations = st.sidebar.slider("Nombre d'itérations", 100, 5000, 1000, 100)
            test_size = st.sidebar.slider("Taille du set de test", 0.1, 0.5, 0.2, 0.05)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Aperçu des données")
                st.dataframe(st.session_state.df_processed.head(10), use_container_width=True)
            
            with col2:
                st.subheader("⚙️ Configuration")
                st.write(f"**Type de tâche:** {st.session_state.task_type}")
                st.write(f"**Variable cible:** {st.session_state.target_column}")
                st.write(f"**Nombre de features:** {len(st.session_state.feature_columns)}")
                st.write(f"**Learning rate:** {learning_rate}")
                st.write(f"**Itérations:** {max_iterations}")
            
            if st.button("🚀 Entraîner le modèle", type="primary"):
                with st.spinner("Entraînement en cours..."):
                    try:
                        # Préparer les données
                        X = st.session_state.df_processed[st.session_state.feature_columns].values
                        y = st.session_state.df_processed[st.session_state.target_column].values
                        
                        # Vérifier que les données sont numériques
                        if not np.issubdtype(X.dtype, np.number):
                            st.error("❌ Erreur: Les variables explicatives contiennent des valeurs non-numériques")
                            st.stop()
                        
                        if not np.issubdtype(y.dtype, np.number):
                            st.error("❌ Erreur: La variable cible contient des valeurs non-numériques")
                            st.stop()
                        
                        # Vérifier les valeurs manquantes
                        if np.isnan(X).any():
                            st.error("❌ Erreur: Les variables explicatives contiennent des valeurs manquantes")
                            st.stop()
                        
                        if np.isnan(y).any():
                            st.error("❌ Erreur: La variable cible contient des valeurs manquantes")
                            st.stop()
                        
                        # Normalisation des features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Division train/test
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size, random_state=42
                        )
                        
                        # Créer et entraîner le modèle
                        model = GradientDescent(
                            learning_rate=learning_rate,
                            max_iterations=max_iterations,
                            task_type=st.session_state.task_type
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Sauvegarder dans la session
                        st.session_state.model = model
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.scaler = scaler
                        
                        st.success("✅ Modèle entraîné avec succès!")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
                        st.write("**Détails de l'erreur:**")
                        st.write(f"- Type de X: {type(X[0][0]) if len(X) > 0 and len(X[0]) > 0 else 'N/A'}")
                        st.write(f"- Type de y: {type(y[0]) if len(y) > 0 else 'N/A'}")
                        st.write(f"- Shape de X: {X.shape}")
                        st.write(f"- Shape de y: {y.shape}")
                        st.write("**Vérifiez que:**")
                        st.write("1. Toutes les colonnes sélectionnées sont numériques")
                        st.write("2. La variable cible est appropriée pour le type de tâche")
                        st.write("3. Le prétraitement a été effectué correctement")
        else:
            st.info("🔧 Veuillez d'abord effectuer le prétraitement des données")
    
    with tab5:
        st.header("📈 Résultats et Performance")
        
        if st.session_state.model is not None:
            model = st.session_state.model
            X_train = st.session_state.X_train
            X_test = st.session_state.X_test
            y_train = st.session_state.y_train
            y_test = st.session_state.y_test
            
            # Prédictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Métriques de performance
            col1, col2, col3 = st.columns(3)
            
            if st.session_state.task_type == 'classification':
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                with col1:
                    st.metric("🎯 Précision (Train)", f"{train_accuracy:.3f}")
                with col2:
                    st.metric("🎯 Précision (Test)", f"{test_accuracy:.3f}")
                with col3:
                    st.metric("📊 Différence", f"{abs(train_accuracy - test_accuracy):.3f}")
            
            else:  # régression
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                
                with col1:
                    st.metric("📉 MSE (Test)", f"{test_mse:.3f}")
                with col2:
                    st.metric("📈 R² Score (Test)", f"{test_r2:.3f}")
                with col3:
                    st.metric("🔄 RMSE (Test)", f"{np.sqrt(test_mse):.3f}")
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📉 Courbe d'apprentissage")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(model.costs))),
                    y=model.costs,
                    mode='lines',
                    name='Coût',
                    line=dict(color='#667eea', width=3)
                ))
                fig.update_layout(
                    title="Évolution du coût pendant l'entraînement",
                    xaxis_title="Itérations",
                    yaxis_title="Coût",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Prédictions vs Réalité")
                
                if st.session_state.task_type == 'regression':
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test,
                        y=y_test_pred,
                        mode='markers',
                        name='Prédictions',
                        marker=dict(color='#667eea', size=8, opacity=0.7)
                    ))
                    
                    # Ligne parfaite
                    min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Ligne parfaite',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Prédictions vs Valeurs réelles",
                        xaxis_title="Valeurs réelles",
                        yaxis_title="Prédictions",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:  # classification
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_test_pred)
                    
                    fig = px.imshow(cm, 
                                  text_auto=True,
                                  aspect="auto",
                                  title="Matrice de confusion",
                                  labels=dict(x="Prédictions", y="Réalité", color="Nombre"))
                    st.plotly_chart(fig, use_container_width=True)
            
            # Informations sur le modèle
            st.subheader("🔍 Informations du modèle")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Poids du modèle:**")
                weights_df = pd.DataFrame({
                    'Feature': st.session_state.feature_columns,
                    'Poids': model.weights
                })
                st.dataframe(weights_df, use_container_width=True)
            
            with col2:
                st.write("**Paramètres:**")
                st.write(f"- Biais: {model.bias:.4f}")
                st.write(f"- Learning rate: {model.learning_rate}")
                st.write(f"- Itérations: {model.max_iterations}")
                st.write(f"- Coût final: {model.costs[-1]:.4f}")
        
        else:
            st.info("🧠 Veuillez d'abord entraîner un modèle")

if __name__ == "__main__":
    main()