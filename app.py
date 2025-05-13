import streamlit as st
import re
import contractions
import emoji
import string
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer
import datetime

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse de Sentiment des Tweets",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'interface
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #14171A;
        margin-bottom: 15px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 18px;
    }
    .positive-sentiment {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .neutral-sentiment {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
    }
    .negative-sentiment {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F5F8FA;
        color: #657786;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    .sidebar-content {
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction de prétraitement du texte
def pre_process(tweet):
    pattern_web = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    pattern_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    pattern_hash = r'#(\w+)'
    pattern_handle = r'@\w+'
    pattern_repeat = r'(.)\1{2,}'
    
    t_mod = re.sub(pattern_web, '', tweet)  # Supprime les URLs
    t_mod = re.sub(pattern_email, ' ', t_mod)  # Supprime les emails
    t_mod = re.sub(pattern_hash, " \\1", t_mod)  # Supprime '#'
    t_mod = re.sub(pattern_handle, " ", t_mod)  # Supprime les handles
    t_mod = emoji.demojize(t_mod)  # Convertit les emojis en texte
    
    t_mod = re.sub(r'`', "'", t_mod)  # Remplace ` par '
    t_mod = contractions.fix(t_mod)  # Étend les contractions
    
    t_mod = re.sub(pattern_repeat, r'\1', t_mod)  # Normalise les caractères répétés
    t_mod = re.sub(r'[0-9]', " ", t_mod)  # Supprime les nombres
    
    pattern_punc = "[" + re.escape(string.punctuation) + "]"
    t_mod = re.sub(pattern_punc, " ", t_mod)  # Supprime la ponctuation
    
    t_mod = t_mod.lower()  # Convertit en minuscules
    t_mod = re.sub(r'\s+', " ", t_mod)  # Supprime les espaces multiples
    
    return t_mod.strip()

# Fonction pour charger le tokenizer et le modèle
@st.cache_resource
def load_model_and_tokenizer():
    with st.spinner('Chargement du modèle en cours...'):
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        # Charger le modèle SavedModel directement
        saved_model = tf.saved_model.load('tweet_sentiment_analysis/model_distilBERT')
        # Récupérer la signature serving_default
        inference_func = saved_model.signatures['serving_default']
    return tokenizer, inference_func

# Fonction d'analyse du sentiment
def analyze_tweet(text, tokenizer, inference_func):
    dict_sentiments = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
    cleaned_text = pre_process(text)
    tokens = tokenizer(cleaned_text, return_tensors="tf", max_length=128, truncation=True, 
                      padding="max_length", return_attention_mask=True, return_token_type_ids=False)
    
    # Appeler la signature serving_default avec les arguments nommés
    outputs = inference_func(
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask']
    )
    
    # Extraire les logits
    y_score = outputs['logits']
    probabilities = tf.nn.softmax(y_score, axis=1).numpy()[0]
    y_pred = np.argmax(y_score, axis=1)
    
    return dict_sentiments[y_pred[0]], probabilities, cleaned_text

# Barre latérale avec informations
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.subheader("À propos de cette application")
    st.write("""
    Cette application utilise un modèle DistilBERT fine-tuné pour analyser le sentiment 
    des tweets en trois catégories: positif, neutre ou négatif.
    """)
    
    st.subheader("Comment ça marche")
    st.write("""
    1. Entrez votre tweet dans la zone de texte
    2. Cliquez sur 'Analyser'
    3. Consultez le résultat de l'analyse et les statistiques
    """)
    
    st.subheader("Prétraitement appliqué")
    st.write("""
    • Suppression des URLs et emails
    • Conversion des emojis en texte
    • Nettoyage de la ponctuation
    • Normalisation du texte
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# En-tête principal
st.markdown('<div class="main-header">🐦 Analyse de Sentiment des Tweets</div>', unsafe_allow_html=True)

# Corps de l'application
col1, col2 = st.columns([3, 1])

with col1:
    # Champ de saisie du tweet avec taille augmentée et placeholder
    tweet_input = st.text_area(
        "Entrez un tweet à analyser:",
        height=150,
        placeholder="Qu'avez-vous en tête aujourd'hui? Entrez votre texte ici pour analyser son sentiment..."
    )

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    # Bouton pour analyser avec style amélioré
    analyze_button = st.button("Analyser le sentiment", use_container_width=True)

# Traitement de l'analyse
if analyze_button:
    if tweet_input.strip():
        try:
            with st.spinner('Analyse du sentiment en cours...'):
                # Charger le modèle et le tokenizer
                tokenizer, inference_func = load_model_and_tokenizer()
                
                # Analyser le sentiment
                sentiment, probabilities, cleaned_text = analyze_tweet(tweet_input, tokenizer, inference_func)
                
                # Déterminer la classe CSS pour le style
                sentiment_class = f"{sentiment.lower()}-sentiment"
                
                # Afficher le résultat avec un style visuel
                st.markdown(f"""
                <div class="result-box {sentiment_class}">
                    <h3>Résultat de l'analyse</h3>
                    <p>Sentiment prédit: <strong>{sentiment}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Afficher les probabilités
                st.subheader("Détails de l'analyse")
                col_neg, col_neut, col_pos = st.columns(3)
                with col_neg:
                    st.metric("Négatif", f"{probabilities[0]*100:.1f}%")
                with col_neut:
                    st.metric("Neutre", f"{probabilities[1]*100:.1f}%")
                with col_pos:
                    st.metric("Positif", f"{probabilities[2]*100:.1f}%")
                
                # Afficher le texte prétraité
                with st.expander("Texte prétraité"):
                    st.code(cleaned_text)
                
        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")
    else:
        st.warning("⚠️ Veuillez entrer un tweet à analyser.")

# Pied de page avec copyright
current_year = datetime.datetime.now().year
st.markdown(f"""
<div class="footer">
    © {current_year} | Application réalisée par Aminata BA et Dié Sylla | Tous droits réservés
</div>
""", unsafe_allow_html=True)