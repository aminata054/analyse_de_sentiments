# Analyse de Sentiment des Tweets

Cette application utilise des modèles de deep learning pour analyser le sentiment des tweets et déterminer s'ils sont positifs, neutres ou négatifs. Développée par **Aminata BA** et **Dié Sylla**, elle est disponible en deux versions : une version de base qui analyse les tweets en anglais, et une version améliorée qui supporte également les tweets en français grâce à une traduction automatique.

## Fonctionnalités

### Version de base
- Interface utilisateur moderne avec Streamlit
- Analyse de sentiment pour les tweets en anglais
- Prétraitement complet des tweets (suppression d'URLs, normalisation, etc.)
- Visualisation des résultats avec pourcentages de confiance
- Affichage du texte prétraité

### Version améliorée
- Toutes les fonctionnalités de la version de base
- Détection automatique de la langue du tweet
- Traduction automatique des tweets français vers l'anglais
- Affichage du texte original et de sa traduction
- Interface utilisateur améliorée avec indications sur le processus de traduction

## Architecture technique

L'application est construite autour de plusieurs composants clés :

1. **Frontend** : Interface utilisateur développée avec Streamlit
2. **Modèle de sentiment** : DistilBERT fine-tuné pour l'analyse de sentiment
3. **Modèle de traduction** (version améliorée uniquement) : Helsinki-NLP pour la traduction français-anglais
4. **Prétraitement** : Pipeline de nettoyage et normalisation du texte

Le flux de traitement est le suivant :
1. Entrée utilisateur (tweet)
2. [Version améliorée] Détection de la langue et traduction si nécessaire
3. Prétraitement du texte (nettoyage, normalisation)
4. Tokenization via DistilBERT
5. Inférence via le modèle SavedModel
6. Affichage des résultats à l'utilisateur

## Installation

### Prérequis
- Python 3.7+
- pip ou conda pour l'installation des dépendances

### Dépendances
```bash
pip install streamlit tensorflow transformers emoji contractions requests numpy
```

### Modèles requis
Le modèle principal doit être situé dans le répertoire `tweet_sentiment_analysis/model_distilBERT`. La version améliorée téléchargera automatiquement le modèle de traduction Helsinki-NLP lors de la première utilisation.

## Utilisation

### Lancement de l'application
```bash
streamlit run app.py  # Version de base
# ou
streamlit run app_multilingual.py  # Version améliorée avec support français
```

### Interface utilisateur
1. Entrez votre tweet dans la zone de texte
2. Cliquez sur "Analyser le sentiment"
3. Consultez les résultats d'analyse

## Description des fichiers

### Version de base (`app.py`)
Le code utilise une interface Streamlit pour afficher une zone de texte où les utilisateurs peuvent entrer des tweets en anglais. Le prétraitement nettoie les URLs, la ponctuation, et normalise le texte avant de l'envoyer au modèle DistilBERT pour l'analyse de sentiment. Les résultats sont affichés visuellement avec des codes couleur selon le sentiment détecté.

Fonctions principales :
- `pre_process()` : Nettoie et normalise le texte du tweet
- `load_model_and_tokenizer()` : Charge le modèle DistilBERT et le tokenizer
- `analyze_tweet()` : Prétraite le tweet et effectue l'inférence pour déterminer le sentiment

### Version française (`app_french.py`)
Cette version étend la version de base en ajoutant la prise en charge du français. Elle utilise un modèle de traduction Helsinki-NLP pour convertir les tweets français en anglais avant l'analyse. Le code détecte automatiquement la langue d'entrée et affiche la traduction à l'utilisateur pour plus de transparence.

### Version améliorée (`tweet.py`)
Cette version étend les deux autres versions de base en permettant à l'utilisateur de choisir la langue qu'il préfère avant de passer au prétraitement

Fonctions supplémentaires :
- `detect_language()` : Identifie si le texte est principalement en français ou en anglais
- `translate_text()` : Traduit le texte français vers l'anglais
- `load_models_and_tokenizer()` : Charge à la fois le modèle d'analyse et de traduction

## Fine-tuning et modèles
L'application repose sur un modèle DistilBERT fine-tuné pour la tâche d'analyse de sentiment sur un jeu de données de tweets.
Le fine-tuning a été réalisé sur un corpus annoté manuellement avec trois classes (positif, neutre, négatif).

### 🔧 Détails du fine-tuning (`evaluate_model.ipynb`)
- Modèle de base : `distilbert-base-uncased`

- Tâche : classification de sentiment à 3 classes

- Données : tweets annotés

- Durée : quelques minutes sur GPU

 Remarque : Les performances en français sont affectées par la qualité de la traduction. Le modèle peut confondre les émotions lors du passage par la langue anglaise.


## Performances et limitations

### Performances
- Le modèle DistilBERT offre un bon équilibre entre précision et vitesse
- Les temps de réponse sont généralement inférieurs à 2 secondes pour la version de base
- La version avec traduction peut nécessiter 3-5 secondes supplémentaires pour la traduction

### Limitations
- La détection de langue est basique et peut être améliorée
- La traduction peut parfois altérer les nuances émotionnelles du texte original
- Le modèle est entraîné principalement sur des tweets et peut être moins précis sur d'autres types de textes
- La version actuelle ne prend pas en charge d'autres langues que le français et l'anglais

Développé avec ❤️ par Aminata BA et Dié Sylla
