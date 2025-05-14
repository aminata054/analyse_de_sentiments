import pandas as pd
import os
os.environ['HF_HOME'] = 'C:/hf_cache'
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch


# Chemin du fichier CSV original
input_file = "test.csv"  
output_file = "test_traduit_francais.csv"

# Chargement du fichier
df = pd.read_csv(input_file)
texts_en = df['text'].astype(str).tolist()

# Initialisation du modèle de traduction EN → FR
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Utilisation GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Fonction pour la traduction par lots
def translate_batch(texts, batch_size=32):
    translated_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Traduction"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        translated = model.generate(**inputs, max_length=256)
        batch_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translated_texts.extend(batch_translations)
    return translated_texts

# Traduction de tous les textes
df['text_fr'] = translate_batch(texts_en)

# Sauvegarde dans un nouveau CSV
df.to_csv(output_file, index=False)
print(f"✅ Traduction terminée. Fichier enregistré sous : {output_file}")
