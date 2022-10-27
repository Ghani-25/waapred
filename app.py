import streamlit as st
import gdown
import os
import math
import torch
import pandas as pd
url = "https://drive.google.com/uc?export=download&id=1rBG3CI5b7uG90TOX7c4mJytdPF560M_F"
output = "BESTmodel_weights.pt"
gdown.download(url, output, quiet=False)

#model = torch.hub.load_state_dict_from_url('https://github.com/Ghani-25/waapred/blob/db5336462ef64618a84fca6ba4e7224316fe7393/BESTmodel_weights.pt')
#model.eval()
with st.sidebar.form("Input"):
    queryText = st.text_area("Response rate to predict:", height=4, max_chars=None)
    btnResult = st.form_submit_button('Run')

if btnResult:
    st.sidebar.text('Button pushed')

    # run query
    #modelfile = "https://drive.google.com/drive/u/3/folders/1EmXO09Yxm9BlPIhgOI5RrXmVHdA0Y3ny"
    #gdown.download_folder(modelfile, quiet=True, use_cookies=False)
    #model = torch.hub.load_state_dict_from_url('https://github.com/Ghani-25/waapred/blob/db5336462ef64618a84fca6ba4e7224316fe7393/BESTmodel_weights.pt')
    #model.eval()
    urll = './BESTmodel_weights.pt'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    model = torch.load(urll, map_location=torch.device('cpu'))
    model.to(device)
    model.eval()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
    from torch.utils.data import DataLoader

    BASE_MODEL = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    y_preds = []
    encoded = tokenizer(queryText, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cpu")
    y_preds += model(**encoded).logits.reshape(-1).tolist()

    indications = ["Rédiger un message compris entre 100 et 150 caractères", "Mettre la phrase d'accroche en avant", "S'adresser à la personne avec son prénom/nom"]
    if y_preds[0] <= 20.60 :
        realvalue = y_preds[0]
        realone = f'Le taux de prédiction est compris entre {realvalue-((realvalue*11)/100)} et {realvalue+((realvalue*11)/100)}, pensez à modifier votre message en considérant les indications suivantes {*indications,}'
    elif y_preds[0] > 20.60 and y_preds[0] < 22.99 :
        realvalue = y_preds[0] * 2.11
        realone = f'Le taux de prédiction est compris entre {realvalue-((realvalue*11)/100)} et {realvalue+((realvalue*11)/100)}, votre message peut etre amélioré en considérant les indications suivantes {*indications,}'
    else:
        realvalue = y_preds[0] * 3.11
        realone = f'Le taux de prédiction est compris entre {realvalue-((realvalue*11)/100)} et {realvalue+((realvalue*11)/100)}, votre message semblerait être bon :)'

    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame([queryText, realone], ["CONTENT", "Prediction"]).T
    st.dataframe(df)
