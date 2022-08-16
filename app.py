import streamlit as st
import torch
import gdown
import os
import math
import pandas as pd
with st.sidebar.form("Input"):
    queryText = st.text_area("Response rate to predict:", height=4, max_chars=None)
    btnResult = st.form_submit_button('Run')

if btnResult:
    st.sidebar.text('Button pushed')

    # run query
    modelfile = "https://drive.google.com/drive/u/3/folders/1EmXO09Yxm9BlPIhgOI5RrXmVHdA0Y3ny"
    gdown.download_folder(modelfile, quiet=True, use_cookies=False)
    url = 'model/BESTmodel_weights.pt'
    model = torch.load(url)
    model.eval()
    # model = torch.hub.load_state_dict_from_url('https://github.com/Ghani-25/waapred/blob/db5336462ef64618a84fca6ba4e7224316fe7393/BESTmodel_weights.pt')
    # model.eval()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
    from torch.utils.data import DataLoader

    BASE_MODEL = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)

    y_preds = []
    encoded = tokenizer(queryText, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cuda")
    y_preds += model(**encoded).logits.reshape(-1).tolist()

    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame([queryText, y_preds], ["CONTENT", "Prediction"]).T
    st.dataframe(df)
