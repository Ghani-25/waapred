from flask import Flask, render_template, request, session, redirect

#Import main library
import streamlit as st
import gdown
import torch
import os
import math
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

#initialise the app
app = Flask(__name__)

#import the model


url = "https://drive.google.com/uc?export=download&id=1rBG3CI5b7uG90TOX7c4mJytdPF560M_F"
output = "BESTmodel_weights.pt"
gdown.download(url, output, quiet=False)
urll = './BESTmodel_weights.pt'
device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")        
model = torch.load(urll, map_location=torch.device('cpu'))
model.to(device)
model.eval()

#create our "home" route using the "index.html" page

@app.route("/", methods = ['POST'])
def predict():
    BASE_MODEL = "camembert-base"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    y_preds = []
    encoded = tokenizer(request.form.get("comment"), truncation=True, padding="max_length", max_length=256, return_tensors="pt").to("cpu")
    y_preds += model(**encoded).logits.reshape(-1).tolist()

    pd.set_option('display.max_rows', 500)
    df = pd.DataFrame([request.form.get("comment"), y_preds], ["CONTENT", "Prediction"]).T
    return render_template('simple.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
