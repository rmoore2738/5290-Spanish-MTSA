from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import joblib
from transformers import MarianTokenizer, MarianMTModel
import json
from flask import Flask, request, jsonify

loaded_model=joblib.load("BinaryLogReg.pkl")
loaded_stop=joblib.load("stopwords (1).pkl")
loaded_vec=joblib.load("Vectorizer.pkl")

spanish_model = joblib.load("spanish_model.pkl")
spanish_vec = joblib.load("spanish_vectorizer.pkl")

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-en")

app = Flask(__name__)

def classify_english(document):
    label = {0: 'negative', 1: 'positive'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[y], proba

def classify_spanish(document):
    label = {0: 'negativo', 1: 'positivo'}
    X = spanish_vec.transform([document])
    y = spanish_model.predict(X)[0]
    proba = np.max(spanish_model.predict_proba(X))
    return y, proba

def translate(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model.generate(input_ids=input_ids, decoder_start_token_id=model.config.decoder_start_token_id, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class ReviewForm(Form):
    moviereview = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    #return render_template('reviewform.html', form=form, jsonfile=json.dumps(data))
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    #with open('file.json', 'w') as f:
        #json.dump(request.form, f)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify_english(review)
        return render_template('results.html',content=review,prediction=y,probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/results_spanish', methods=['POST'])
def results_spanish():
    form = ReviewForm(request.form)
    #with open('file.json', 'w') as f:
        #json.dump(request.form, f)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify_spanish(review)
        return render_template('results_spanish.html',content=review,prediction=y,probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/translate', methods=['POST'])
def translate_route():
    #data = request.get_json()
    form = ReviewForm(request.form)
    #with open('file.json', 'w') as f:
        #json.dump(request.form, f)
    if request.method == 'POST' and form.validate():
        spanish_sentence = request.form.to_dict('moviereview')
        spanish_sentence = str(spanish_sentence)
        spanish_sentence = spanish_sentence.removeprefix("{'moviereview': '")
        spanish_sentence = spanish_sentence.removesuffix("', 'submit_btn': 'Translate'}")
        #if spanish_sentence:
        translated_sentence = translate(spanish_sentence)
        return render_template('translate.html',content=spanish_sentence,translate=translated_sentence)
            #return jsonify({"translation": translated_sentence})
        #else:
    return render_template('reviewform.html', form=form)
            #return jsonify({"error": "Please provide a 'spanish_sentence' key in the request body."})


if __name__ == '__main__':
    app.run(debug=True)