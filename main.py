# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn
from fastapi import FastAPI 
from nltk.tokenize import WordPunctTokenizer, sent_tokenize
#nltk.download('punkt')

app = FastAPI(
    title="NLP Project API")

origins = [    
    "http://localhost",
    "http://localhost:8080",
    "file://",
    "null"    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "BinaryLogReg.pkl"), "rb") as f:
    sent_model = joblib.load(f)
'''
# load the machine translation models 
with open(
    join(dirname(realpath(__file__)), "english_seqs.pkl"), "rb") as f:
    translate_model = joblib.load(f)
'''
# cleaning the input text
def text_cleaning(text):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers

    sentences = sent_tokenize(text)
    
    sentences = [[text.lower() for text in sent.split() if text not in stopwords.words('english')] 
                for sent in sentences]
    # return a list of sentences
    return [" ".join(sent) for sent in sentences]

def vectorize(text):
    # vectorize the text
    vectorizer = joblib.load('Vectorizer.pkl')
    return vectorizer.transform(text)

@app.get("/predict-review")
def predict_sentiment(review: str):
    # clean the review
    cleaned_review = text_cleaning(review)
    # vectorize cleaned review
    vector = vectorize(cleaned_review)
    # perform prediction on vectorized review
    prediction = sent_model.predict(vector)
    output = int(prediction[0])
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    result = {"prediction": sentiments[output]}
    return result

@app.get("/translate-review")
def translate(review: str):
    # clean the review
    cleaned_review = text_cleaning(review)

    # perform prediction
    translate = translate_model.predict([cleaned_review])
    
    # show results
    result = {"Translation": translate}
    return result