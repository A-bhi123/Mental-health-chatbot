import os
import nltk

# NLTK data download - production safe
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('popular', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

# Keras / TensorFlow load - compatible with keras 2.10 model
import tf_keras as keras
model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model.h5'))

import json
import random


from spacy.language import Language
from spacy_langdetect import LanguageDetector



def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")

if not Language.has_factory("language_detector"):
    Language.factory("language_detector", func=get_lang_detector)

nlp.add_pipe('language_detector', last=True)


BASE_DIR = os.path.dirname(__file__)
intents = json.loads(open(os.path.join(BASE_DIR, 'intents.json')).read())
words = pickle.load(open(os.path.join(BASE_DIR, 'texts.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(BASE_DIR, 'labels.pkl'), 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."

def chatbot_response(msg):
    res = getResponse(predict_class(msg, model), intents)
    return res

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg', '')
    if not userText:
        return "Please enter a message."
    return chatbot_response(userText)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
