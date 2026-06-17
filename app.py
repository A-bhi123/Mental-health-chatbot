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

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# HuggingFace cache directory - writable on all platforms
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.dirname(__file__), 'hf_cache')
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

# English to Swahili translator
eng_swa_tokenizer = AutoTokenizer.from_pretrained("Rogendo/en-sw", cache_dir=os.environ['TRANSFORMERS_CACHE'])
eng_swa_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/en-sw", cache_dir=os.environ['TRANSFORMERS_CACHE'])

eng_swa_translator = pipeline(
    "text2text-generation",
    model=eng_swa_model,
    tokenizer=eng_swa_tokenizer,
)

def translate_text_eng_swa(text):
    translated_text = eng_swa_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text

# Swahili to English translator
swa_eng_tokenizer = AutoTokenizer.from_pretrained("Rogendo/sw-en", cache_dir=os.environ['TRANSFORMERS_CACHE'])
swa_eng_model = AutoModelForSeq2SeqLM.from_pretrained("Rogendo/sw-en", cache_dir=os.environ['TRANSFORMERS_CACHE'])

swa_eng_translator = pipeline(
    "text2text-generation",
    model=swa_eng_model,
    tokenizer=swa_eng_tokenizer,
)

def translate_text_swa_eng(text):
    translated_text = swa_eng_translator(text, max_length=128, num_beams=5)[0]['generated_text']
    return translated_text


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
    doc = nlp(msg)
    detected_language = doc._.language['language']

    chatbotResponse = "I'm here to help. Could you please rephrase that?"

    if detected_language == "en":
        res = getResponse(predict_class(msg, model), intents)
        chatbotResponse = res
    elif detected_language == 'sw':
        translated_msg = translate_text_swa_eng(msg)
        res = getResponse(predict_class(translated_msg, model), intents)
        chatbotResponse = translate_text_eng_swa(res)

    return chatbotResponse


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

    doc = nlp(userText)
    detected_language = doc._.language['language']

    bot_response_translate = userText

    if detected_language == 'sw':
        bot_response_translate = translate_text_swa_eng(userText)

    chatbot_response_text = chatbot_response(bot_response_translate)

    if detected_language == 'sw':
        chatbot_response_text = translate_text_eng_swa(chatbot_response_text)

    return chatbot_response_text


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
