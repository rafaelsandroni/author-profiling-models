from nltk.corpus import stopwords
import re, string
import numpy as np
from bs4 import BeautifulSoup
import html

def clean(doc, lang = 'portuguese'):
    
    stop_words = set(stopwords.words(lang))
    
    # Lowercase
    doc = doc.lower()    
    # Remove HTML codes
    try:
        doc = BeautifulSoup(doc, features="lxml").get_text()    
    except:
        pass
    # Remove numbers
    # doc = re.sub(r"[0-9]+", "", doc)
    # remove HTML space code
    # tokens = tokens.replace('&nbsp', string.whitespace)
    # Split in tokens
    tokens = doc.split()
    # Remove Stopwords
    tokens = [w for w in tokens if not w in stop_words]
    # Remove punctuation
    # tokens = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]
    # remove html codes
    tokens = [html.unescape(w) for w in tokens]
    
    # Tokens with less then two characters will be ignored
    tokens = [word for word in tokens if len(word) > 1]

    return ' '.join(tokens)




from sklearn import preprocessing

def labelEncoder(y):
    le = preprocessing.LabelEncoder()
    le.fit(y)

    # print('>> classes', list(le.classes_))

    return (le.transform(y), len(le.classes_), le.classes_)

