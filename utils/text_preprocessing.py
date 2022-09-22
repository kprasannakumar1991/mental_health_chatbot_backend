from bs4 import BeautifulSoup
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop = set(stopwords.words('english'))

def remove_special_characters(sentence):
    
    text = str(sentence)
    text = text.lower()
    
    text = text.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
    text = text.replace("what's", "what is").replace("it's", "it is").replace("i'm", "i am")
    text = text.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
    text = text.replace("'ll", " will").replace("n't", " not").replace("'re", " are").replace("'ve", " have")
    text = text.replace("?", "")
    
    text = re.sub('[^a-zA-Z0-9\n]', ' ', text) #------------------- Replace every special char with space
    text = re.sub('\s+', ' ', text).strip() #---------------------- Replace excess whitespaces
    
    text = BeautifulSoup(text).get_text()   
    
    return text


def remove_stopwords(sentence):
    sentence = remove_special_characters(sentence)
    
    clean_tokens = []
    tokens = word_tokenize(sentence)
    for token in tokens:
        if token.lower() not in stop:
            clean_tokens.append(token.lower())
            
    
    cleaned_sentence = " ".join(clean_tokens)
    
    return cleaned_sentence