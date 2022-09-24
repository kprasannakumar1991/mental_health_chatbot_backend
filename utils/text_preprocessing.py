from bs4 import BeautifulSoup
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# stop = set(stopwords.words('english'))

stop= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

lemmatizer = WordNetLemmatizer()

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
    
    text = BeautifulSoup(text, "html.parser").get_text()   
    
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

def lemmatize(sentence):
    sentence = remove_special_characters(sentence)
    
    clean_tokens = []
    tokens = word_tokenize(sentence)
    for token in tokens:
        lemmetized_word = lemmatizer.lemmatize(token.lower())
        clean_tokens.append(lemmetized_word)
            
    cleaned_sentence = " ".join(clean_tokens)
    
    return cleaned_sentence

def text_process_question(question):
    # question = remove_stopwords(question)
    question = lemmatize(question)
    return question


def text_process_answer(answer):
    return remove_special_characters(answer)