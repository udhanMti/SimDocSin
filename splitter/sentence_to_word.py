import nltk
import sys
import string
from sinling import SinhalaTokenizer


def sentence_to_word(sentence, lang):

    if(lang=='en'):
        words = nltk.word_tokenize(sentence)
        return [e.lower() for e in words]
    else:
        tokenizer = SinhalaTokenizer()

        return tokenizer.tokenize(sentence)
    

def sentence_to_word_si(sentence):
    tokenizer = None #SinhalaTokenizer()
    words = tokenizer.tokenize(sentence)
    return [e for e in words if e not in ('.', ',','(',')','-','/','|','&','#','*')]