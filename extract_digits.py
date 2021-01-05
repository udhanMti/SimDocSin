import re
from doc_to_sentence import doc_to_sentence

def extract_digits(doc, lang):
    sentences = doc_to_sentence(doc, lang)
    digits = []
    for sentence in sentences:
        temp = re.findall(r'\d+', sentence) 
        res = list(map(int, temp)) 
        digits += res
    return digits

def get_digit_similarity(s_digits,t_digits):
    count=0
    for digit in s_digits:
        if(digit in t_digits):
            count+=1
    if(len(s_digits)>0):
        return float(count)/len(s_digits)
    else:
        return 0