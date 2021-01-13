from splitter.doc_to_sentence import doc_to_sentence
from splitter.sentence_to_word import sentence_to_word
import math

def get_sentence_frequency_list(doc, lang):
    sentences = doc_to_sentence(doc, lang)
    length = len(sentences)
    weights = [] 
    for sent in sentences:
         frequency = sentences.count(sent)
         weights.append(frequency/length)

    return weights

def get_sentence_count():
    # get no of same sentence in doc
    #assume all are distinct
    return 1.0

def get_sentence_length_weighting_list(doc, lang):
    weight=[]
    sentences = doc_to_sentence(doc, lang)
    for sentence in sentences:
        weight.append(get_sentence_count()*len(sentence.split()))
    total_tokens= float(sum(weight))

    return [x/total_tokens for x in weight]

def get_sentence_length_weighting_list_2(sentences, lang):
    weight=[]

    for sentence in sentences:
        weight.append(get_sentence_count()*len(sentence.split()))
    total_tokens= float(sum(weight))

    return [x/total_tokens for x in weight]

def get_senetence_frequencies(doc, word, lang):
    frequency = 0
    sentences = doc_to_sentence(doc, lang)
    for sentence in sentences:
       if(word in sentence):
           frequency+=1  
    return frequency
           
def documentMassNormalization(wieght_list):
    total_weight=float(sum(wieght_list))
    for i in range(len(wieght_list)):
        wieght_list[i]=wieght_list[i]/total_weight
    return wieght_list

def get_idf_weighting_list(doc,sentence_count,N, lang):
    weights=[]
    sentences = doc_to_sentence(doc, lang)
    for sentence in sentences:
        sent =sentence.strip()
        weights.append(1+math.log((1.0+N)/(1.0+sentence_count[sent])))
    return weights

def get_intra_doc_word_idf_weighting_list(doc, lang):
    weights=[]
    sentences=[]
    sentences = doc_to_sentence(doc, lang)
    for sentence in sentences:
        weight = 0

        words = sentence_to_word(sentence,lang)
       
        for word in words:
            #weight+= (words.count(word)/len(words)) * (len(sentences)/get_senetence_frequencies(doc, word))
            weight+= 1+math.log((1.0+len(sentences))/(1.0+get_senetence_frequencies(doc, word, lang)))
        weights.append(weight)
    return weights

def get_inter_doc_word_idf_weighting_list(doc, word_count, N, lang):
    weights=[]
    sentences=[]
    sentences = doc_to_sentence(doc, lang)
    for sentence in sentences:
        weight = 0
        words = sentence_to_word(sentence,lang)
  
        for word in words:
            #weight+= (words.count(word)/len(words)) * (len(sentences)/get_senetence_frequencies(doc, word))
            weight+= 1+math.log((1.0+N)/(1.0+word_count[word]))
        weights.append(weight)
    return weights

def sentence_count_web_domain(documents, lang):
    sentence_count={}
    for doc in documents:
        sentences= doc_to_sentence(doc, lang)
        for sentence in sentences:
            sent = sentence.strip()
            if (sent in sentence_count):
                sentence_count[sent] +=1
            else :
                sentence_count[sent] = 1
    return sentence_count

def word_count_over_docs(documents, lang):
    word_count={}
    for doc in documents:
        sentences=[]
        sentences = doc_to_sentence(doc, lang)
        my_words=[]
        for sentence in sentences:
            words = []
            if(lang=='en'):
              words = sentence_to_word(sentence,"en")
            elif(lang=='si'):
              words = sentence_to_word(sentence,"si")
            for word in words:
                if(word not in my_words):
                    my_words.append(word)

        for word in my_words:
              if (word in word_count):
                word_count[word] +=1
              else :
                word_count[word] = 1
    return word_count

def get_slidf_weighting_list(sentence_weight, idf_weight):
    return [documentMassNormalization([sentence_weight[j][i]*idf_weight[j][i] for i in range (len(sentence_weight[j]))]) for j in range (len(sentence_weight))]
