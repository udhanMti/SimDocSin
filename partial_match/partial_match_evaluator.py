import numpy as np
import json
from laser_control import get_embeddig_list
from doc_to_sentence import doc_to_sentence
from sentence_to_word import sentence_to_word
from datetime import datetime


def get_similarity_matrix(embeds1, embeds2):#, sentences_s, sentences_t):
    matrix = []
    
    dict_s = {}
    dict_t = {}

    k = 5
    for i,sent_s in enumerate(embeds1):
        #dict_s[i] = [-1,[]] # #
        x = []
        for j, sent_t in enumerate(embeds2):
            #x = dict_s[i][1]
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))
            if(cos_similarity>0.65):
               
                x.append([j,cos_similarity])

        dict_s[i] = x
         
       
    _dict = {}

    for i in dict_s:
        neighbours = []
        x=None
        y=-1
        for j in dict_s[i]:
            #matrix.append((i,j[0]))
            neighbours.append(j[0])
            if(j[1]>y):
                y = j[1]
                x = j[0]
        if(x!=None):
            matrix.append((i,x))
        if(len(neighbours)>0):
            neighbours.remove(x)
        _dict[i] = neighbours
    
    return [matrix,_dict]

def diagonal_extract(matrix, dict_s):
    start_position =  matrix[0]
    x = start_position[0]
    y = start_position[1]
    d = []
    
    while ((x,y) in matrix) or ((x-1,y) in matrix) or ((x,y-1) in matrix) or ((x in dict_s) and (y in dict_s[x])) or ((x-1 in dict_s) and (y in dict_s[x-1])) or ((x in dict_s) and ((y-1) in dict_s[x])):

        if (x,y) in matrix:
            d.append((x,y))
            x+=1
            y+=1
        elif (x-1, y) in matrix:
            d.append((x-1, y))
            y+=1
        elif (x, y-1) in matrix:
            d.append((x, y-1))
            x+=1

        elif (x in dict_s) and (y in dict_s[x]):
            d.append((x,y))
            x+=1
            y+=1
        elif (x-1 in dict_s) and (y in dict_s[x-1]):
            d.append((x-1, y))
            y+=1    
        elif (x in dict_s) and (y-1) in dict_s[x]:
            d.append((x, y-1))
            x+=1


    return d  
  


def sequence_matching(matrix, dict_s, sentences_s, sentences_t):
  
    diagonals = []
    while len(matrix)>0:
        d = diagonal_extract(matrix, dict_s)
        #print(d)
        #diagonals.append(d)


        s = []
        t = []

        for i in d:
           
                if (i[1] not in t):
                    t.append(i[1])        

                if (i[0] not in s):
                    s.append(i[0])

        if(len(s)>1):
            diagonals.append((s,t))

        for each in d:
            if(each in matrix):
                matrix.remove(each)
    return diagonals

file1  = open('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/embedded_data/embedding_dms.json',encoding='utf8')
data1 = json.load(file1)

file2  = open('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/data/longdoc_dms_si.json',encoding='utf8')
data2 = json.load(file2)

file3  = open('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/embedded_data/embedding_dms_si.json',encoding='utf8')
data3 = json.load(file3)

source_lang = 'en'
target_lang = 'si'

combined_target_doc = data2['content_'+target_lang]

combined_target_embedd = data3[0]['embed_'+ target_lang]

sentences_in_combined_target_doc = doc_to_sentence(combined_target_doc,target_lang)

start=datetime.now()
#print('0.6')
## ======== code for evaluation =================
recall_numerator = 0
precision_numerator = 0
precision_denominator = 0
source_doc_count = 50
r=0
for doc in data1[:source_doc_count]:
    print(r)
    r+=1

    doc_s = doc['content_'+source_lang]
    source_embedd =  doc['embed_'+source_lang]

    target_doc = doc['content_' + target_lang]
    
    actual_target_sentences_pre = doc_to_sentence(target_doc, target_lang)
    actual_target_sentences=[]
    for sentence in actual_target_sentences_pre:
        if(len(sentence_to_word(sentence, target_lang))>2):
           # print(sentence)
            actual_target_sentences.append(sentence)

    matrix, dict_s = get_similarity_matrix(source_embedd, combined_target_embedd)
    sentences_in_source = doc_to_sentence(doc_s,source_lang)
    diagonals = sequence_matching(matrix, dict_s, sentences_in_source, sentences_in_combined_target_doc)

    true_predicted_target_sentences = set()
    predicted_target_sentences = set()

    for index,diagonal in enumerate(diagonals):
        partial_source = []
        partial_target = []
    
        
        for j in diagonal[1]:
   
            sent = sentences_in_combined_target_doc[j]
            if(len(sentence_to_word(sent, target_lang))>2):
                predicted_target_sentences.add(sent)
            
                if(sent in actual_target_sentences):
                    true_predicted_target_sentences.add(sent)
            
     
    recall_numerator += len(true_predicted_target_sentences)/len(actual_target_sentences)
    if len(predicted_target_sentences) > 0 :
        precision_numerator += len(true_predicted_target_sentences)/len(predicted_target_sentences)
        precision_denominator+=1
   
print(datetime.now()-start)
recall = recall_numerator / source_doc_count
print("recall: ",recall)

if(precision_denominator>0):
    precision = precision_numerator/ precision_denominator
    print("precision: ",precision)
else:
    print('precision not defined')