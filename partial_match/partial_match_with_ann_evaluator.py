import numpy as np
import json
from laser_control import get_embeddig_list
from doc_to_sentence import doc_to_sentence
from sentence_to_word import sentence_to_word
from datetime import datetime
from annoy import AnnoyIndex
from greedy_mover_distance import greedy_mover_distance
from weight_schema import *

f=1024
u = AnnoyIndex(f, 'euclidean')
u.load('./ann/test.ann')

def get_similarity_matrix(embeds1, embeds2):
    matrix = []
    dict_s = {}
    dict_t = {}
    k = 5
    for i,sent_s in enumerate(embeds1):
        lst = u.get_nns_by_vector(sent_s, 10, 100000)

        x = []
        for j in lst:
         
            sent_t = embeds2[j]
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

        s = []
        t = []
       
        for i in d:
    
            if (i[1] not in t):
                t.append(i[1])        

            if (i[0] not in s):
                s.append(i[0])

        if(len(s)>0):
            diagonals.append((s,t))
        for each in d:
            if(each in matrix):
                matrix.remove(each)
    return diagonals

def post_processor(diagonals, gap_threshold, min_length):
    diagonals = sorted(diagonals,key=lambda x: x[0][0])
    #gap_threshold = 40
    #print(diagonals)
    processed_diagonals = []
    for i in range(len(diagonals)):
        diagonal = diagonals[i]
        temp = []
        new_diagonal = diagonal
        last = 0
        for j in range(len(processed_diagonals)):
            last = j
            pro_diagonal = processed_diagonals[j]
            temp.append(pro_diagonal)
            gap_s = (diagonal[0][0] - pro_diagonal[0][-1])
            gap_t = (diagonal[1][0] - pro_diagonal[1][-1])
            if ((gap_s <= gap_threshold) and (gap_s>=0) and (gap_t <= gap_threshold) and (gap_t >=0)):
                s = [_s for _s in range(pro_diagonal[0][0], diagonal[0][-1] + 1)]
                t = [_t for _t in range(pro_diagonal[1][0], diagonal[1][-1] + 1)]
                new_diagonal = (s, t)
                temp = temp[:-1]
                break
            elif((gap_s <= gap_threshold) and (gap_t <= gap_threshold) and (pro_diagonal[1][0] <= diagonal[1][0])):
                s = [_s for _s in range(pro_diagonal[0][0], max(pro_diagonal[0][-1],diagonal[0][-1]) + 1)]
                t = [_t for _t in range(pro_diagonal[1][0], max(pro_diagonal[1][-1],diagonal[1][-1]) + 1)]
                new_diagonal = (s, t)
                temp = temp[:-1]
                break

        temp.append(new_diagonal)
        temp.extend(processed_diagonals[last + 1:])
        processed_diagonals = temp
    #print(processed_diagonals)

    final_diagonals = []
    for diagonal in processed_diagonals:
        if((len(diagonal[1])>min_length)):# or (len(diagonal[1])>1)):
         final_diagonals.append(diagonal)

    return final_diagonals


file1  = open('./embedded_data/embedding_dms_checked.json',encoding='utf8')
data1 = json.load(file1)

file2  = open('./data/longdoc_dms_si_checked.json',encoding='utf8')
data2 = json.load(file2)

file3  = open('./embedded_data/embedding_dms_si_checked.json',encoding='utf8')
data3 = json.load(file3)

source_lang = 'en'
target_lang = 'si'

combined_target_doc = data2['content_'+target_lang]

combined_target_embedd = data3[0]['embed_'+ target_lang]

sentences_in_combined_target_doc = doc_to_sentence(combined_target_doc,target_lang)
#print(sentences_in_combined_target_doc)
#for sent in sentences_in_combined_target_doc:
#    print(sent, len(sent))
start=datetime.now()

## ======== code for evaluation =================

source_doc_count = 1
#counter = 0
gap_threshold = 0
min_length = 1
#for gap_threshold in [100]:#[0,2,5,10,15,20,40,60,100]:
#  for min_length in [0]:#:[0,1,2,5,10,15]:
recall_numerator = 0
precision_numerator = 0
precision_denominator = 0
for doc in data1[:source_doc_count]:
    #print(counter)
    #counter+=1
    
    
    doc_s = doc['content_'+source_lang]
    source_embedd =  doc['embed_'+source_lang]

    target_doc = doc['content_' + target_lang]
    actual_target_sentences_pre = doc_to_sentence(target_doc, target_lang)
    actual_target_sentences=[]
    for sentence in actual_target_sentences_pre:
        if(len(sentence_to_word(sentence, target_lang))>2):
          # print(sentence)
            actual_target_sentences.append(sentence)
    #print('**************')
    #print(len(source_embedd))

    matrix, dict_s = get_similarity_matrix(source_embedd, combined_target_embedd)
    sentences_in_source = doc_to_sentence(doc_s,source_lang)
    #print(len(sentences_in_source))
    diagonals = sequence_matching(matrix, dict_s, sentences_in_source, sentences_in_combined_target_doc)
    #print(diagonals)
    diagonals = post_processor(diagonals, gap_threshold+1, min_length) 
    #print(diagonals)

    true_predicted_target_sentences = set()
    predicted_target_sentences = set()

    for index,diagonal in enumerate(diagonals):
        partial_source = []
        partial_target = []
    
        notVal = True
        for j in diagonal[1]:
  
            sent = sentences_in_combined_target_doc[j]
            if(len(sentence_to_word(sent, target_lang))>2):
                #print(sent)
                predicted_target_sentences.add(sent)
          
                if(sent in actual_target_sentences):
                    #print(j)
                    #print(sent)
                    notVal = False
                    true_predicted_target_sentences.add(sent)
            #else:
              #print(j)
            #  print(sent)
        #if(notVal):
        #    for d in diagonal[0]:
        #        print(sentences_in_source[d])
        #    print("---------")
        #    for d in diagonal[1]:
        #        print(sentences_in_combined_target_doc[d])
        #    print("===========================")
        
    
    recall_numerator += len(true_predicted_target_sentences)/len(actual_target_sentences)
    if len(predicted_target_sentences) > 0 :
        precision_numerator += len(true_predicted_target_sentences)/len(predicted_target_sentences)
        precision_denominator+=1

print("gap_thresold: ", gap_threshold )
print("min_length: ", min_length)
#print(datetime.now()-start)
recall = recall_numerator / source_doc_count
print("recall: ",recall)

if(precision_denominator>0):
    precision = precision_numerator/ precision_denominator
    print("precision: ",precision)
else:
    print('precision not defined')
f1 = (2*precision*recall)/(precision+recall)
print("f1: ", f1)