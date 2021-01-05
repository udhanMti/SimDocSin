
import numpy as np
import json
from laser_control import get_embeddig_list
from doc_to_sentence import doc_to_sentence
from datetime import datetime
from annoy import AnnoyIndex
from greedy_mover_distance import greedy_mover_distance
from weight_schema import *

f=1024
u = AnnoyIndex(f, 'euclidean')
u.load('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/ann/test.ann')

def get_matching_sequence(i,sentences_s, sentences_t):

    sequence = []
    j_s = 0
    j_t = 0
    while (j_s<len(sentences_s)) and (j_t<len(sentences_t)):
        sent_s = np.array(sentences_s[j_s])
        sent_t = np.array(sentences_t[j_t])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[max(0,j_s-1)])
        sent_t = np.array(sentences_t[j_t])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[j_s])
        sent_t = np.array(sentences_t[max(0,j_t-1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(cos_similarity_1>0.65):
            sequence.append(i+j_t)
            j_s+=1
            j_t+=1
        elif(cos_similarity_2>0.65):
            sequence.append(i+j_t)
            j_t+=1
        elif(cos_similarity_3>0.65):
            j_s+=1
        else:
            return [sequence,j_s]
        
    return [sequence,j_s]

def get_matching_sequence_2(x,y,sentences_s, sentences_t):
    up_x, up_y = x, y
    down_x, down_y = x, y
    sequence = []
    j = 0

    while (up_x>=0) and (up_y>=0):
        
        sent_s = np.array(sentences_s[up_x])
        sent_t = np.array(sentences_t[up_y])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[min(x,up_x+1)])
        sent_t = np.array(sentences_t[up_y])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[up_x])
        sent_t = np.array(sentences_t[min(y,up_y+1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(cos_similarity_1>0.65):
            if(up_y!=y):
                sequence.insert(0,up_y)
            up_x-=1
            up_y-=1
        elif(cos_similarity_2>0.65):
            if(up_y!=y):
                sequence.insert(0,up_y)
            up_y-=1
        elif(cos_similarity_3>0.65):
            up_x-=1
        else:
            break

    while (down_x<len(sentences_s)) and (down_y<len(sentences_t)):
        sent_s = np.array(sentences_s[down_x])
        sent_t = np.array(sentences_t[down_y])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[max(x,down_x-1)])
        sent_t = np.array(sentences_t[down_y])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[down_x])
        sent_t = np.array(sentences_t[max(y,down_y-1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(cos_similarity_1>0.65):
            sequence.append(down_y)
            down_x+=1
            down_y+=1
            j+=1
        elif(cos_similarity_2>0.65):
            sequence.append(down_y)
            down_y+=1
        elif(cos_similarity_3>0.65):
            down_x+=1
            j+=1
        else:
            return [sequence,j]
        
    return [sequence,j]

def get_matching_partials(sentences_s, sentences_t):
    
    matching_partials = []
    matching_sources = []
    i=0
    while i<len(sentences_s):
        
        lst = u.get_nns_by_vector(sentences_s[i], 3, 100000)
        best_sequence = []
        matching_source_indices = []
        matching_sequences = []
        #print(i)
        matched_count = 0
        for candidate in lst:
            #print('candidate ',candidate)
            #sequence, j = get_matching_sequence(candidate,sentences_s[i:],sentences_t[candidate:])
            
            sequence, j = get_matching_sequence_2(i,candidate,sentences_s,sentences_t)

            if(len(sequence)>len(best_sequence)):
                best_sequence = sequence
                #print(best_sequence)
                matched_count = j
            if (len(sequence)>1) :
                matching_sequences.append(sequence)
                matching_source_indices.append([i, i+j])
        
        i = i+max(matched_count,1) #last_in_sequence
        
        #if(len(best_sequence)>0):
        #    matching_partials.append(best_sequence)

        matching_partials.extend(matching_sequences)
        matching_sources.extend(matching_source_indices)

    #return matching_partials
    return  [matching_partials, matching_sources]


file1  = open('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/embedded_data/embedding_army.json',encoding='utf8')
data1 = json.load(file1)

file2  = open('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/data/longdoc_army_si.json',encoding='utf8')
data2 = json.load(file2)

file3  = open('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/embedded_data/embedding_army_si.json',encoding='utf8')
data3 = json.load(file3)

source_lang = 'en'
target_lang = 'si'

long_doc = data2['content_'+target_lang]

target_embedd = data3[0]['embed_'+ target_lang] #get_embeddig_list(long_doc, "si")
sentences_t = doc_to_sentence(long_doc,target_lang)

true_count = 0
false_count_s = 0
false_count_t = 0
partial_count = 0
false_single_sentence_count=0
start=datetime.now()
partials = []
#r=0
for doc in data1[:200]:

    doc_s = doc['content_'+source_lang]
    source_embedd =  doc['embed_'+source_lang]#get_embeddig_list(doc_s)

    sentences_s = doc_to_sentence(doc_s,source_lang)
    
   

    matching_partials, temp = get_matching_partials(source_embedd, target_embedd)

    
    ans =''
    partially = False
    falsePositive = False
    foundTrue = False

    partially_matched = []

    source_splitted = doc_to_sentence(doc['content_'+target_lang],target_lang)
    for i in source_splitted:
        ans += i
    #print(len(matching_partials))
    for partial in matching_partials:
        #print(partial)
        partial_sentences = []
        s=''
        for sent in partial:
            s+=sentences_t[sent]
            partial_sentences.append(sentences_t[sent])

        if (s == ans):
            #print('exat')
            # print(z)
            true_count += 1
            foundTrue = True

        else:
           
            
            matching_list = list(set(partial_sentences) & set(source_splitted))
            if(len(matching_list)>0):
                partially = True
                partially_matched.extend(matching_list)
            else:
                false_count_t+=1
                #print('false')
                falsePositive = True


    if (falsePositive):
        false_count_s += 1
    if (not foundTrue):
        if (partially):
            partial_count += 1
            partial_ratio = len(set(partially_matched))/float(len(source_splitted)) 
            partials.append(partial_ratio)


print(datetime.now()-start)
print(true_count)
print(partial_count)
print(false_count_s)
print(false_count_t)

