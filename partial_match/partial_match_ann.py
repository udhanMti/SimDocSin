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

def get_similarity_matrix(embeds1, embeds2):
    matrix = []
    dict_s = {}
    dict_t = {}

    k = 5
    for i,sent_s in enumerate(embeds1):
        #dict_s[i] = [-1,[]]
        lst = u.get_nns_by_vector(sent_s, 10, 100000)

        x = []
        for j in lst:

            sent_t = embeds2[j]

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
            # print(sentences_t[i[1]])
            if (i[1] not in t):
                t.append(i[1])


            if (i[0] not in s):
                s.append(i[0])


        if(len(t)>0):
            diagonals.append((s,t))
        for each in d:
            if(each in matrix):
                matrix.remove(each)
    return diagonals



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

start=datetime.now()

true_count = 0
false_count_s = 0
false_count_t = 0
partial_count = 0
false_single_sentence_count=0
start=datetime.now()
partials = []
for doc in data1[:200]:
    doc_s = doc['content_'+source_lang]
    source_embedd =  doc['embed_'+source_lang]#get_embeddig_list(doc_s)
    sentences_s = doc_to_sentence(doc_s,source_lang)
    matrix, dict_s = get_similarity_matrix(source_embedd, target_embedd)
    diagonals = sequence_matching(matrix, dict_s, sentences_s, sentences_t)


    source_splitted = doc_to_sentence(doc['content_'+target_lang],target_lang)

    ans =''
    partially = False
    falsePositive = False

    partially_matched = []


    for i in source_splitted:
        ans += i
    for i in diagonals:
        #candidate_splitted = doc_to_sentence(i,target_lang)
        s=''
        for sent in i[1]:
            s+=sentences_t[sent]
        if (s == ans):
            # print(i)
            # print(z)
            true_count += 1


    if (falsePositive):
        false_count_s += 1
    if (partially):
        partial_count += 1
        partial_ratio = len(set(partially_matched))/float(len(source_splitted))
        partials.append(partial_ratio)

print(datetime.now()-start)
print(true_count)
print(partial_count)
print(false_count_s)
print(false_count_t)



