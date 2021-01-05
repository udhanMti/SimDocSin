import numpy as np
import json
from laser_control import get_embeddig_list
from doc_to_sentence import doc_to_sentence
from datetime import datetime

def get_similarity_matrix(embeds1, embeds2):
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

        s = []
        #print(d)
        #for i in d:
        #    print(sentences_s[i[0]])
        added = []
        for i in d:
            #print(sentences_t[i[1]])
            if(i[1] not in added):
                #s+=sentences_t[i[1]]
                s.append(sentences_t[i[1]])
                added.append(i[1])
        if(len(s)>0):
            diagonals.append(s)

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

target_embedd = data3[0]['embed_'+target_lang] #get_embeddig_list(long_doc, "si")
sentences_t = doc_to_sentence(long_doc,target_lang)
true_count = 0
false_count_s = 0
false_count_t = 0
partial_count = 0
print(len(sentences_t))
start=datetime.now()

partials = []
count=0
for doc in data1[:5]:
    #print(count)
    count+=1
    doc_s = doc['content_'+source_lang]
    source_embedd = doc['embed_'+source_lang]#get_embeddig_list(doc_s)
    sentences_s = doc_to_sentence(doc_s,source_lang)
    matrix, dict_s = get_similarity_matrix(source_embedd, target_embedd)#, sentences_s, sentences_t)
    #if((z==20) | (z==33)):
    #print(matrix)
    diagonals = sequence_matching(matrix,dict_s, sentences_s, sentences_t)

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
        for sent in i:
            s+=sent
        if (s==ans):

            true_count+=1
        
    
        else:
            
            matching_list = list(set(i) & set(source_splitted))
            if(len(matching_list)>0):
                partially = True
                partially_matched.extend(matching_list)
            else:
                false_count_t+=1
                #print('false')
                falsePositive = True
            

    if(falsePositive):
        false_count_s += 1
    if (partially):
        #print('partial')
        partial_count += 1
        partial_ratio = len(set(partially_matched))/float(len(source_splitted)) 
        partials.append(partial_ratio)





print(datetime.now()-start)
print(true_count)
print(partial_count)
print(false_count_s)
print(false_count_t)


