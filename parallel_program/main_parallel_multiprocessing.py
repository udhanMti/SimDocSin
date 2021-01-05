import json
from datetime import datetime

from doc_matcher import best_matching
from parallel_program.greedy_mover_distance_with_multiprocessing import greedy_mover_distance
from weight_schema import *

#from doc_to_sentence import *
#from scipy.stats import norm
#import matplotlib.pyplot as plt

'''file  = open('data/fyp16_docset_removed.json',encoding='utf8')
data = json.load(file)'''
file  = open('./embedded_data/embedding_army_0_1000.json',encoding='utf8')
data = json.load(file)

# print(sentence_count_web_domain(en_documents))

source_docs = []
target_docs = []
source_docs_weights=[]
target_docs_weights=[]

source_docs_weights_sent_len=[]
target_docs_weights_sent_len=[]
source_docs_weights_sent_len_normalized=[]
target_docs_weights_sent_len_normalized=[]

en_documents=[]
si_documents=[]

source_docs_weights_intra_doc_word_idf = []
target_docs_weights_intra_doc_word_idf = []

source_digits = []
target_digits = []

source_names = []
source_designations = []


i=0
start=datetime.now()
for docs in data[:1000]:
    i+=1
    #print(i)
    doc_en = docs['content_en']
    doc_si = docs['content_si']

    en_documents.append(doc_en)
    si_documents.append(doc_si)

    
    source_embedd = docs ['embed_en']#get_embeddig_list(doc_en)
    target_embedd = docs ['embed_si']#get_embeddig_list(doc_si, "si")
    source_docs.append(source_embedd)
    target_docs.append(target_embedd)

    #source_digits.append(docs['numbers_en'])
    #target_digits.append(docs['numbers_si'])

    #source_names.append(docs['names_en'])
    #source_designations.append(docs['designations_en'])

    #get frequency weights
    #source_docs_weights.append(docs['weights_freq_en'])
    #target_docs_weights.append(docs['weights_freq_si'])

    #get sentence length weights
    en_weight= get_sentence_length_weighting_list(doc_en, "en")
    si_weight = get_sentence_length_weighting_list(doc_si, "si")
    #source_docs_weights_sent_len.append(en_weight)
    #target_docs_weights_sent_len.append(si_weight)
    source_docs_weights_sent_len_normalized.append(documentMassNormalization(en_weight))
    target_docs_weights_sent_len_normalized.append(documentMassNormalization(si_weight))

    #source_docs_weights_intra_doc_word_idf.append(docs['weights_word_idf_en'])
    #target_docs_weights_intra_doc_word_idf.append(docs['weights_word_idf_si'])



# print(source_docs_weights)
# print(source_docs_weights_sent_len)
# print(target_docs_weights)
# print(target_docs_weights_sent_len)

print(datetime.now()-start)

#print(mp.cpu_count())

scores ={}
for i in range(len(source_docs)):
   print(i)
   source_doc = source_docs[i].copy()
   for j in range(len(target_docs)): 
        target_doc = target_docs[j].copy()
        #if((len(source_doc)>(len(target_doc)*1.5)) | (len(target_doc)>(len(source_doc)*1.5))):
        #    continue
        distance=greedy_mover_distance(source_doc,target_doc,source_docs_weights_sent_len_normalized[i].copy()
                                            ,target_docs_weights_sent_len_normalized[j].copy())
        scores[(i,j)] = distance

#print(scores)
print(datetime.now()-start)

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

print(datetime.now()-start)

matches_s,matches_t = best_matching(sorted_scores)
count_s=0.0
for key in matches_s.keys():
    if (matches_s[key][0]==key):
        count_s+=1


count_t=0.0
for key in matches_t.keys():
    
    if (matches_t[key][0]==key):
        count_t+=1
        

print(count_s, " ", count_t)

print(datetime.now()-start)