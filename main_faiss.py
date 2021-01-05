import json
#from laser_control import get_embeddig_list
from greedy_mover_distance import greedy_mover_distance
from doc_matcher import competitive_matching, best_matching_2, best_matching
from datetime import datetime
#from margin_base_ann import margin_base_score
#from annoy import AnnoyIndex
import collections
from extract_digits import get_digit_similarity
from weight_schema import *
import faiss
import numpy as np

file  = open('embedded_data/embedding_army_0_1000.json',encoding='utf8')
data = json.load(file)
print(len(data))
# print(sentence_count_web_domain(en_documents))
map_file =  open('ann/sent_to_doc_map.json',encoding='utf8')
maps = json.load(map_file)

f=1024
#u = AnnoyIndex(f, 'euclidean')
#u.load('ann/test.ann')

res = faiss.StandardGpuResources()
cpu_index = faiss.read_index('ann/faiss_index.index')
index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

'''
#res = faiss.StandardGpuResources()
cpu_index = faiss.read_index('ann/faiss_index.index')
index = cpu_index#faiss.index_cpu_to_gpu(res, 0, cpu_index)
'''
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

start=datetime.now()
for docs in data[:1000]:
    doc_en = docs['content_en']
    doc_si = docs['content_si']

    en_documents.append(doc_en)
    si_documents.append(doc_si)

    # Get laser embedding
    #source_embedd = get_embeddig_list(doc_en)
    #target_embedd = get_embeddig_list(doc_si)
    source_embedd = docs ['embed_en']
    target_embedd = docs ['embed_si']
    source_docs.append(source_embedd)
    target_docs.append(target_embedd)



    #get frequency weights
    # source_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(source_embedd)))
    # target_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(target_embedd)))

    #get sentence length weights
    en_weight= get_sentence_length_weighting_list(doc_en, "en")
    si_weight = get_sentence_length_weighting_list(doc_si, "si")
    #source_docs_weights_sent_len.append(en_weight)
    #target_docs_weights_sent_len.append(si_weight)
    source_docs_weights_sent_len_normalized.append(documentMassNormalization(en_weight))
    target_docs_weights_sent_len_normalized.append(documentMassNormalization(si_weight))

    #source_docs_weights_sent_len_normalized.append(docs['weight_en'])
    #target_docs_weights_sent_len_normalized.append(docs['weight_si'])

    # source_docs_weights_intra_doc_word_idf.append(documentMassNormalization(get_intra_doc_word_idf_weighting_list(doc_en)))
    # target_docs_weights_intra_doc_word_idf.append(documentMassNormalization(get_intra_doc_word_idf_weighting_list(doc_si)))



# print(source_docs_weights)
# print(source_docs_weights_sent_len_normalized)
# print(target_docs_weights)
# print(target_docs_weights_sent_len_normalized)

print(datetime.now()-start)

k=10
scores ={}
for i in range(len(source_docs)):
    all_docs = []
    for embed in source_docs[i]:
        #lst = u.get_nns_by_vector(embed,10, 100000)
        xq = []
        xq.append(embed)
        np_xq = np.array(xq).astype('float32')
        D, I = index.search(np_xq, k) 

        for sent in I[0]:
            #print(i,sent)
            all_docs.append(maps[str(sent)])
            #print(sent, maps[str(sent)])
    counts = collections.Counter(all_docs)
    new_list = sorted(all_docs, key=counts.get, reverse=True)

    all_docs = []#new_list[:10] #[]
    for y in new_list:
        if len(all_docs) == 20:
            break
        if y not in all_docs:
            all_docs.append(y)
    for j in all_docs:
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_weights_sent_len_normalized[i].copy()
                                            ,target_docs_weights_sent_len_normalized[j].copy())

#scores =  margin_base_score(source_docs,target_docs,source_docs_weights_sent_len_normalized,target_docs_weights_sent_len_normalized)
# print(scores)
sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
#print(sorted_scores)
matches_s, matches_t = best_matching_2(sorted_scores)
# print(matches_s)
# print(matches_t)
count_s = 0.0
for key in matches_s.keys():
    #if (matches_s[key][0] == key):
    #    count_s += 1
    a = matches_s[key][0][1]
    b = 0.0
    k = len(matches_s[key])
    for t in range(1,k):
        b+=matches_s[key][t][1]

    if ( b!=0 and (a/b)>24):
            s_digits = source_digits[key]
            if(len(s_digits)>0):
                for candidate in matches_s[key]:
                    digit_similarity = get_digit_similarity(s_digits,target_digits[candidate[0]])
                    if(digit_similarity>0.3):
                        if(candidate[0]==key):
                            count_s+=1
                        break
                else:
                    if (matches_s[key][0][0]==key):
                        count_s+=1
            else:
                if (matches_s[key][0][0]==key):
                    count_s+=1
    else:
        if (matches_s[key][0][0]==key):
                    count_s+=1
  
count_t = 0.0
for key in matches_t.keys():
    #if (matches_t[key][0] == key):
    #    count_t += 1

    a = matches_t[key][0][1]
    b = 0.0
    k = len(matches_t[key])
    for t in range(1,k):
        b+=matches_t[key][t][1]
    if( b!=0 and (a/b)>24):
        s_digits = target_digits[key]
        if(len(s_digits)>0):
            for candidate in matches_t[key]:
                digit_similarity = get_digit_similarity(s_digits,source_digits[candidate[0]])
                if(digit_similarity>0.3):
                    if(candidate[0]==key):
                        count_t+=1
                    break
            else:
                if (matches_t[key][0][0]==key):
                    count_t+=1
        else:
            if (matches_t[key][0][0]==key):
                count_t+=1
    else:
        if (matches_t[key][0][0]==key):
                    count_t+=1
  
print(count_s, " ", count_t)

print(datetime.now()-start)



count_s = 0.0
for key in matches_s.keys():
    #if (matches_s[key][0] == key):
    #    count_s += 1
    a = matches_s[key][0][1]
    b = 0.0
    k = len(matches_s[key])
    for t in range(1,k):
        b+=matches_s[key][t][1]

    if ( b!=0 and (a/b)>23):
            s_digits = source_digits[key]
            if(len(s_digits)>0):
                for candidate in matches_s[key]:
                    digit_similarity = get_digit_similarity(s_digits,target_digits[candidate[0]])
                    if(digit_similarity>0.5):
                        if(candidate[0]==key):
                            count_s+=1
                        break
                else:
                    if (matches_s[key][0][0]==key):
                        count_s+=1
            else:
                if (matches_s[key][0][0]==key):
                    count_s+=1
    else:
        if (matches_s[key][0][0]==key):
                    count_s+=1
  
count_t = 0.0
for key in matches_t.keys():
    #if (matches_t[key][0] == key):
    #    count_t += 1

    a = matches_t[key][0][1]
    b = 0.0
    k = len(matches_t[key])
    for t in range(1,k):
        b+=matches_t[key][t][1]
    if( b!=0 and (a/b)>23):
        s_digits = target_digits[key]
        if(len(s_digits)>0):
            for candidate in matches_t[key]:
                digit_similarity = get_digit_similarity(s_digits,source_digits[candidate[0]])
                if(digit_similarity>0.5):
                    if(candidate[0]==key):
                        count_t+=1
                    break
            else:
                if (matches_t[key][0][0]==key):
                    count_t+=1
        else:
            if (matches_t[key][0][0]==key):
                count_t+=1
    else:
        if (matches_t[key][0][0]==key):
                    count_t+=1
  
print(count_s, " ", count_t)

print(datetime.now()-start)
#print('*******************************************************************************************************')
