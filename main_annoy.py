import json
from greedy_mover_distance import greedy_mover_distance
from doc_matcher import competitive_matching, best_matching_2, best_matching
from datetime import datetime
#from margin_base_ann import margin_base_score
from annoy import AnnoyIndex
import collections
from extract_digits import *
from weight_schema import *
from extract_ne import extract_names, extract_designations, get_ne_similarity


file  = open('embedded_data/embedding_army_0_1000.json',encoding='utf8')
data = json.load(file)
print(len(data))
# print(sentence_count_web_domain(en_documents))
map_file =  open('ann/sent_to_doc_map.json',encoding='utf8')
maps = json.load(map_file)

f=1024
u = AnnoyIndex(f, 'euclidean')
u.load('ann/test.ann')

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

    source_names.append(extract_names(doc_en))
    source_designations.append(extract_designations(doc_en))

    source_digits.append(extract_digits(doc_en, "en"))
    target_digits.append(extract_digits(doc_si, "si"))

    en_weight= get_sentence_length_weighting_list(doc_en, "en")
    si_weight = get_sentence_length_weighting_list(doc_si, "si")

    source_docs_weights_sent_len_normalized.append(documentMassNormalization(en_weight))
    target_docs_weights_sent_len_normalized.append(documentMassNormalization(si_weight))

print(datetime.now()-start)

scores ={}
for i in range(len(source_docs)):
    all_docs = []
    for embed in source_docs[i]:
        lst = u.get_nns_by_vector(embed,10, 100000)
        for sent in lst:

            all_docs.append(maps[str(sent)])

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


sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

matches_s, matches_t = best_matching(sorted_scores)
count_s = 0.0
for key in matches_s.keys():
    if (matches_s[key][0] == key):
        count_s += 1

count_t = 0.0
for key in matches_t.keys():

    if (matches_t[key][0] == key):
        count_t += 1

print(count_s, " ", count_t)

print(datetime.now() - start)

