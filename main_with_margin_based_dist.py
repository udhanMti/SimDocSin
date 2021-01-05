import json
from laser_control import get_embeddig_list
from greedy_mover_distance import greedy_mover_distance
from weight_schema import *
from doc_matcher import competitive_matching, best_matching, best_matching_2
from datetime import datetime
from doc_to_sentence import *
from margin_base_distance_calculator import margin_base_score
from extract_digits import extract_digits, get_digit_similarity

file  = open('army_parallel_new.json',encoding='utf8')
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

#175 173
start=datetime.now()
for docs in data[0:6]:
    doc_en = docs['content_en']
    doc_si = docs['content_si']

    en_documents.append(doc_en)
    si_documents.append(doc_si)

    # Get laser embedding
    source_embedd = get_embeddig_list(doc_en)
    target_embedd = get_embeddig_list(doc_si, "si")
    source_docs.append(source_embedd)
    target_docs.append(target_embedd)

    source_digits.append(extract_digits(doc_en, "en"))
    target_digits.append(extract_digits(doc_si, "si"))

    #get frequency weights
    source_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(doc_en,"en")))
    target_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(doc_si,"si")))

    #get sentence length weights
    en_weight= get_sentence_length_weighting_list(doc_en, "en")
    si_weight = get_sentence_length_weighting_list(doc_si, "si")
    source_docs_weights_sent_len.append(en_weight)
    target_docs_weights_sent_len.append(si_weight)
    source_docs_weights_sent_len_normalized.append(documentMassNormalization(en_weight))
    target_docs_weights_sent_len_normalized.append(documentMassNormalization(si_weight))

    source_docs_weights_intra_doc_word_idf.append(documentMassNormalization(get_intra_doc_word_idf_weighting_list(doc_en, "en")))
    target_docs_weights_intra_doc_word_idf.append(documentMassNormalization(get_intra_doc_word_idf_weighting_list(doc_si, "si")))


print(datetime.now()-start)


scores = margin_base_score(source_docs,target_docs,source_docs_weights_sent_len_normalized,target_docs_weights_sent_len_normalized)#{}


sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

matches_s,matches_t = best_matching_2(sorted_scores)

count_s = 0.0
total = 0
for key in matches_s.keys():
    total += 1
    if (matches_s[key][0] == key):
        count_s += 1


count_t = 0.0
for key in matches_t.keys():

    if (matches_t[key][0] == key):
        count_t += 1


print(count_s, " ", count_t, total)
