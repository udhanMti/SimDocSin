import json
from laser_control import get_embeddig_list
from greedy_mover_distance import greedy_mover_distance
from weight_schema import *
from doc_matcher import competitive_matching, best_matching, best_matching_2
from datetime import datetime
from extract_ne import extract_names, extract_designations, get_ne_similarity
from extract_digits import extract_digits, get_digit_similarity

file = open('data/army_parallel_new.json', encoding='utf8')
data = json.load(file)

# print(sentence_count_web_domain(en_documents))

source_docs = []
target_docs = []
source_docs_weights = []
target_docs_weights = []

source_docs_weights_sent_len = []
target_docs_weights_sent_len = []
source_docs_weights_sent_len_normalized = []
target_docs_weights_sent_len_normalized = []

en_documents = []
si_documents = []

source_docs_weights_intra_doc_word_idf = []
target_docs_weights_intra_doc_word_idf = []

source_names = []
source_designations = []

source_digits = []
target_digits = []

i = 0
start = datetime.now()
for docs in data[:500]:
    i += 1
    print(i)
    doc_en = docs['content_en']
    doc_si = docs['content_si']

    en_documents.append(doc_en)
    si_documents.append(doc_si)

    # Get laser embedding
    source_embedd = get_embeddig_list(doc_en)
    target_embedd = get_embeddig_list(doc_si, "si")
    source_docs.append(source_embedd)
    target_docs.append(target_embedd)

    source_names.append(extract_names(doc_en))
    source_designations.append(extract_designations(doc_en))

    source_digits.append(extract_digits(doc_en, "en"))
    target_digits.append(extract_digits(doc_si, "si"))

    # get frequency weights
    source_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(source_embedd)))
    target_docs_weights.append(documentMassNormalization(get_sentence_frequency_list(target_embedd)))

    # get sentence length weights
    en_weight = get_sentence_length_weighting_list(doc_en, "en")
    si_weight = get_sentence_length_weighting_list(doc_si, "si")
    source_docs_weights_sent_len.append(en_weight)
    target_docs_weights_sent_len.append(si_weight)
    source_docs_weights_sent_len_normalized.append(documentMassNormalization(en_weight))
    target_docs_weights_sent_len_normalized.append(documentMassNormalization(si_weight))

    source_docs_weights_intra_doc_word_idf.append(
        documentMassNormalization(get_intra_doc_word_idf_weighting_list(doc_en, "en")))
    target_docs_weights_intra_doc_word_idf.append(
        documentMassNormalization(get_intra_doc_word_idf_weighting_list(doc_si, "si")))


print(datetime.now() - start)

scores = {}
# count = 0
for i in range(len(source_docs)):

    for j in range(len(target_docs)):
        
        source_doc = source_docs[i].copy()
        target_doc = target_docs[j].copy()

        distance = greedy_mover_distance(source_doc, target_doc, source_docs_weights_sent_len_normalized[i].copy()
                                         , target_docs_weights_sent_len_normalized[j].copy())
        scores[(i, j)] = distance

print(datetime.now() - start)

sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

matches_s, matches_t = best_matching_2(sorted_scores)
count_s = 0.0
for key in matches_s.keys():
    a = matches_s[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_s[key][t][1]
    if ((a / b) > 0.24):
        s_names = source_names[key]
        s_designations = source_designations[key]
        s_digits = source_digits[key]
        for candidate in matches_s[key]:
                ne_similarity = get_ne_similarity(s_names, s_designations, si_documents[candidate[0]])
                digit_similarity = get_digit_similarity(s_digits,target_digits[candidate[0]])
                if (ne_similarity + digit_similarity > 0.8):
                    if (candidate[0] == key):
                        count_s += 1

                    break
        else:
            if (matches_s[key][0][0] == key):
                count_s += 1


    else:
        if (matches_s[key][0][0] == key):
            count_s += 1

count_t = 0.0
for key in matches_t.keys():
    a = matches_t[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_t[key][t][1]
    if ((a / b) > 0.24):
        s_digits = target_digits[key]
        for candidate in matches_t[key]:
                ne_similarity = get_ne_similarity(source_names[candidate[0]], source_designations[candidate[0]], si_documents[key])
                digit_similarity = get_digit_similarity(s_digits,source_digits[candidate[0]])
                if (ne_similarity + digit_similarity> 0.8):
                    if (candidate[0] == key):
                        count_t += 1
                        
                    break
        else:
            if (matches_t[key][0][0] == key):
                count_t += 1


    else:
        if (matches_t[key][0][0] == key):
            count_t += 1

print(count_s, " ", count_t)


print(datetime.now() - start)

count_s = 0.0
for key in matches_s.keys():
    a = matches_s[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_s[key][t][1]
    if ((a / b) > 0.23):
        s_names = source_names[key]
        s_designations = source_designations[key]
        s_digits = source_digits[key]
        for candidate in matches_s[key]:
                ne_similarity = get_ne_similarity(s_names, s_designations, si_documents[candidate[0]])
                digit_similarity = get_digit_similarity(s_digits,target_digits[candidate[0]])
                if (ne_similarity + digit_similarity > 0.8):
                    if (candidate[0] == key):
                        count_s += 1

                    break
        else:
            if (matches_s[key][0][0] == key):
                count_s += 1


    else:
        if (matches_s[key][0][0] == key):
            count_s += 1

count_t = 0.0
for key in matches_t.keys():
    a = matches_t[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_t[key][t][1]
    if ((a / b) > 0.23):

        s_digits = target_digits[key]
        for candidate in matches_t[key]:
                ne_similarity = get_ne_similarity(source_names[candidate[0]], source_designations[candidate[0]], si_documents[key])
                digit_similarity = get_digit_similarity(s_digits,source_digits[candidate[0]])
                if (ne_similarity + digit_similarity> 0.8):
                    if (candidate[0] == key):
                        count_t += 1
                        
                    break
        else:
            if (matches_t[key][0][0] == key):
                count_t += 1


    else:
        if (matches_t[key][0][0] == key):
            count_t += 1

print(count_s, " ", count_t)


print(datetime.now() - start)
