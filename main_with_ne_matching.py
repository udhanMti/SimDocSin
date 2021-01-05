import json
from laser_control import get_embeddig_list
from greedy_mover_distance import greedy_mover_distance
from weight_schema import *
from doc_matcher import competitive_matching, best_matching, best_matching_2
from datetime import datetime
from extract_ne import extract_names, extract_designations, get_ne_similarity

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

# print(source_docs_weights)
# print(source_docs_weights_sent_len)
# print(target_docs_weights)
# print(target_docs_weights_sent_len)

print(datetime.now() - start)

scores = {}
# count = 0
for i in range(len(source_docs)):
    # min_distance = 1000000
    # target = -1
    print(i)
    for j in range(len(target_docs)):
        
        source_doc = source_docs[i].copy()
        target_doc = target_docs[j].copy()
        # if((len(source_doc)>(len(target_doc)*1.5)) | (len(target_doc)>(len(source_doc)*1.5))):
        #    continue
        distance = greedy_mover_distance(source_doc, target_doc, source_docs_weights_sent_len_normalized[i].copy()
                                         , target_docs_weights_sent_len_normalized[j].copy())
        scores[(i, j)] = distance
        # if(distance<min_distance):
        #    min_distance=distance
        #    target = j
        # if(i==target):
        #   count+=1
# print(count)
print(datetime.now() - start)

sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

'''matches = competitive_matching(sorted_scores)
match = matches[0]
match_2 = matches[1]
match_3 = matches[2]
# print(match)
print(datetime.now()-start)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
    else:
        print('----------------------------------------------------')
        print(en_documents[pair[0]])
        print(si_documents[pair[1]])
        print('----------------------------------------------------')
print(count)
print('*********************************************')
for pair in match_2:
    if (pair[0]==pair[1]):
        count+=1
        print('----------------------------------------------------')
        print(en_documents[pair[0]])
        print(si_documents[pair[1]])
        print('----------------------------------------------------')
print(count)
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
for pair in match_3:
    if (pair[0]==pair[1]):
        count+=1
        print('----------------------------------------------------')
        print(en_documents[pair[0]])
        print(si_documents[pair[1]])
        print('----------------------------------------------------')
print(count)
print ("Matched document pairs for word-idf-intra-doc weighting ",count*100/len(match))'''

matches_s, matches_t = best_matching_2(sorted_scores)
count_s = 0.0
for key in matches_s.keys():
    # gap_1 = matches_s[key][1][1] - matches_s[key][0][1]
    # gap_2 = matches_s[key][2][1] - matches_s[key][1][1]
    a = matches_s[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_s[key][t][1]
    if ((a / b) > 0.24):
        s_names = source_names[key]
        s_designations = source_designations[key]
        for candidate in matches_s[key]:
                ne_similarity = get_ne_similarity(s_names, s_designations, si_documents[candidate[0]])
                if (ne_similarity > 0.3):
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
    # gap_1 = matches_t[key][1][1] - matches_t[key][0][1]
    # gap_2 = matches_t[key][2][1] - matches_t[key][1][1]
    a = matches_t[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_t[key][t][1]
    if ((a / b) > 0.24):

        for candidate in matches_t[key]:
                ne_similarity = get_ne_similarity(source_names[candidate[0]], source_designations[candidate[0]], si_documents[key])
                if (ne_similarity > 0.3):
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




count_s = 0.0
for key in matches_s.keys():
    # gap_1 = matches_s[key][1][1] - matches_s[key][0][1]
    # gap_2 = matches_s[key][2][1] - matches_s[key][1][1]
    a = matches_s[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_s[key][t][1]
    if ((a / b) > 0.24):
        s_names = source_names[key]
        s_designations = source_designations[key]
        for candidate in matches_s[key]:
                ne_similarity = get_ne_similarity(s_names, s_designations, si_documents[candidate[0]])
                if (ne_similarity > 0.3):
                    if (candidate[0] == key):
                        count_s += 1
                        

                    break


    else:
        if (matches_s[key][0][0] == key):
            count_s += 1

count_t = 0.0
for key in matches_t.keys():
    # gap_1 = matches_t[key][1][1] - matches_t[key][0][1]
    # gap_2 = matches_t[key][2][1] - matches_t[key][1][1]
    a = matches_t[key][0][1]
    b = 0.0
    for t in range(1, 5):
        b += matches_t[key][t][1]
    if ((a / b) > 0.24):
        for candidate in matches_t[key]:
                ne_similarity = get_ne_similarity(source_names[candidate[0]], source_designations[candidate[0]], si_documents[key])
                if (ne_similarity > 0.3):
                    if (candidate[0] == key):
                        count_t += 1
                       
                    break


    else:
        if (matches_t[key][0][0] == key):
            count_t += 1

print(count_s, " ", count_t)



count_s = 0.0
for key in matches_s.keys():
    if (matches_s[key][0][0] == key):
            count_s += 1

count_t = 0.0
for key in matches_t.keys():
    if (matches_t[key][0][0] == key):
            count_t += 1

print(count_s, " ", count_t)
'''

for key in matches_s.keys():
    if (matches_s[key][1][0]==key):
        count_s+=1
     #   print(en_documents[key])
     #   print(si_documents[matches_s[key][1][0]])
     #   print(matches_s[key][0][1],matches_s[key][1][1],matches_s[key][2][1])
     #   print('-------------------------------------------------------')
for key in matches_t.keys():
    if (matches_t[key][1][0]==key):
        count_t+=1
     #   print(si_documents[key])
     #   print(en_documents[matches_t[key][1][0]])
     #   print(matches_t[key][0][1],matches_t[key][1][1],matches_t[key][2][1])
     #   print('-------------------------------------------------------')

print(count_s, " ", count_t)

for key in matches_s.keys():
    if (matches_s[key][2][0]==key):
        count_s+=1
      #  print(en_documents[key])
      #  print(si_documents[matches_s[key][2][0]])
      #  print(matches_s[key][0][1],matches_s[key][1][1],matches_s[key][2][1])
      #  print('-------------------------------------------------------')

for key in matches_t.keys():
    if (matches_t[key][2][0]==key):
        count_t+=1
     #   print(si_documents[key])
     #   print(en_documents[matches_t[key][2][0]])
     #   print(matches_t[key][0][1],matches_t[key][1][1],matches_t[key][2][1])
     #   print('-------------------------------------------------------')

print(count_s, " ", count_t)

print ("Matched document pairs for word-idf-intra-doc weighting ",count_s*100/len(matches_s)," ",count_t*100/len(matches_t))
'''
'''
word_count_en= word_count_over_docs(en_documents, "en")
word_count_si= word_count_over_docs(si_documents, "si")
N_en=len(en_documents)
N_si=len(si_documents)
#source_docs_weights_idf=[]
#target_docs_weights_idf=[]

source_docs_weights_inter_doc_word_idf_normalized=[]
target_docs_weights_inter_doc_word_idf_normalized=[]

for doc in en_documents:
    weight_doc= get_inter_doc_word_idf_weighting_list(doc,word_count_en,N_en, "en")
    #source_docs_weights_idf.append(weight_doc)
    source_docs_weights_inter_doc_word_idf_normalized.append(documentMassNormalization(weight_doc))

for doc in si_documents:
    weight_doc=get_inter_doc_word_idf_weighting_list(doc,word_count_si,N_si, "si")
    #target_docs_weights_idf.append(weight_doc)
    target_docs_weights_inter_doc_word_idf_normalized.append(documentMassNormalization(weight_doc))

scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_weights_inter_doc_word_idf_normalized[i].copy()
                                            ,target_docs_weights_inter_doc_word_idf_normalized[j].copy())

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

match=competitive_matching(sorted_scores)
# print(match)
print(datetime.now()-start)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for word-idf-inter-doc weighting",count*100/len(match))



scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i].copy(),target_docs[j].copy(),source_docs_weights[i].copy()
                                            ,target_docs_weights[j].copy())


sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

match=competitive_matching(sorted_scores)
# print(match)
print(datetime.now()-start)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for sentence frequency ",count*100/len(match))
'''
'''

scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_weights_sent_len_normalized[i].copy()
                                            ,target_docs_weights_sent_len_normalized[j].copy())

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}'''
'''
match=competitive_matching(sorted_scores)
# print(match)
print(datetime.now()-start)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for sentence length ",count*100/len(match))
'''
'''
matches_s,matches_t = best_matching_2(sorted_scores)
count_s=0.0
for key in matches_s.keys():
    if (matches_s[key][0][0]==key):
        count_s+=1
    #else:
        #print(en_documents[matches_s[key][0][0]])
        #print(si_documents[key])
        #print('-------------------------------------------------------')

count_t=0.0
for key in matches_t.keys():
    if (matches_t[key][0][0]==key):
        count_t+=1


print(count_s, " ", count_t)

for key in matches_s.keys():
    if (matches_s[key][1][0]==key):
        count_s+=1
        #print(en_documents[matches_s[key][1][0]])
        #print(si_documents[key])
        #print('-------------------------------------------------------')

for key in matches_t.keys():
    if (matches_t[key][1][0]==key):
        count_t+=1

print(count_s, " ", count_t)

for key in matches_s.keys():
    if (matches_s[key][2][0]==key):
        count_s+=1
        #print(en_documents[matches_s[key][2][0]])
        #print(si_documents[key])
        #print('-------------------------------------------------------')

for key in matches_t.keys():
    if (matches_t[key][2][0]==key):
        count_t+=1

print(count_s, " ", count_t)

print ("Matched document pairs for sentence length ",count_s*100/len(matches_s)," ",count_t*100/len(matches_t))
'''
'''
sentence_count_en= sentence_count_web_domain(en_documents)
sentence_count_si= sentence_count_web_domain(si_documents)
N_en=len(en_documents)
N_si=len(si_documents)
source_docs_weights_idf=[]
target_docs_weights_idf=[]

source_docs_weights_idf_normalized=[]
target_docs_weights_idf_normalized=[]

for doc in en_documents:
    weight_doc= get_idf_weighting_list(doc,sentence_count_en,N_en)
    source_docs_weights_idf.append(weight_doc)
    source_docs_weights_idf_normalized.append(documentMassNormalization(weight_doc))

for doc in si_documents:
    weight_doc=get_idf_weighting_list(doc,sentence_count_si,N_si)
    target_docs_weights_idf.append(weight_doc)
    target_docs_weights_idf_normalized.append(documentMassNormalization(weight_doc))

scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_weights_idf_normalized[i].copy()
                                            ,target_docs_weights_idf_normalized[j].copy())

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

match=competitive_matching(sorted_scores)
# print(match)
print(datetime.now()-start)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for idf weighting",count*100/len(match))

source_docs_slidf_weight= get_slidf_weighting_list(source_docs_weights_sent_len.copy()
                                                                             ,source_docs_weights_idf.copy())
target_docs_slidf_weight= get_slidf_weighting_list(target_docs_weights_sent_len.copy()
                                                                             ,target_docs_weights_idf.copy())



scores ={}

for i in range(len(source_docs)):
    for j in range(len(target_docs)):
        scores[(i,j)]=greedy_mover_distance(source_docs[i],target_docs[j],source_docs_slidf_weight[i].copy()
                                            ,target_docs_slidf_weight[j].copy())

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

match=competitive_matching(sorted_scores)
# print(match)
print(datetime.now()-start)

count=0.0
for pair in match:
    if (pair[0]==pair[1]):
        count+=1
print ("Matched document pairs for slidf weighting",count*100/len(match))

# print(source_docs_weights)
# print(source_docs_weights_sent_len)
# print(source_docs_weights_sent_len_normalized)
# print(source_docs_weights_idf)
# print(source_docs_weights_idf_normalized)
# print(source_docs_slidf_weight)
'''
