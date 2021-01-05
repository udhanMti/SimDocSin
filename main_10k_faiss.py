import json

from doc_to_sentence import doc_to_sentence 
from doc_matcher import competitive_matching, best_matching_2, best_matching
from datetime import datetime


file  = open('/content/gdrive/Shared drives/FYP/1M_db_embed_scrape/embedding_army_0_1000.json',encoding='utf8')
data = json.load(file)
print(len(data))
map_file =  open('ann/sent_to_doc_map.json',encoding='utf8')
maps = json.load(map_file)

f=1024

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

    en_documents.append(doc_en)

    source_embedd = docs ['embed_en']
    source_docs.append(source_embedd)

    source_docs_weights_sent_len_normalized.append(docs['weight_en'])


  
path = '/content/gdrive/Shared drives/FYP/1M_db_embed_scrape'
file_names = ['embedding_army_0_1000','embedding_defence_0_705',
'embedding_army_1000_1039', 'embedding_hiru_cleaned','embedding_dms_pairs_1_100','embedding_wsws_new_222',
'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
'embedding_dms_pairs_101_150','embedding_dms_pairs_151_175','embedding_dms_pairs_176_200','embedding_dms_pairs_201_225','embedding_dms_pairs_226_237',
'embedding_dms_non_pairs','embedding_itn_sinhala_0_1000','embedding_newslk_sin_0_1000','embedding_sirasa_local_sinhala_0_1000','embedding_sirasa_local_sinhala_1000_2000','embedding_hiru_40000_41000']


for file_name in file_names:
    file  = open(path + '/' +file_name + '.json' ,encoding='utf8')
    data = json.load(file)
    sent_count = 0

    if (file_name == file_names[-1]):
        for docs in data[:676]:
            doc_si = docs['content_si']
            sent_count = sent_count + len(doc_to_sentence(doc_si,'si'))
            si_documents.append(doc_si)

            target_embedd = docs['embed_si']  # get_embeddig_list(doc_si, "si")
            target_docs.append(target_embedd)
            target_docs_weights_sent_len_normalized.append(docs['weight_si'])
            #target_digits.append(extract_digits(doc_si, 'si'))

    else:
        for docs in data:
            doc_si = docs['content_si']
            sent_count = sent_count + len(doc_to_sentence(doc_si,'si'))
            si_documents.append(doc_si)

            target_embedd = docs['embed_si']  # get_embeddig_list(doc_si, "si")
            target_docs.append(target_embedd)
            target_docs_weights_sent_len_normalized.append(docs['weight_si'])
            #target_digits.append(extract_digits(doc_si, 'si'))
    
    print (file_name + " : " + str(sent_count))





print(datetime.now()-start)


from margin_base_dist_calculator_for_annoy import margin_base_score

scores =  margin_base_score(source_docs,target_docs,source_docs_weights_sent_len_normalized,target_docs_weights_sent_len_normalized)

sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

matches_s, matches_t = best_matching_2(sorted_scores)


count_s = 0.0
for key in matches_s.keys():
    if (matches_s[key][0][0] == key):
        count_s += 1

count_t = 0.0
for key in matches_t.keys():
    if (matches_t[key][0][0] == key):
        count_t += 1

print(count_s, " ", count_t)

print(datetime.now()-start)
