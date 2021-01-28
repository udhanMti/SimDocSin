import json
from annoy import AnnoyIndex
import numpy as np

# file  = open('embedding.json', encoding='utf8')
# embed_data = json.load(file)

sent_to_doc_map = {}

f = 1024
t = AnnoyIndex(f, 'euclidean')

i=0

sent_count ={}
sent_count[0] = 0

for j in range(198):
    # si_doc =embed_data[j]['content_si']
    si_doc_embed = np.load('../db/'+str(j//1000)+'/si/' + str(j%1000) + '.npy')

    sent_count[j+1]= sent_count[j]+len(si_doc_embed)

    # sentences = doc_to_sentence(si_doc)
    for k in range(len(si_doc_embed)):
        sent_to_doc_map[i]=j
        t.add_item(i, si_doc_embed[k])
        i=i+1

t.build(25)
t.save('../index/test.ann')

with open("../index/sent_to_doc_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_to_doc_map, outfile, ensure_ascii=False)

with open("../index/sent_count_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_count, outfile, ensure_ascii=False)