import json
import numpy as np
import faiss
file  = open('embedded_data/embedding_army_0_1000.json',encoding='utf8')
embed_data = json.load(file)

sent_to_doc_map = {}

f = 1024

vectors = []

sent_count ={}
sent_count[0] = 0

i=0
for j in range(1000):
    si_doc =embed_data[j]['content_si']
    si_doc_embed = embed_data[j]['embed_si']

    sent_count[j+1]= sent_count[j]+len(si_doc_embed)

    # sentences = doc_to_sentence(si_doc)
    for k in range(len(si_doc_embed)):
        sent_to_doc_map[i]=j
        #t.add_item(i, si_doc_embed[k])
        vectors.append(si_doc_embed[k])
        i+=1
        
np_vectors = np.array(vectors).astype('float32')

res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(f)   # build the index
#gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

index.add(np_vectors)                  # add vectors to the index

#cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(index, "ann/faiss_index.index")

'''
index = faiss.IndexFlatL2(f)   # build the index
index.add(np_vectors)                  # add vectors to the index

#cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(index, "ann/faiss_index.index")
'''

with open("./ann/sent_to_doc_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_to_doc_map, outfile, ensure_ascii=False)

with open("./ann/sent_count_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_count, outfile, ensure_ascii=False)
