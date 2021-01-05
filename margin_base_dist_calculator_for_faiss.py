from greedy_mover_distance import greedy_mover_distance
import faiss
import numpy as np
import collections
import json

f=1024
#u = AnnoyIndex(f, 'euclidean')
#u.load('test.ann')
res = faiss.StandardGpuResources()
cpu_index = faiss.read_index('ann/faiss_index.index')
index = cpu_index#faiss.index_cpu_to_gpu(res, 0, cpu_index)

map_file =  open('ann/sent_to_doc_map.json',encoding='utf8')
maps = json.load(map_file)

def get_score(dict_s, dict_t, pair, pairs, k):
    sent_s = pair[0]
    sent_t = pair[1]
    a = pairs[pair]
    b = sum(dict_s[sent_s])/(2*min(k, len(dict_s[sent_s]))) + sum(dict_t[sent_t])/(2*min(k, len(dict_t[sent_t])))
    return a/b

def margin_base_score(source_docs,target_docs,source_weight,target_weight):
    k = 3
    scores = {}
    dict_s = {}
    dict_t = {}
    for i in range(len(source_docs)):
        print(i)
        all_docs = []
        for embed in source_docs[i]:
          #lst = u.get_nns_by_vector(embed, 10,search_k=100000)
          xq = []
          xq.append(embed)
          np_xq = np.array(xq).astype('float32')
          D, I = index.search(np_xq, 10) 
          for sent in I[0]:
            all_docs.append(maps[str(sent)])

        counts = collections.Counter(all_docs)
        new_list = sorted(all_docs, key=counts.get, reverse=True)

        all_docs = []
        for y in new_list:
          if len(all_docs) == 20:
              break
          if y not in all_docs:
              all_docs.append(y)

        for j in all_docs:
            distance = greedy_mover_distance(source_docs[i], target_docs[j],source_weight[i].copy(),
                                             target_weight[j].copy())
            # distance = s[(i,j)]
            scores[(i, j)] = distance

            if (i in dict_s):
                kneighbours = dict_s[i]
                if (len(kneighbours) < k):
                    kneighbours.append(distance)
                else:
                    max_distance = max(kneighbours)
                    if (distance < max_distance):
                        kneighbours.append(distance)
                        kneighbours.remove(max_distance)
                dict_s[i] = kneighbours
            else:
                dict_s[i] = [distance]

            if (j in dict_t):
                kneighbours = dict_t[j]
                if (len(kneighbours) < k):
                    kneighbours.append(distance)
                else:
                    max_distance = max(kneighbours)
                    if (distance < max_distance):
                        kneighbours.append(distance)
                        kneighbours.remove(max_distance)
                dict_t[j] = kneighbours
            else:
                dict_t[j] = [distance]

    for pair in scores:
        score = get_score(dict_s, dict_t, pair, scores, k)
        scores[pair] = score
    return scores

