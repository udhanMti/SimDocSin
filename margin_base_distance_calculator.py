from greedy_mover_distance import greedy_mover_distance
from annoy import AnnoyIndex
import collections
import json
import numpy as np

def get_score(dict_s, dict_t, pair, pairs, k):
    sent_s = pair[0]
    sent_t = pair[1]
    a = pairs[pair]
    b = sum(dict_s[sent_s])/(2*min(k, len(dict_s[sent_s]))) + sum(dict_t[sent_t])/(2*min(k, len(dict_t[sent_t])))
    return a/b

def margin_base_score(source_document, source_weights, target_lang):
    if (target_lang == 'si'):
        f = 1024
        u = AnnoyIndex(f, 'euclidean')
        u.load('../index/test_si.ann')

        map_file = open('../index/sent_to_doc_map_si.json', encoding='utf8')
        maps = json.load(map_file)

        cmap_file = open('../index/sent_count_map_en.json', encoding='utf8')
        cmaps = json.load(cmap_file)
    else:
        f = 1024
        u = AnnoyIndex(f, 'euclidean')
        u.load('../index/test_en.ann')
        map_file = open('../index/sent_to_doc_map_en.json', encoding='utf8')
        maps = json.load(map_file)

        cmap_file = open('../index/sent_count_map_en.json', encoding='utf8')
        cmaps = json.load(cmap_file)

    k = 3
    scores = {}
    dict_s = {}
    dict_t = {}

    for i in range(len(source_document)):
        all_docs = []
        source_docs = source_document[i]
        source_weight = source_weights[i]
        for embed in source_docs:
            lst = u.get_nns_by_vector(embed, 10, search_k=100000)
            for sent in lst:
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
            target_weight = np.load(
                '../db/' + str(j // 1000) + '/w' + target_lang + '/' + str(j % 1000) + '.npy')

            # add read embedding from ann here
            cc = int(cmaps[str(j)])
            target_docs = []
            for kk in range(len(target_weight)):
                target_docs.append(u.get_item_vector(cc + kk))

            # below line for read from .npy embd files
            # target_docs = np.load('../db/' + str(j // 1000) + '/'+target_lang+'/' + str(j % 1000) + '.npy')

            distance = greedy_mover_distance(source_document[i].copy(), target_docs.copy(), source_weights[i].copy(),
                                             target_weight.copy())
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