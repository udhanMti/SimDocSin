import json
from embedder.laser_control import get_embeddig_list
from greedy_mover_distance import greedy_mover_distance
from weight_schema import *
from splitter.doc_to_sentence import *
from annoy import AnnoyIndex

import numpy as np

#  from scipy.stats import norm
# import matplotlib.pyplot as plt
f = 1024
u = AnnoyIndex(f, 'euclidean')
u.load('../index/test.ann')

map_file = open('../index/sent_to_doc_map.json', encoding='utf8')
sent_to_doc_maps = json.load(map_file)

map_file = open('../index/sent_count_map.json', encoding='utf8')
sent_count_maps = json.load(map_file)


def get_similarity_matrix(embeds1, threshold):
    global u, map_file, sent_to_doc_maps, sent_count_maps
    matrix = []
    dict_s = {}
    dict_t = {}
    # pairs = {}
    k = 5
    for i, sent_s in enumerate(embeds1):
        # dict_s[i] = [-1,[]]
        lst = u.get_nns_by_vector(sent_s, 10, 100000)
        x = []
        for j in lst:

            # all_docs.append(maps[str(sent)])
            # for j, sent_t in enumerate(embeds2):
            sent_t = u.get_item_vector(j)  # embeds2[j]

            # x = dict_s[i][1]
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s, sent_t.T) / (
                    np.sqrt(np.dot(sent_s, sent_s.T)) * np.sqrt(np.dot(sent_t, sent_t.T)))
            if (cos_similarity > threshold):
                x.append([j, cos_similarity])

        dict_s[i] = x

    _dict = {}

    for i in dict_s:
        neighbours = []
        x = None
        y = -1
        for j in dict_s[i]:
            neighbours.append(j[0])
            if (j[1] > y):
                y = j[1]
                x = j[0]
        if (x != None):
            matrix.append((i, x))
        if (len(neighbours) > 0):
            neighbours.remove(x)
        _dict[i] = neighbours

    return [matrix, _dict]


def diagonal_extract(matrix, dict_s):
    global u, map_file, sent_to_doc_maps, sent_count_maps
    start_position = matrix[0]
    x = start_position[0]
    y = start_position[1]
    d = []
    while ((x, y) in matrix) or ((x - 1, y) in matrix) or ((x, y - 1) in matrix) or (
            (x in dict_s) and (y in dict_s[x])) or ((x - 1 in dict_s) and (y in dict_s[x - 1])) or (
            (x in dict_s) and ((y - 1) in dict_s[x])):

        if (x, y) in matrix:
            d.append((x, y))
            x += 1
            y += 1
        elif (x - 1, y) in matrix:
            d.append((x - 1, y))
            y += 1
        elif (x, y - 1) in matrix:
            d.append((x, y - 1))
            x += 1

        elif (x in dict_s) and (y in dict_s[x]):
            d.append((x, y))
            x += 1
            y += 1
        elif (x - 1 in dict_s) and (y in dict_s[x - 1]):
            d.append((x - 1, y))
            y += 1
        elif (x in dict_s) and (y - 1) in dict_s[x]:
            d.append((x, y - 1))
            x += 1

    return d


def sequence_matching(matrix, dict_s, threshold_length):
    global u, map_file, sent_to_doc_maps, sent_count_maps
    diagonals = []
    while len(matrix) > 0:
        d = diagonal_extract(matrix, dict_s)

        s = []
        t = []

        for i in d:
            if (i[1] not in t):
                t.append(i[1])

            if (i[0] not in s):
                s.append(i[0])

        if (len(t) > threshold_length):
            diagonals.append((s, t))
        for each in d:
            if (each in matrix):
                matrix.remove(each)
    return diagonals


def main_partial(source, threshold_index, threshold_length, source_lang='en'):
    global u, map_file, sent_to_doc_maps, sent_count_maps
    if (source_lang == 'si'):
        u = AnnoyIndex(f, 'euclidean')
        u.load('../index/test_en.ann')

        map_file = open('../index/sent_to_doc_map_en.json', encoding='utf8')
        sent_to_doc_maps = json.load(map_file)

        map_file = open('../index/sent_count_map_en.json', encoding='utf8')
        sent_count_maps = json.load(map_file)
    else:
        u = AnnoyIndex(f, 'euclidean')
        u.load('../index/test.ann')

        map_file = open('../index/sent_to_doc_map.json', encoding='utf8')
        sent_to_doc_maps = json.load(map_file)

        map_file = open('../index/sent_count_map.json', encoding='utf8')
        sent_count_maps = json.load(map_file)
    source = source[0]

    results = []

    threshold = 0.62 + ((0.68 - 0.62) / 4) * (4 - int(threshold_index))
    target_lang = 'si'
    if (source_lang == 'si'):
        target_lang = 'en'

    min_length = 0
    if (threshold_length == '1+'):
        min_length = 1
    elif (threshold_length == '2+'):
        min_length = 2
    elif (threshold_length == '5+'):
        min_length = 5
    elif (threshold_length == '10+'):
        min_length = 10

    # file1 = open('../MassDoc/embed_data/embedding.json',
    #            encoding='utf8')
    # data1 = json.load(file1)
    # target_docs = []
    # for doc in data1:
    #    target_docs.append(doc['content_' + target_lang])

    # doc_s = doc['content_' + source_lang]
    source_embedd = get_embeddig_list(source, source_lang)
    matrix, dict_s = get_similarity_matrix(source_embedd, threshold)

    sentences_s = doc_to_sentence(source, source_lang)
    diagonals = sequence_matching(matrix, dict_s, min_length)

    matching_partials = {}
    matching_partials_scores = {}

    target_doc_id = 0
    for index, i in enumerate(diagonals):
        partial_source = []
        partial_target = []

        partial_embed_source = []
        partial_embed_target = []

        for j in i[0]:
            partial_embed_source.append(source_embedd[j])
            partial_source.append(sentences_s[j])

        docs_targets_belong_to = []
        docs_targets_ids = []

        matching_doc_index = sent_to_doc_maps[str(i[1][0])]
        # matching_doc = target_docs[matching_doc_index]
        matching_doc = open('../db/' + str(matching_doc_index // 1000) + '/doc' + target_lang + '/' + str(
            matching_doc_index % 1000) + '.txt', encoding='utf-8').read()

        docs_targets_belong_to.append(matching_doc)
        docs_targets_ids.append(target_doc_id)
        target_doc_id += 1

        sentences_t = doc_to_sentence(matching_doc, target_lang)
        index_of_first_sentence = int(sent_count_maps[str(matching_doc_index)])
        '''
        while (len(sentences_t) < len(i[1])):
            matching_doc_index = sent_to_doc_maps[str(i[1][len(sentences_t)])]
            matching_doc = open('../MassDoc/db/' + str(matching_doc_index // 1000) + '/doc'+target_lang+'/' + str(matching_doc_index % 1000) + '.txt').read()
            sentences_t.extend(doc_to_sentence(matching_doc, target_lang))
        '''

        relative_index = i[1][0] - index_of_first_sentence
        for j in i[1]:
            partial_embed_target.append(u.get_item_vector(j))

            if (relative_index > len(sentences_t) - 1):
                matching_doc_index += 1
                matching_doc = open(
                    '../db/' + str(matching_doc_index // 1000) + '/doc' + target_lang + '/' + str(
                        matching_doc_index % 1000) + '.txt', encoding='utf-8').read()

                docs_targets_belong_to.append(matching_doc)
                docs_targets_ids.append(target_doc_id)
                target_doc_id += 1

                sentences_t = doc_to_sentence(matching_doc, target_lang)
                relative_index = 0

            partial_target.append(sentences_t[relative_index])
            relative_index += 1

        source_weight = documentMassNormalization(get_sentence_length_weighting_list_2(partial_source, source_lang))
        target_weight = documentMassNormalization(get_sentence_length_weighting_list_2(partial_target, target_lang))

        distance = greedy_mover_distance(partial_embed_source, partial_embed_target, source_weight, target_weight)

        score = (1 / distance) * len(i[0])

        matching_partials[index] = [partial_source, partial_target, docs_targets_belong_to, docs_targets_ids]
        matching_partials_scores[index] = score

    sorted_scores = {k: v for k, v in sorted(matching_partials_scores.items(), key=lambda item: item[1], reverse=True)}

    for key in sorted_scores.keys():
        result = {}
        result['matching_partal_source'] = matching_partials[key][0]
        result['matching_partal_target'] = matching_partials[key][1]
        result['documents_targets_belong_to'] = matching_partials[key][2]
        result['documents_targets_ids'] = matching_partials[key][3]
        result['score'] = matching_partials_scores[key]

        results.append(result)

    resultsNotAvailable = False
    if (len(results) == 0):
        resultsNotAvailable = True

    return results, resultsNotAvailable