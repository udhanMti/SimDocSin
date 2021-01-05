import numpy as np
import json
from doc_to_sentence import doc_to_sentence
from datetime import datetime
from annoy import AnnoyIndex
from sentence_to_word import sentence_to_word

from datetime import datetime
import pandas as pd

start = datetime.now()

f = 1024
u = AnnoyIndex(f, 'euclidean')
u.load('test.ann')

file1 = open('/content/drive/Shareddrives/FYP/1M_db_new/dms_new_25.json', encoding='utf8')
data1 = json.load(file1)
source_doc_count = 200
map_file = open('sent_to_doc_map.json', encoding='utf8')
sent_to_doc_maps = json.load(map_file)

map_file = open('sent_count_map.json', encoding='utf8')
sent_count_maps = json.load(map_file)

source_lang = 'en'
target_lang = 'si'


start = datetime.now()


def get_similarity_matrix_ann(embeds1):
    matrix = []
    dict_s = {}
    dict_t = {}
    k = 5

    for i, sent_s in enumerate(embeds1):
        lst = u.get_nns_by_vector(sent_s, 10, 100000)

        x = []
        for j in lst:
            sent_t = u.get_item_vector(j)
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s, sent_t.T) / (
                np.sqrt(np.dot(sent_s, sent_s.T)) * np.sqrt(np.dot(sent_t, sent_t.T)))

            if (cos_similarity > 0.525):
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


'''
def get_similarity_matrix_ann_num(embeds1, embeds2, sentences_in_source):
    matrix = []
    dict_s = {}

    for i, sent_s in enumerate(embeds1):
        lst = u.get_nns_by_vector(sent_s, 10, 100000)

        x = []
        for j in lst:

            sent_t = embeds2[j]
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s, sent_t.T) / (
                        np.sqrt(np.dot(sent_s, sent_s.T)) * np.sqrt(np.dot(sent_t, sent_t.T)))
            if (cos_similarity > 0.70):
                x.append([j, cos_similarity])

        dict_s[i] = x

    _dict = {}

    for i in dict_s:
        dict_s[i] = sorted(dict_s[i], key=lambda x: x[1], reverse=True)

    for i in dict_s:
        if len(dict_s[i]) > 0:
            a = dict_s[i][0][1]
            b = [dict_s[i][j][0] for j in range(1, min(4, len(dict_s[i])))]
            m = len(b)

            if m == 0 or a - (sum(b) / m) >= 0.02:
                matrix.append((i, dict_s[i][0][0]))
                _dict[i] = [dict_s[i][j][0] for j in range(1, len(dict_s[i]))]
            else:

                source_dig = extract_digits(sentences_in_source[i], 'en')

                for r in range(m):
                    x, y = dict_s[i][r]
                    target_dig = extract_digits(sentences_in_combined_target_doc[x], 'si')
                    digit_similarity = get_digit_similarity(source_dig, target_dig)
                    if digit_similarity >= 0.8:
                        matrix.append((i, dict_s[i][r][0]))
                        _dict[i] = [dict_s[i][j][0] for j in range(0, len(dict_s[i])) if j != r]
                        print(r)
                        break
                else:
                    matrix.append((i, dict_s[i][0][0]))
                    _dict[i] = [dict_s[i][j][0] for j in range(1, len(dict_s[i]))]

    # print(_dict)
    return [matrix, _dict]


def get_similarity_matrix_ann_num_ne(embeds1, embeds2, sentences_in_source):
    matrix = []
    dict_s = {}

    for i, sent_s in enumerate(embeds1):
        lst = u.get_nns_by_vector(sent_s, 10, 100000)

        x = []
        for j in lst:

            sent_t = embeds2[j]
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s, sent_t.T) / (
                        np.sqrt(np.dot(sent_s, sent_s.T)) * np.sqrt(np.dot(sent_t, sent_t.T)))
            if (cos_similarity > 0.60):
                x.append([j, cos_similarity])

        dict_s[i] = x

    _dict = {}

    for i in dict_s:
        dict_s[i] = sorted(dict_s[i], key=lambda x: x[1], reverse=True)

    for i in dict_s:
        if len(dict_s[i]) > 0:
            a = dict_s[i][0][1]
            b = [dict_s[i][j][0] for j in range(1, min(4, len(dict_s[i])))]
            m = len(b)

            if m == 0 or a - (sum(b) / m) >= 0.02:
                matrix.append((i, dict_s[i][0][0]))
                _dict[i] = [dict_s[i][j][0] for j in range(1, len(dict_s[i]))]
            else:

                source_dig = extract_digits(sentences_in_source[i], 'en')
                source_name = extract_names(sentences_in_source[i], 'en')
                source_designation = extract_designations(sentences_in_source[i], 'en')
                for r in range(m):
                    x, y = dict_s[i][r]
                    target_dig = extract_digits(sentences_in_combined_target_doc[x], 'si')
                    digit_similarity = get_digit_similarity(source_dig, target_dig)
                    ne_similarity = get_ne_similarity(source_name, source_designation,
                                                      sentences_in_combined_target_doc[x])
                    if digit_similarity + ne_similarity >= 2.0:
                        print("hello j")
                        matrix.append((i, dict_s[i][r][0]))
                        _dict[i] = [dict_s[i][j][0] for j in range(0, len(dict_s[i])) if j != r]
                        break
                else:
                    matrix.append((i, dict_s[i][0][0]))
                    _dict[i] = [dict_s[i][j][0] for j in range(1, len(dict_s[i]))]

    # print(_dict)
    return [matrix, _dict]

'''


def get_similarity_matrix(embeds1, embeds2):  # , sentences_s, sentences_t):
    matrix = []
    dict_s = {}
    dict_t = {}

    k = 5
    for i, sent_s in enumerate(embeds1):
        # dict_s[i] = [-1,[]] # #
        x = []
        for j, sent_t in enumerate(embeds2):
            # x = dict_s[i][1]
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s, sent_t.T) / (
                np.sqrt(np.dot(sent_s, sent_s.T)) * np.sqrt(np.dot(sent_t, sent_t.T)))
            if (cos_similarity > 0.65):
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


def get_similarity_matrix_margin(embeds1, embeds2):  # , sentences_s, sentences_t):
    matrix = []
    dict_s = {}
    dict_t = {}
    scores = {}
    dict_st = {}
    k = 3
    _dict = {}
    for i, sent_s in enumerate(embeds1):
        # dict_s[i] = [-1,[]] # #
        # x = []
        for j, sent_t in enumerate(embeds2):
            # x = dict_s[i][1]
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s, sent_t.T) / (
                np.sqrt(np.dot(sent_s, sent_s.T)) * np.sqrt(np.dot(sent_t, sent_t.T)))
            scores[(i, j)] = cos_similarity
            ### nearest neighbours of target sentences
            if (j in dict_t):
                y = dict_t[j]
                if (len(y) < k):
                    y.append(cos_similarity)
                else:
                    if (min(y) < cos_similarity):
                        y.remove(min(y))
                        y.append(cos_similarity)
                dict_t[j] = y
            else:
                dict_t[j] = [cos_similarity]

            ### nearest neighbours of source sentences
            if (i in dict_s):
                y = dict_s[i]
                # print(y)
                if (len(y) < k):
                    y.append(cos_similarity)
                else:
                    if (min(y) < cos_similarity):
                        y.remove(min(y))
                        y.append(cos_similarity)
                dict_s[i] = y
            else:
                dict_s[i] = [cos_similarity]

    # for pair in scores:
    #     score = get_score(dict_s, dict_t, pair, scores, k)
    #     scores[pair] = score

    for i in range(len(embeds1)):
        x = []
        for j in range(len(embeds2)):
            a = scores[(i, j)]
            b = sum(dict_s[i]) / (2 * min(k, len(dict_s[i]))) + sum(dict_t[j]) / (2 * min(k, len(dict_t[j])))
            margin = a / b
            if margin > 0.98:
                x.append([j, margin])
        dict_st[i] = x

    _dict = {}

    # print(dict_st)
    for i in dict_st:
        neighbours = []
        x = None
        y = -1
        for j in dict_st[i]:
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


def get_similarity_matrix_margin_ann(embeds1):
    matrix = []
    dict_s = {}
    dict_t = {}
    scores = {}
    dict_st = {}
    k = 3
    _dict = {}
    for i, sent_s in enumerate(embeds1):
        lst = u.get_nns_by_vector(sent_s, 10, 100000)
        for j in lst:
            sent_t = u.get_item_vector(j)
            sent_s = np.array(sent_s)
            sent_t = np.array(sent_t)
            cos_similarity = np.dot(sent_s, sent_t.T) / (
                np.sqrt(np.dot(sent_s, sent_s.T)) * np.sqrt(np.dot(sent_t, sent_t.T)))
            scores[(i, j)] = cos_similarity

            ### nearest neighbours of target sentences
            if (j in dict_t):
                y = dict_t[j]
                if (len(y) < k):
                    y.append(cos_similarity)
                else:
                    if (min(y) < cos_similarity):
                        y.remove(min(y))
                        y.append(cos_similarity)
                dict_t[j] = y
            else:
                dict_t[j] = [cos_similarity]

            ### nearest neighbours of source sentences
            if (i in dict_s):
                y = dict_s[i]
                # print(y)
                if (len(y) < k):
                    y.append(cos_similarity)
                else:
                    if (min(y) < cos_similarity):
                        y.remove(min(y))
                        y.append(cos_similarity)
                dict_s[i] = y
            else:
                dict_s[i] = [cos_similarity]

                # for pair in scores:
                #     score = get_score(dict_s, dict_t, pair, scores, k)
                #     scores[pair] = score

    for i in range(len(embeds1)):
        x = []
        for kk, j in scores.keys():
            if (i, j) not in scores:
                continue
            a = scores[(i, j)]
            b = sum(dict_s[i]) / (2 * min(k, len(dict_s[i]))) + sum(dict_t[j]) / (2 * min(k, len(dict_t[j])))
            margin = a / b
            if margin > 0.98:
                x.append([j, margin])
        dict_st[i] = x

    _dict = {}

    # print(dict_st)
    for i in dict_st:
        neighbours = []
        x = None
        y = -1
        for j in dict_st[i]:
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
    # print(d)
    return d


def diagonal_extract_down(matrix, dict_s):
    start_position = matrix[0]
    x = start_position[0]
    y = start_position[1]
    d = []

    while ((x, y) in matrix) or ((x + 1, y) in matrix) or ((x, y + 1) in matrix) or (
        (x in dict_s) and (y in dict_s[x])) or ((x + 1 in dict_s) and (y in dict_s[x + 1])) or (
        (x in dict_s) and ((y + 1) in dict_s[x])):
        if (x, y) in matrix:
            d.append((x, y))
            x -= 1
            y -= 1
        elif (x + 1, y) in matrix:
            d.append((x + 1, y))
            y -= 1
        elif (x, y + 1) in matrix:
            d.append((x, y + 1))
            x -= 1
        elif (x in dict_s) and (y in dict_s[x]):
            d.append((x, y))
            x -= 1
            y -= 1
        elif (x + 1 in dict_s) and (y in dict_s[x + 1]):
            d.append((x + 1, y))
            y -= 1
        elif (x in dict_s) and (y + 1) in dict_s[x]:
            d.append((x, y + 1))
            x -= 1
    # print(d)
    return d


def sequence_matching_down(matrix, dict_s, sentences_s):
    diagonals = []
    new_matrix = []
    while len(matrix) > 0:
        d = diagonal_extract_down(matrix, dict_s)
        new_matrix.extend(d)
        # print(d)
        s = []
        t = []
        for i in d:
            if (i[1] not in t):
                t.append(i[1])

            if (i[0] not in s):
                s.append(i[0])

        # if(len(s)>1 or len(sentences_s)==1):
        #     diagonals.append((s,t))
        if (len(t) > 0):
            diagonals.append((s[::-1], t[::-1]))
        # for each in d:
        #     if(each in matrix):
        #         matrix.remove(each)

        for kk in s:
            for y in matrix:
                if y[0] == kk:
                    matrix.remove(y)

    return diagonals[::-1]


def sequence_matching(matrix, dict_s, sentences_s):
    diagonals = []
    new_matrix = []
    while len(matrix) > 0:
        # print ("Check")
        # print (matrix)
        # print (dict_s)
        d = diagonal_extract(matrix, dict_s)
        new_matrix.extend(d)
        # print(d)
        s = []
        t = []
        for i in d:
            if (i[1] not in t):
                t.append(i[1])

            if (i[0] not in s):
                s.append(i[0])

        # if(len(s)>1 or len(sentences_s)==1):
        #     diagonals.append((s,t))
        if (len(s) > 0):
            diagonals.append((s, t))
        for each in d:
            if (each in matrix):
                matrix.remove(each)

                # for kk in s:
                #     for y in matrix:
                #         if y[0] == kk:
                #                matrix.remove(y)

    # matrix = new_matrix[::-1]
    print(diagonals)
    return diagonals
    # return sequence_matching_down(matrix,dict_s,sentences_s)


def get_sentence(j):
    global sent_count_maps, sent_to_doc_maps
    doc_no = sent_to_doc_maps[str(j)]
    doc = open('./army/' + str(doc_no // 1000) + '/docsi/' + str(doc_no % 1000) + '.txt', encoding='utf-8').read()
    doc = doc_to_sentence(doc, 'si')
    # print(doc_no,j)
    sent_no = j - sent_count_maps[str(doc_no)]
    # print(doc[sent_no])
    return doc[sent_no]


def post_processor(diagonals, gap_threshold, min_length):
    diagonals = sorted(diagonals, key=lambda x: x[0][0])
    # gap_threshold = 40
    # print(diagonals)
    processed_diagonals = []
    for i in range(len(diagonals)):
        diagonal = diagonals[i]
        temp = []
        new_diagonal = diagonal
        last = 0
        for j in range(len(processed_diagonals)):
            last = j
            pro_diagonal = processed_diagonals[j]
            temp.append(pro_diagonal)
            gap_s = (diagonal[0][0] - pro_diagonal[0][-1])
            gap_t = (diagonal[1][0] - pro_diagonal[1][-1])
            if ((gap_s <= gap_threshold) and (gap_s >= 0) and (gap_t <= gap_threshold) and (gap_t >= 0)):
                s = [_s for _s in range(pro_diagonal[0][0], diagonal[0][-1] + 1)]
                t = [_t for _t in range(pro_diagonal[1][0], diagonal[1][-1] + 1)]
                new_diagonal = (s, t)
                temp = temp[:-1]
                break
            elif ((gap_s <= gap_threshold) and (gap_t <= gap_threshold) and (pro_diagonal[1][0] <= diagonal[1][0])):
                s = [_s for _s in range(pro_diagonal[0][0], max(pro_diagonal[0][-1], diagonal[0][-1]) + 1)]
                t = [_t for _t in range(pro_diagonal[1][0], max(pro_diagonal[1][-1], diagonal[1][-1]) + 1)]
                new_diagonal = (s, t)
                temp = temp[:-1]
                break

        temp.append(new_diagonal)
        temp.extend(processed_diagonals[last + 1:])
        processed_diagonals = temp
    # print(processed_diagonals)

    final_diagonals = []
    for diagonal in processed_diagonals:
        if ((len(diagonal[1]) > min_length)):  # or (len(diagonal[1])>1)):
            final_diagonals.append(diagonal)

    return final_diagonals


## ======== code for evaluation =================

# for gap_threshold in [100]:#[0,2,5,10,15,20,40,60,100]:
#  for min_length in [0]:#:[0,1,2,5,10,15]:
gap_threshold = 10
min_length = 1

recall_numerator = 0
precision_numerator = 0
precision_denominator = 0

c = 0

source_df = []
target_df = []
valid_df = []
doc_df = []
doc_no = 0
for doc in data1[:source_doc_count]:  # source_doc_count
    doc_no = doc_no + 1
    doc_s = doc['content_' + source_lang]
    source_embedd = doc['embed_' + source_lang]
    print(len(source_embedd))

    target_doc = doc['content_' + target_lang]
    actual_target_sentences_pre = doc_to_sentence(target_doc, target_lang)
    actual_target_sentences = []
    for sentence in actual_target_sentences_pre:
        if (len(sentence_to_word(sentence, target_lang)) > 2):
            actual_target_sentences.append(sentence)

    sentences_in_source = doc_to_sentence(doc_s, source_lang)
    # print (len(sentences_in_source))
    matrix, dict_s = get_similarity_matrix_ann(source_embedd)
    # print ("match")
    # print(matrix)
    # print(dict_s)

    if len(sentences_in_source) > 1:
        c += 1

    diagonals = sequence_matching(matrix, dict_s, sentences_in_source)
    # print(diagonals)
    diagonals = post_processor(diagonals, gap_threshold + 1, min_length)
    print(diagonals)
    print(c)
    true_predicted_target_sentences = set()
    predicted_target_sentences = set()

    # print(target_doc)
    for index, diagonal in enumerate(diagonals):
        partial_source = []
        partial_target = []

        # print (diagonal)
        for j in diagonal[1]:

            sent = get_sentence(j)
            # print("Sentence")
            # print(sent)
            if (len(sentence_to_word(sent, target_lang)) > 2):
                predicted_target_sentences.add(sent)

                if (sent in actual_target_sentences):
                    sent_input = []
                    for n in diagonal[0]:
                        sent_input.append(sentences_in_source[n])
                    sent_join = " /n".join(sent_input)

                    source_df.append(sent_join)
                    target_df.append(sent)
                    valid_df.append(1)
                    doc_df.append(doc_no)

                    true_predicted_target_sentences.add(sent)
                else:
                    sent_input = []
                    for n in diagonal[0]:
                        sent_input.append(sentences_in_source[n])
                    sent_join = " ".join(sent_input)

                    source_df.append(sent_join)
                    target_df.append(sent)
                    valid_df.append(0)
                    doc_df.append(doc_no)

    recall_numerator += len(true_predicted_target_sentences) / len(actual_target_sentences)
    # print(len(true_predicted_target_sentences)/len(actual_target_sentences))

    if len(predicted_target_sentences) > 0:
        precision_numerator += len(true_predicted_target_sentences) / len(predicted_target_sentences)
        # print(len(true_predicted_target_sentences)/len(predicted_target_sentences))
        precision_denominator += 1

print(c)
recall = recall_numerator / source_doc_count
print("recall: ", recall)

if (precision_denominator > 0):
    precision = precision_numerator / precision_denominator
    print("precision: ", precision)
    print(2 * (recall * precision) / (recall + precision))
else:
    print('precision not defined')

output = {'source': source_df, 'target': target_df, 'doc_no': doc_df, 'validity': valid_df}
# df = pd.DataFrame([source_df, target_df],index=['row 1', 'row 2'],columns=['source', 'target'])
df = pd.DataFrame(output, columns=['source', 'target', 'doc_no', 'validity'])
df.to_excel("/content/drive/MyDrive/FYP-Colab_wsws/data/output_dms.xlsx", encoding='utf-8')

print(datetime.now() - start)