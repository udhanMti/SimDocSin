
import numpy as np
import time

def eucledian_distance(sent_s,sent_t):

    tmp = sent_s - sent_t

    return np.sqrt(np.dot(tmp.T, tmp))

def get_cosine_similarity(sent_s,sent_t):

    return np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

def greedy_mover_distance(doc_s,doc_t,weight_s,weight_t):
    
    doc_s = np.array(doc_s)
    doc_t = np.array(doc_t)

    dict_pairs = {}
    dict_s = {}
    dict_t = {}

    for i in range(len(doc_s)):
        for j in range(len(doc_t)):

            distance = eucledian_distance(doc_s[i],doc_t[j])
            dict_pairs[(i,j)] = distance


    pairs = {h: v for h, v in sorted(dict_pairs.items(), key=lambda item: item[1])}

    distance = 0.0
    for pair in pairs:
        sent_s = pair[0]
        sent_t = pair[1]
        #score = get_score(dict_s, dict_t, pair, pairs, k)
        # print(pair)
        flow = min(weight_s[sent_s],weight_t[sent_t])
        weight_s[sent_s] = weight_s[sent_s]-flow
        weight_t[sent_t] = weight_t[sent_t]-flow
        distance = distance + flow * pairs[pair] #score
    # print(time.time() - start)
    return distance

def get_score(dict_s, dict_t, pair, pairs, k):
    sent_s = pair[0]
    sent_t = pair[1]
    a = pairs[pair]
    b = sum(dict_s[sent_s])/(2*min(k, len(dict_s[sent_s]))) + sum(dict_t[sent_t])/(2*min(k, len(dict_t[sent_t])))
    return a/b


def greedy_mover_distance_cssl2(doc_s, doc_t, weight_s, weight_t):
    k = 3
    dict = {}
    dict_s = {}
    dict_t = {}
    for i in range(len(doc_s)):
        for j in range(len(doc_t)):
            distance =  get_cosine_similarity(doc_s[i], doc_t[j])#eucledian_distance(doc_s[i],doc_t[j])
            dict[(i, j)] = distance
            if (i in dict_s):
                kneighbours = dict_s[i]
                if (len(kneighbours) < k):
                    kneighbours.append(distance)
                else:
                    min_distance = min(kneighbours)
                    if (distance > min_distance):
                        kneighbours.append(distance)
                        kneighbours.remove(min_distance)
                dict_s[i] = kneighbours
            else:
                dict_s[i] = [distance]

            if (j in dict_t):
                kneighbours = dict_t[j]
                if (len(kneighbours) < k):
                    kneighbours.append(distance)
                else:
                    min_distance = min(kneighbours)
                    if (distance > min_distance):
                        kneighbours.append(distance)
                        kneighbours.remove(min_distance)
                dict_t[j] = kneighbours
            else:
                dict_t[j] = [distance]
    new_dict={}
    for pair in dict:
        sent_s = pair[0]
        sent_t = pair[1]
        score = 2-cslr_score(dict_s, dict_t, pair, dict, k)
        new_dict[(sent_s,sent_t)]=score

    pairs = {k: v for k, v in sorted(new_dict.items(), key=lambda item: item[1])}


    distance = 0.0
    for pair in pairs:
        sent_s = pair[0]
        sent_t = pair[1]

        flow = min(weight_s[sent_s], weight_t[sent_t])
        weight_s[sent_s] = weight_s[sent_s] - flow
        weight_t[sent_t] = weight_t[sent_t] - flow
        distance = distance + flow * pairs[pair]  # pairs[pair]
    return distance

def cslr_score(dict_s, dict_t, pair, pairs, k):
    sent_s = pair[0]
    sent_t = pair[1]
    a = pairs[pair]
    return 2*a - (sum(dict_s[sent_s]) / min(k,len(dict_s[sent_s]))) - (sum(dict_t[sent_t]) / min(k,len(dict_t[sent_t])))
