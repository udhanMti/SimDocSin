from greedy_mover_distance import greedy_mover_distance

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
    #print(s)
    for i in range(len(source_docs)):
        print(i)
        for j in range(len(target_docs)):
            distance = greedy_mover_distance(source_docs[i], target_docs[j],source_weight[i].copy(),
                                              target_weight[j].copy())
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