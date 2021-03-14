from embedder.laser_control import get_embeddig_list
import numpy as np
from weight_schema import *
from doc_matcher import best_matching_3
from margin_base_distance_calculator import margin_base_score
from extract_digits import extract_digits, get_digit_similarity


def main(sources, threshold_index, source_lang='en'):
    threshold = 0.965 + ((0.99 - 0.965) / 4) * (4 - int(threshold_index))
    target_lang = 'si'
    if (source_lang == 'si'):
        target_lang = 'en'

    source_docs = []
    source_digits = []
    source_docs_weights_sent_len_normalized = []
    source_documents = []

    for source in sources:
        doc_source = source
        source_documents.append(doc_source)
        source_embedd = get_embeddig_list(doc_source, source_lang)
        source_digits.append(extract_digits(doc_source, source_lang))

        # source_names.append(extract_names(doc_source,source_lang))
        # source_designations.append(extract_designations(doc_source,source_lang))

        source_docs.append(source_embedd)
        source_weight = get_sentence_length_weighting_list(doc_source, source_lang)
        source_docs_weights_sent_len_normalized.append(documentMassNormalization(source_weight))

    scores = margin_base_score(source_docs, source_docs_weights_sent_len_normalized, target_lang)  # {}

    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

    matches_s = best_matching_3(sorted_scores, threshold)

    results = []
    targets_count = 0
    target_digits = np.load('../db/' + target_lang + '_digits.npy', allow_pickle=True)

    for key in matches_s.keys():
        result = {}
        result['source'] = source_documents[key]
        if (len(matches_s[key]) == 0):
            result['target'] = [(-1, 'No match')]
        else:

            matching_target_documents = []

            # s_names = source_names[key]
            # s_designations = source_designations[key]

            s_digits = source_digits[key]
            if (len(s_digits) > 0):
                matching_targets = {}
                x = 0
                y = len(matches_s[key])
                for candidate in matches_s[key]:
                    # kk=candidate[0]
                    # target_documents = open('../db/' + str(kk // 1000) + '/doc' + target_lang + '/' + str(
                    #     kk % 1000) + '.txt', encoding='utf-8').read()
                    ne_similarity = 0  # get_ne_similarity(s_names, s_designations, target_documents[candidate[0]])
                    digit_similarity = get_digit_similarity(s_digits, target_digits[candidate[0]])
                    matching_targets[candidate[0]] = digit_similarity + ne_similarity + (y - x) * 0.00001
                    # if digit+en = 0 then order
                    x += 1
                    # if(digit_similarity>0.3):
                    #    result['target'] =  target_docs[candidate[0]]
                    #    break
                # else:
                #    result['target'] = target_docs[matches_s[key][0][0]]
                temp = {k: v for k, v in sorted(matching_targets.items(), key=lambda item: item[1], reverse=True)}

                for k in temp.keys():
                    target_documents = open('../db/' + str(k // 1000) + '/doc' + target_lang + '/' + str(
                        k % 1000) + '.txt', encoding='utf-8').read()
                    matching_target_documents.append((targets_count, target_documents))
                    targets_count += 1

            else:
                for k in matches_s[key]:
                    target_documents = open('../db/' + str(k[0] // 1000) + '/doc' + target_lang + '/' + str(
                        k[0] % 1000) + '.txt',encoding='utf-8').read()
                    matching_target_documents.append((targets_count, target_documents))
                    targets_count += 1

            result['target'] = matching_target_documents

        results.append(result)
    resultsNotAvailable = False
    if (len(results) == 0):
        resultsNotAvailable = True

    return results, resultsNotAvailable
    #return results
