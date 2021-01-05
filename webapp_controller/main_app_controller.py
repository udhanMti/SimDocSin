import json
from laser_control import get_embeddig_list

from weight_schema import *
from doc_matcher import competitive_matching, best_matching, best_matching_2, best_matching_3
from datetime import datetime
from margin_base_distance_calculator import margin_base_score
from extract_digits import extract_digits, get_digit_similarity

def main(sources, threshold_index, source_lang='en'):

        threshold = 0.965 + ((0.99-0.965)/4) * (4 - int(threshold_index))
        target_lang = 'si'
        if(source_lang=='si'):
            target_lang = 'en'

        file  = open('C:/Users/Udhan/Desktop/FYP/MassDoc/MassivelyDocAlignment/embedded_data/embedding_hiru.json',encoding='utf8')
        data = json.load(file)

        source_docs = []
        target_docs = []

        source_digits = []
        target_digits = []

        source_names = []
        source_designations = []

        source_docs_weights_sent_len_normalized=[]
        target_docs_weights_sent_len_normalized=[]

        source_documents=[]
        target_documents=[]

        for source in sources:
            doc_source = source
            source_documents.append(doc_source)
            source_embedd = get_embeddig_list(doc_source, source_lang)
            source_digits.append(extract_digits(doc_source, source_lang))


            source_docs.append(source_embedd)
            source_weight= get_sentence_length_weighting_list(doc_source, source_lang)
            source_docs_weights_sent_len_normalized.append(documentMassNormalization(source_weight))

        start=datetime.now()
        for docs in data[:200]:

            doc_target = docs['content_'+target_lang]

            #en_documents.append(doc_en)
            target_documents.append(doc_target)

            target_digits.append(extract_digits(doc_target, target_lang))

            
            target_embedd = docs ['embed_'+target_lang]

            target_docs.append(target_embedd)

            target_weight = get_sentence_length_weighting_list(doc_target, target_lang)

            target_docs_weights_sent_len_normalized.append(documentMassNormalization(target_weight))



        scores = margin_base_score(source_docs,target_docs,source_docs_weights_sent_len_normalized,target_docs_weights_sent_len_normalized)#{}

        
        sorted_scores= {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

        matches_s = best_matching_3(sorted_scores, threshold)

        results = []

        targets_count = 0

        for key in matches_s.keys():
            result = {}
            result['source'] = source_documents[key]
            if (len(matches_s[key])==0):

                result['target'] = [(-1, 'No match')]
            else:
    
                matching_target_documents = []


                s_digits = source_digits[key]
                if(len(s_digits)>0):
                    matching_targets = {}
                    x=0
                    y = len(matches_s[key])
                    for candidate in matches_s[key]:
                        ne_similarity = 0#get_ne_similarity(s_names, s_designations, target_documents[candidate[0]])
                        digit_similarity = get_digit_similarity(s_digits,target_digits[candidate[0]])
                        matching_targets[candidate[0]] = digit_similarity + ne_similarity + (y-x)*0.00001
                        x+=1

                    temp = {k: v for k, v in sorted(matching_targets.items(), key=lambda item: item[1], reverse=True)}
                   
                    for k in temp.keys():
                        matching_target_documents.append((targets_count,target_documents[k]))
                        targets_count+=1
                   
                else:
                    for k in matches_s[key]:
                        matching_target_documents.append((targets_count,target_documents[k[0]]))
                        targets_count+=1

                result['target'] = matching_target_documents
                
            results.append(result)
        
        return results
