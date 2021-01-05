
import numpy as np
import json
from laser_control import get_embeddig_list
from doc_to_sentence import doc_to_sentence
from datetime import datetime
from annoy import AnnoyIndex
from greedy_mover_distance import greedy_mover_distance
from weight_schema import *

f=1024
u = AnnoyIndex(f, 'euclidean')
u.load('./ann/test.ann')

def get_matching_sequence(i,sentences_s, threshold, sentences_t):
    

    sequence = []
    j_s = 0
    j_t = 0
    while (j_s<len(sentences_s)) and (j_t+i<len(sentences_t)):
        sent_s = np.array(sentences_s[j_s])
        sent_t = np.array(sentences_t[j_t+i])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[max(0,j_s-1)])
        sent_t = np.array(sentences_t[j_t+i])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[j_s])
        sent_t = np.array(sentences_t[max(0,j_t-1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(cos_similarity_1>threshold):
            sequence.append(i+j_t)
            j_s+=1
            j_t+=1
        elif(cos_similarity_2>threshold):
            sequence.append(i+j_t)
            j_t+=1
        elif(cos_similarity_3>threshold):
            j_s+=1
        else:
            return [sequence,j_s]
        
    return [sequence,j_s]

def get_matching_sequence_2(x,y,sentences_s, threshold, sentences_t):
    up_x, up_y = x, y
    down_x, down_y = x, y
    sequence = []
    j = 0
    k = 0
    continuous_length = 0
    relax_rate = 0

    while (up_x>=0) and (up_y>=0):
        
        sent_s = np.array(sentences_s[up_x])
        sent_t = np.array(sentences_t[up_y])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))
        
        sent_s = np.array(sentences_s[min(x,up_x+1)])
        sent_t = np.array(sentences_t[up_y])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[up_x])
        sent_t = np.array(sentences_t[min(y,up_y+1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(cos_similarity_1>(threshold - relax_rate*continuous_length)):
            if(up_y!=y):
                sequence.insert(0,up_y)
            #print(up_x, up_y)
            up_x-=1
            up_y-=1
            k-=1
            continuous_length+=1
        elif(cos_similarity_2>(threshold - relax_rate*continuous_length)):
            if(up_y!=y):
                sequence.insert(0,up_y)
            #print(up_x+1, up_y)
            up_y-=1
            continuous_length+=1
        elif(cos_similarity_3>(threshold - relax_rate*continuous_length)):
            #print(up_x, up_y+1)
            up_x-=1
            k-=1
            #continuous_length+=1
        else:
            break
       
    
    #print(k,sequence)
    while (down_x<len(sentences_s)) and (down_y<len(sentences_t)):
        sent_s = np.array(sentences_s[down_x])
        sent_t = np.array(sentences_t[down_y])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[max(x,down_x-1)])
        sent_t = np.array(sentences_t[down_y])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[down_x])
        sent_t = np.array(sentences_t[max(y,down_y-1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(cos_similarity_1>(threshold - relax_rate*continuous_length)):
            #print(down_x, down_y)
            sequence.append(down_y)
            down_x+=1
            down_y+=1
            j+=1
            continuous_length+=1
        elif(cos_similarity_2>(threshold - relax_rate*continuous_length)):
            #print(down_x-1, down_y)
            sequence.append(down_y)
            down_y+=1
            continuous_length+=1
        elif(cos_similarity_3>(threshold - relax_rate*continuous_length)):
            #print(down_x, down_y-1)
            down_x+=1
            j+=1
            #continuous_length+=1
        else:
            #print(sequence)
            return [sequence,j, min(-1,k)]
        
    #print(sequence)
    return [sequence,j, min(-1,k)]

def get_matching_sequence_3(x,y,sentences_s, threshold, sentences_t):
    up_x, up_y = x, y
    down_x, down_y = x, y
    sequence = []
    j = 0
    k = 0
    commutative_similarity = 0
    continuous_length = 0

    while (up_x>=0) and (up_y>=0):
        
        sent_s = np.array(sentences_s[up_x])
        sent_t = np.array(sentences_t[up_y])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))
        
        sent_s = np.array(sentences_s[min(x,up_x+1)])
        sent_t = np.array(sentences_t[up_y])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[up_x])
        sent_t = np.array(sentences_t[min(y,up_y+1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(((commutative_similarity + cos_similarity_1)/(continuous_length+1)) > threshold):
            if(up_y!=y):
                sequence.insert(0,up_y)
            #print(up_x, up_y)
            up_x-=1
            up_y-=1
            k-=1
            continuous_length+=1
            commutative_similarity += cos_similarity_1

        elif(((commutative_similarity + cos_similarity_2)/(continuous_length+1)) > threshold):
            if(up_y!=y):
                sequence.insert(0,up_y)
            #print(up_x+1, up_y)
            up_y-=1
            continuous_length+=1
            commutative_similarity += cos_similarity_2

        elif(((commutative_similarity + cos_similarity_3)/(continuous_length+1)) > threshold):
            #print(up_x, up_y+1)
            up_x-=1
            k-=1
            continuous_length+=1
            commutative_similarity += cos_similarity_3

        else:
            break
       
    
    #print(k,sequence)
    while (down_x<len(sentences_s)) and (down_y<len(sentences_t)):
        sent_s = np.array(sentences_s[down_x])
        sent_t = np.array(sentences_t[down_y])
        cos_similarity_1 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[max(x,down_x-1)])
        sent_t = np.array(sentences_t[down_y])
        cos_similarity_2 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        sent_s = np.array(sentences_s[down_x])
        sent_t = np.array(sentences_t[max(y,down_y-1)])
        cos_similarity_3 = np.dot(sent_s,sent_t.T)/(np.sqrt(np.dot(sent_s,sent_s.T))*np.sqrt(np.dot(sent_t,sent_t.T)))

        if(((commutative_similarity + cos_similarity_1)/(continuous_length+1)) > threshold):
            #print(down_x, down_y)
            sequence.append(down_y)
            down_x+=1
            down_y+=1
            j+=1
            continuous_length+=1
            commutative_similarity += cos_similarity_1

        elif(((commutative_similarity + cos_similarity_2)/(continuous_length+1)) > threshold):
            #print(down_x-1, down_y)
            sequence.append(down_y)
            down_y+=1
            continuous_length+=1
            commutative_similarity += cos_similarity_2

        elif(((commutative_similarity + cos_similarity_3)/(continuous_length+1)) > threshold):
            #print(down_x, down_y-1)
            down_x+=1
            j+=1
            continuous_length+=1
            commutative_similarity += cos_similarity_3
            
        else:
            #print(sequence)
            return [sequence,j, min(-1,k)]
        
    #print(sequence)
    return [sequence,j, min(-1,k)]
		
def get_matching_partials(sentences_s, sentences_t):
    threshold = 0.75
    threshold_length = 1
    
    all_matches = []
    #matching_partials = []
    #matching_sources = []
    i=0
    while i<len(sentences_s):
        
        lst = u.get_nns_by_vector(sentences_s[i], 3, 100000)
        best_sequence = []
        #matching_source_indices = []
        #matching_sequences = []
        matches = []
        #print(i)
        matched_count = 0
        for candidate in lst:
            #print('candidate ',candidate)
            #sequence, j = get_matching_sequence(candidate,sentences_s[i:], threshold,sentences_t[candidate:])
            
            sequence, j, k = get_matching_sequence_3(i,candidate,sentences_s, threshold,sentences_t)
            if(len(sequence)>len(best_sequence)):
                best_sequence = sequence
                #print(best_sequence)
                matched_count = j
            if (((j-k-1)>threshold_length) or (len(sentences_s)<=threshold_length) ):
                #print(sequence)
                #print(j)
                #matching_sequences.append(sequence)
                #matching_source_indices.append([i+k+1, i+j])
                matches.append(([i+k+1, i+j],sequence))
        
        i = i+max(matched_count,1) #last_in_sequence
        
        #if(len(best_sequence)>0):
        #    matching_partials.append(best_sequence)
        #matching_partials.extend(matching_sequences)
        #matching_sources.extend(matching_source_indices)
        all_matches.extend(matches)
    #return matching_partials
    return  all_matches#[matching_partials, matching_sources, all_matches]

def column(matrix, i):
    return [row[i] for row in matrix]

def post_processor(diagonals, gap_threshold, min_length):
    diagonals = sorted(diagonals,key=lambda x: x[0][0])
    #gap_threshold = 40
    #print(diagonals)
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
            if ((gap_s <= gap_threshold) and (gap_s>=0) and (gap_t <= gap_threshold) and (gap_t >=0)):
                s = [_s for _s in range(pro_diagonal[0][0], diagonal[0][-1] + 1)]
                t = [_t for _t in range(pro_diagonal[1][0], diagonal[1][-1] + 1)]
                new_diagonal = (s, t)
                temp = temp[:-1]
                break
            elif((gap_s <= gap_threshold) and (gap_t <= gap_threshold) and (pro_diagonal[1][0] <= diagonal[1][0])):
                s = [_s for _s in range(pro_diagonal[0][0], max(pro_diagonal[0][-1],diagonal[0][-1]) + 1)]
                t = [_t for _t in range(pro_diagonal[1][0], max(pro_diagonal[1][-1],diagonal[1][-1]) + 1)]
                new_diagonal = (s, t)
                temp = temp[:-1]
                break

        temp.append(new_diagonal)
        temp.extend(processed_diagonals[last + 1:])
        processed_diagonals = temp
    #print(processed_diagonals)

    final_diagonals = []
    for diagonal in processed_diagonals:
        if((len(diagonal[1])>min_length)):# or (len(diagonal[1])>1)):
         final_diagonals.append(diagonal)

    return final_diagonals

file1  = open('./embedded_data/embedding_dms_checked.json',encoding='utf8')
data1 = json.load(file1)

file2  = open('./data/longdoc_dms_si_checked.json',encoding='utf8')
data2 = json.load(file2)

file3  = open('./embedded_data/embedding_dms_si_checked.json',encoding='utf8')
data3 = json.load(file3)

source_lang = 'en'
target_lang = 'si'

#long_doc = data2['content_'+target_lang]
#target_embedd = data3[0]['embed_'+ target_lang] #get_embeddig_list(long_doc, "si")
#sentences_t = doc_to_sentence(long_doc,target_lang)

combined_target_doc = data2['content_'+target_lang]
combined_target_embedd = data3[0]['embed_'+ target_lang]
sentences_in_combined_target_doc = doc_to_sentence(combined_target_doc,target_lang)

start = datetime.now()

recall_numerator = 0
precision_numerator = 0
precision_denominator = 0
source_doc_count = 25

#print(u.get_n_items())
#rint(len(sentences_in_combined_target_doc))
count = 0
for doc in data1[:source_doc_count]:
    print(count)
    count+=1

    doc_s = doc['content_' + source_lang]
    source_embedd = doc['embed_' + source_lang]  # get_embeddig_list(doc_s)
    
    target_doc = doc['content_' + target_lang]
    actual_target_sentences = doc_to_sentence(target_doc, target_lang)

    sentences_in_source = doc_to_sentence(doc_s, source_lang)

    all_matches = get_matching_partials(source_embedd, combined_target_embedd)
    all_matches = post_processor(all_matches, 10, 1)
    matching_partials = column(all_matches, 1)
    matching_sources = column(all_matches, 0)
    
    true_predicted_target_sentences = set()
    predicted_target_sentences = set()

    for index, diagonal in enumerate(matching_partials):

        for j in diagonal:
   
            sent = sentences_in_combined_target_doc[j]
            predicted_target_sentences.add(sent)
           
            if(sent in actual_target_sentences):
                true_predicted_target_sentences.add(sent)

    recall_numerator += len(true_predicted_target_sentences)/len(actual_target_sentences)
    if len(predicted_target_sentences) > 0 :
        precision_numerator += len(true_predicted_target_sentences)/len(predicted_target_sentences)
        precision_denominator+=1

print(datetime.now()-start)
recall = recall_numerator / source_doc_count
print("recall: ",recall)

if(precision_denominator>0):
    precision = precision_numerator/ precision_denominator
    print("precision: ",precision)
else:
    print('precision not defined')
f1 = (2*precision*recall)/(recall+precision)
print("f1: ", f1)
