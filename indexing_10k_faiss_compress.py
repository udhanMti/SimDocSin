import json
import faiss 
import numpy as np

path = '/content/drive/Shareddrives/FYP/1M_db_embed_scrape'
file_names = ['embedding_army_0_1000',
'embedding_wsws_new_222',
'embedding_defence_0_705',
'embedding_dms_pairs_1_100',
'embedding_hiru_cleaned',
'embedding_army_1000_1039',
'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
'embedding_dms_pairs_101_150','embedding_dms_pairs_151_175','embedding_dms_pairs_176_200','embedding_dms_pairs_201_225','embedding_dms_pairs_226_237',
'embedding_dms_non_pairs','embedding_itn_sinhala_0_1000','embedding_newslk_sin_0_1000','embedding_sirasa_local_sinhala_0_1000','embedding_sirasa_local_sinhala_1000_2000','embedding_hiru_40000_41000']


sent_to_doc_map = {}

f = 1024

vectors = []
count = 0

i=0
document_count = 0

for file_name in file_names:
    file  = open(path + '/' +file_name + '.json' ,encoding='utf8')
    embed_data = json.load(file)

    if (file_name == file_names[-1]):
        rest = 676
        for j in range(rest):
            si_doc = embed_data[j]['content_si']
            si_doc_embed = embed_data[j]['embed_si']

            for k in range(len(si_doc_embed)):
                sent_to_doc_map[i]=document_count
                vectors.append(si_doc_embed[k])
                i=i+1
            document_count += 1
        count = count + rest
        print (file_name + " : " + str(rest))
        

    else:
        for j in range(len(embed_data)):
            si_doc = embed_data[j]['content_si']
            si_doc_embed = embed_data[j]['embed_si']

            for k in range(len(si_doc_embed)):
                sent_to_doc_map[i]=document_count
                vectors.append(si_doc_embed[k])
                i=i+1
            document_count += 1
        count = count + len(embed_data)
        print (file_name + " : " + str(len(embed_data)))       
  
print ("Sentence Count: " +str(count))
print ("Document Count: " +str(document_count))


m=128
nlist = 2  # number of centroids
quantizer = faiss.IndexFlatL2(f)  # coarse quantizer

index = faiss.IndexIVFPQ(quantizer, f, nlist, m, 8) #define the inverted index 
#IndexIVFPQ (Index *quantizer, size_t d, size_t nlist, size_t M, size_t nbits_per_idx)
#no of bits per index <=8


np_vectors = np.array(vectors).astype('float32')

index.train(np_vectors) # train on the database vectors
index.add(np_vectors)

faiss.write_index(index, "./faiss/faiss_index.index") # save the index to disk


with open("faiss/sent_to_doc_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_to_doc_map, outfile, ensure_ascii=False)