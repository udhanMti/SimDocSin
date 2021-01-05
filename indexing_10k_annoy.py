import json
from annoy import AnnoyIndex

path = '/content/gdrive/Shared drives/FYP/Embedded Data'
file_names = ['embedding_army_0_1000','embedding_defence_0_705',
'embedding_army_1000_1039', 'embedding_hiru_cleaned','embedding_dms_pairs_1_100','embedding_wsws_new_222',
'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
'embedding_dms_pairs_101_150','embedding_dms_pairs_151_175','embedding_dms_pairs_176_200','embedding_dms_pairs_201_225','embedding_dms_pairs_226_237',
'embedding_dms_non_pairs','embedding_itn_sinhala_0_1000','embedding_newslk_sin_0_1000','embedding_sirasa_local_sinhala_0_1000','embedding_sirasa_local_sinhala_1000_2000','embedding_hiru_40000_41000']
#'embedding_dms_circular','embedding_dms_official_letters',
# file  = open('/content/drive/My Drive/FYP-Colab_defence/embedded_data/pip_embedded_wsws_200.json',encoding='utf8')
# embed_data = json.load(file)

sent_to_doc_map = {}

f = 1024
t = AnnoyIndex(f, 'euclidean')


count = 0
i=0
for file_name in file_names:
    file  = open(path + '/' +file_name + '.json' ,encoding='utf8')
    embed_data = json.load(file)

    if (file_name == file_names[-1]):
        rest = 676
        for j in range(rest):
            si_doc =embed_data[j]['content_si']
            si_doc_embed = embed_data[j]['embed_si']

            # sentences = doc_to_sentence(si_doc)
            for k in range(len(si_doc_embed)):
                sent_to_doc_map[i]=j
                t.add_item(i, si_doc_embed[k])
                i=i+1
        count = count + rest
        print (file_name + " : " + str(rest))

    else:
        for j in range(len(embed_data)):
            si_doc =embed_data[j]['content_si']
            si_doc_embed = embed_data[j]['embed_si']

            # sentences = doc_to_sentence(si_doc)
            for k in range(len(si_doc_embed)):
                sent_to_doc_map[i]=j
                t.add_item(i, si_doc_embed[k])
                i=i+1
        count = count + len(embed_data)
        print (file_name + " : " + str(len(embed_data)))
    

print (count)
# s_file  = open('/content/drive/My Drive/FYP-Colab_defence/embedded_data/pip_embedded_defence_1000.json',encoding='utf8')
# s_embed_data = json.load(s_file)

# for j in range(len(s_embed_data)):
#     si_doc = s_embed_data[j]['content_si']
#     si_doc_embed = s_embed_data[j]['embed_si']

#     # sentences = doc_to_sentence(si_doc)
#     for k in range(len(si_doc_embed)):
#         sent_to_doc_map[i]=j
#         t.add_item(i, si_doc_embed[k])
#         i=i+1

t.build(25)
t.save('ann/test.ann')

with open("ann/sent_to_doc_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_to_doc_map, outfile, ensure_ascii=False)