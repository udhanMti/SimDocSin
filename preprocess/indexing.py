from annoy import AnnoyIndex
print("Start Index Creation")
import json
import sys
sys.path.append('../../SimDocSin/')
from datetime import datetime
from preprocess.filename import get_file_paths
start = datetime.now()

args = sys.argv
lang = args[1]

print("Start Loading Target Documents")
paths = get_file_paths(lang)
sent_to_doc_map = {}

f = 1024
t = AnnoyIndex(f, 'euclidean')
t.on_disk_build("../index/test_"+lang+".ann")

sent_count ={}
sent_count[0] = 0
count = 0
i = 0
document_count=0

for file_name in paths:
    file = open(file_name,encoding='utf-8')
    embed_data = json.load(file)

    for j in range(len(embed_data)):
        # si_doc = Embeddings[j]['content_si']
        si_doc_embed = embed_data[j]['embed_'+lang]

        sent_count[document_count + 1] = sent_count[document_count] + len(si_doc_embed)
        # sentences = doc_to_sentence(si_doc)
        for k in range(len(si_doc_embed)):
            sent_to_doc_map[i] = document_count
            t.add_item(i, si_doc_embed[k])
            i = i + 1
        document_count += 1
    count = count + len(embed_data)
    print(file_name + " : " + str(len(embed_data)))

print("Create indexes for Total" + str(count) + "Target Documents")

t.build(25)
# t.save('test.ann')

with open("../index/sent_to_doc_map_"+lang+".json", 'w', encoding="utf8") as outfile:
    json.dump(sent_to_doc_map, outfile, ensure_ascii=False)

with open("../index/sent_count_map_"+lang+".json", 'w', encoding="utf8") as outfile:
    json.dump(sent_count, outfile, ensure_ascii=False)