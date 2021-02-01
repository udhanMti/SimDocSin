from annoy import AnnoyIndex
print("Start Index Creation")
import json
from datetime import datetime
from preprocess.filename import get_file_paths
start = datetime.now()

en_digits = []
si_digits = []

en_names = []
en_designations = []
en_sent_count = []

print("Start Loading Target Documents")
paths = get_file_paths()
sent_to_doc_map = {}

f = 1024
t = AnnoyIndex(f, 'euclidean')
t.on_disk_build('test.ann')

sent_count ={}
sent_count[0] = 0
count = 0
i = 0
document_count=0
for file_name in paths:
    file = open(file_name)
    embed_data = json.load(file)

    for j in range(len(embed_data)):
        # si_doc = embed_data[j]['content_si']
        si_doc_embed = embed_data[j]['embed_si']

        sent_count[document_count + 1] = sent_count[document_count] + len(si_doc_embed)
        # sentences = doc_to_sentence(si_doc)
        for k in range(len(si_doc_embed)):
            sent_to_doc_map[i] = document_count
            t.add_item(i, si_doc_embed[k])
            i = i + 1
        document_count += 1
    count = count + len(embed_data)
    print(file_name + " : " + str(len(embed_data)))

print("Create indexes for Total" + str(count) + "Traget Documents")

t.build(25)
# t.save('test.ann')

with open("sent_to_doc_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_to_doc_map, outfile, ensure_ascii=False)

with open("sent_count_map.json", 'w', encoding="utf8") as outfile:
    json.dump(sent_count, outfile, ensure_ascii=False)