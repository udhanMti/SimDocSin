import json
import sys
sys.path.append('../../SimDocSin/')
from embedder.laser_control import get_embeddig_list
from weight_schema import *
import os
import errno

args =sys.argv
filepath = args[1]
file_type = args[2]
output_file = args[3]

file  = open(filepath,encoding='utf8')
data = json.load(file)

parallel =[]
file_path = ""
i=0

if file_type == 'si':
    for a in data:
        print(i)
        i += 1

        if ('content_si' in a):
            doc_si = a["content_si"]
            if (doc_si != ''):
                target_embedd = get_embeddig_list(doc_si, lang='si')
                a['embed_si'] = target_embedd
                a['weight_si'] = documentMassNormalization(get_sentence_length_weighting_list(doc_si, 'si'))

        parallel.append(a)
        file_path = "../embeddings/sinhala/" + output_file + ".json"


elif file_type == 'en':
    for a in data:
        print(i)
        i += 1
        if ('content_en' in a):
            doc_en = a['content_en']
            if (doc_en != ''):
                source_embedd = get_embeddig_list(doc_en, lang='en')
                a['embed_en'] = source_embedd
                a['weight_en'] = documentMassNormalization(get_sentence_length_weighting_list(doc_en, 'en'))

        parallel.append(a)
        file_path = "../embeddings/english/" + output_file + ".json"

else:
    for a in data:
          print(i)
          i+=1
          if ('content_si' in a):
              doc_si = a["content_si"]
              if (doc_si != ''):
                  target_embedd = get_embeddig_list(doc_si, lang='si')
                  a['embed_si'] = target_embedd
                  a['weight_si'] = documentMassNormalization(get_sentence_length_weighting_list(doc_si, 'si'))

          if('content_en' in a):
              doc_en = a['content_en']
              if (doc_en != ''):
                  source_embedd = get_embeddig_list(doc_en, lang='en')
                  a['embed_en'] = source_embedd
                  a['weight_en'] = documentMassNormalization(get_sentence_length_weighting_list(doc_en, 'en'))

          parallel.append(a)
          file_path = "../embeddings/parallel/" + output_file + ".json"


if not os.path.exists(os.path.dirname(file_path)):
    try:
        os.makedirs(os.path.dirname(file_path))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


if len(parallel) > 0 :
    with open(file_path, 'w', encoding="utf8") as outfile:
        json.dump(parallel, outfile, ensure_ascii=False)