import json
import numpy as np
from laser_control import get_embeddig_list
from extract_digits import extract_digits
from extract_ne import extract_names, extract_designations

from weight_schema import *


file  = open('data/dms_checked.json',encoding='utf8')
data = json.load(file)

parallel =[]
i=0

for a in data[:25]:
      print(i)
      i+=1
      if ('content_en' in a) :
         doc_en = a['content_en']
         if (doc_en != ''):

            source_embedd = get_embeddig_list(doc_en,lang='en')
            a ['embed_en'] = source_embedd
            a ['weight_en'] = documentMassNormalization(get_sentence_length_weighting_list(doc_en, 'en'))

      if ('content_si' in a):
         doc_si = a["content_si"]
         if (doc_si != ''):

            target_embedd = get_embeddig_list(doc_si,lang='si')
            a ['embed_si'] = target_embedd
            a ['weight_si'] = documentMassNormalization(get_sentence_length_weighting_list(doc_si, 'si'))
      
      parallel.append(a)

if len(parallel) > 0 :
    with open("embedded_data/embedding_dms_checked.json", 'w', encoding="utf8") as outfile:
        json.dump(parallel, outfile, ensure_ascii=False)