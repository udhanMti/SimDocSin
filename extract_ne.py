import re
import json

en_designations_file = open("data/designation.en", encoding="utf8")
si_designations_file = open("data/designation.si", encoding="utf8")

designation_dict = dict()

en_designations = []
for i,j in enumerate(en_designations_file):
   en_designations.append(j.rstrip())
en_designations_file.close()

si_designations = []
for i,j in enumerate(si_designations_file):
   si_designations.append(j.rstrip())
   designation_dict[en_designations[i]] = j.rstrip()
   designation_dict[j.strip()] = en_designations[i]

si_designations_file.close()

def extract_designations(doc, lang):
    result = []
    if(lang=='en'):
      result = [k for k in en_designations if re.search(r'\b{}\b'.format(k), doc)]
    elif(lang=='si'):
      result = [k for k in si_designations if re.search(r'\b{}\b'.format(k), doc)]
    return (set(result))


en_names_file = open("data/person-names.en", encoding="utf8")
si_names_file = open("data/person-names.si", encoding="utf8")

names_dict = dict()

en_names = []
for i,j in enumerate(en_names_file):
   en_names.append(j.rstrip())
en_names_file.close()

si_names = []
for i,j in enumerate(si_names_file):
   si_names.append(j.rstrip())
   names_dict[en_names[i]] = j.rstrip()
   names_dict[j.strip()] = en_names[i]

si_names_file.close()

def extract_names(doc, lang):
    result = []
    if(lang=='en'):
        result = [k for k in en_names if ((len(k)>4) and (re.search(r'\b{}\b'.format(k), doc)))]
    elif(lang=='si'):
        result = [k for k in si_names if ((len(k)>4) and (re.search(r'\b{}\b'.format(k), doc)))]
    return set(result)

def get_ne_similarity(s_names, s_designations, t_doc):
    matching_names = [k for k in s_names if re.search(r'\b{}\b'.format(names_dict[k]), t_doc)]
    matching_names_count = len(set(matching_names))

    matching_designations = [k for k in s_designations if re.search(r'\b{}\b'.format(designation_dict[k]), t_doc)]
    matching_designations_count = len(set(matching_designations))

    similarity = 0
    if(len(s_names)>0):
        similarity += float(matching_names_count)/len(s_names)
    if(len(s_designations)>0):
        similarity += float(matching_designations_count) / len(s_designations)
    return similarity



