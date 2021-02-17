import json
import numpy as np
import os
import sys
# from extract_ne import extract_names, extract_designations
from extract_digits import extract_digits
from datetime import datetime
from preprocess.filename import get_file_paths

start = datetime.now()

args = sys.argv
lang = args[1]

si_digits = []

# en_names = []
# en_designations = []
# en_sent_count = []

print("Start Loading Target Documents")
paths= get_file_paths(lang)
i = 0
for filename in paths:
    file = open(filename, encoding='utf8')
    data = json.load(file)

    limit = len(data)

    for j in range(limit):
        ind = str(i // 1000)
        print(ind + " " + str(i % 1000))

        #save sinhala embeddings
        filename = "../db/" + ind + "/"+lang+"/" + str(i % 1000)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        b = np.array(data[j]["embed_"+lang])
        np.save(filename, b)


        # save sinhala weightings
        filename = "../db/" + ind + "/w"+lang+"/" + str(i % 1000)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        b = np.array(data[j]["weight_"+lang])
        np.save(filename, b)

        doc_si = data[j]["content_"+lang]
        si_digits.append(extract_digits(doc_si, lang))

        # save sinhala document
        filename = "../db/" + ind + "/doc"+lang+"/"
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        f = open("../db/" + ind + "/doc"+lang+"/" + str(i % 1000) + ".txt", "w", encoding='utf-8')
        f.write(doc_si)
        f.close()
        i = i + 1

# save english names
# filename = "./army/en_sent_count"
# if not os.path.exists(os.path.dirname(filename)):
#     try:
#         os.makedirs(os.path.dirname(filename))
#     except OSError as exc:  # Guard against race condition
#         if exc.errno != errno.EEXIST:
#             raise
#
# b = np.array(en_sent_count)
# np.save(filename, b)


# save sinhala digits
filename = "../db/"+lang+"_digits"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

b = np.array(si_digits)
np.save(filename, b)


# save english names
# filename = "./army/en_names"
# if not os.path.exists(os.path.dirname(filename)):
#     try:
#         os.makedirs(os.path.dirname(filename))
#     except OSError as exc:  # Guard against race condition
#         if exc.errno != errno.EEXIST:
#             raise
#
# b = np.array(si_digits)
# np.save(filename, b)
#


# save english designation
# filename = "./army/en_designations"
# if not os.path.exists(os.path.dirname(filename)):
#     try:
#         os.makedirs(os.path.dirname(filename))
#     except OSError as exc:  # Guard against race condition
#         if exc.errno != errno.EEXIST:
#             raise
#
# b = np.array(si_digits)
# np.save(filename, b)
#

# print(datetime.now() - start)
