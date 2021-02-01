import json
import numpy as np
import os
# from extract_ne import extract_names, extract_designations
from extract_digits import extract_digits
from datetime import datetime
from preprocess.filename import get_file_paths

start = datetime.now()

en_digits = []
si_digits = []

# en_names = []
# en_designations = []
# en_sent_count = []

print("Start Loading Target Documents")
paths= get_file_paths()
i = 0
for file_name in paths:
    print(file_name)
    file = open(filename, encoding='utf8')
    data = json.load(file)

    limit = len(data)

    for j in range(limit):
        ind = str(i // 1000)
        print(ind + " " + str(i % 1000))
        if i < 0:

            #save english embeddings
            filename = "../db/" + ind + "/en/" + str(i % 1000)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            b = np.array(data[j]["embed_en"])
            np.save(filename, b)


            # save english weightings
            filename = "../db/" + ind + "/wen/" + str(i % 1000)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            b = np.array(data[j]["weight_en"])
            np.save(filename, b)

            doc_en = data[j]["content_en"]
            en_digits.append(extract_digits(doc_en, 'en'))
            # en_names.append(extract_names(doc_en,'en'))
            # en_designations.append(extract_designations(doc_en,'en'))
            # en_sent_count.append(len(doc_to_sentence(doc_en,'en')))

            # save english documents
            filename = "../db/" + ind + "/docen/"
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            f = open("../db/" + ind + "/docen/" + str(i % 1000) + ".txt", "w")
            f.write(doc_en)
            f.close()

        #save sinhala embeddings
        filename = "../db/" + ind + "/si/" + str(i % 1000)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        b = np.array(data[j]["embed_si"])
        np.save(filename, b)


        # save sinhala weightings
        filename = "../db/" + ind + "/wsi/" + str(i % 1000)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        b = np.array(data[j]["weight_si"])
        np.save(filename, b)

        doc_si = data[j]["content_si"]
        si_digits.append(extract_digits(doc_si, 'si'))

        # save sinhala document
        filename = "../db/" + ind + "/docsi/"
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        f = open("../db/" + ind + "/docsi/" + str(i % 1000) + ".txt", "w", encoding='utf-8')
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
filename = "../db/si_digits"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

b = np.array(si_digits)
np.save(filename, b)

# save english digits
filename = "../db/en_digits"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

b = np.array(en_digits)
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
