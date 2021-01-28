import json
import numpy as np
import os
# from extract_ne import extract_names, extract_designations
from extract_digits import extract_digits
from datetime import datetime


start = datetime.now()

en_digits = []
si_digits = []

# en_names = []
# en_designations = []
# en_sent_count = []


print("Start Loading Target Documents")
path = '../embed_data'

file_names = ['embedding']
'''
file_names = ['embedding_army_0_1000', 'embedding_army_1000_1039', 
              'embedding_hiru_cleaned',
              'embedding_defence_0_705', 'embedding_wsws_new_222',
              'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
              'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
'embedding_dms_pairs_101_150','embedding_dms_pairs_151_175','embedding_dms_pairs_176_200','embedding_dms_pairs_201_225','embedding_dms_pairs_226_237',
'embedding_dms_non_pairs','embedding_itn_sinhala_0_1000','embedding_newslk_sin_0_1000','embedding_sirasa_local_sinhala_0_1000','embedding_sirasa_local_sinhala_1000_2000','embedding_hiru_40000_41000']



file_names = ['embedding_army_0_1000', 'embedding_army_1000_1039', 
              'embedding_hiru_cleaned',
              'embedding_defence_0_705', 'embedding_wsws_new_222',
              'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
              'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
              'embedding_dms_audit', 'embedding_dms_comm_reports', 'embedding_dms_gov_site', 'embedding_dms_others',
              'embedding_dms_annual_reports', 'embedding_dms_circular_sinhala',
              'embedding_dms_official_letters_sinhala']


sinhala_data_types = ['embedding_itn_sinhala_', 'embedding_sirasa_local_sinhala_', 'embedding_newslk_sin_',
                      'embedding_hiru_']

end_no = [22000, 30000, 10000, 43000]
start_no = [0, 0, 0, 10000]
for i in range(0, 4):
    for j in range(start_no[i], end_no[i], 1000):
        f_name = sinhala_data_types[i] + str(j) + '_' + str(j + 1000)
        file_names.append(f_name)

#limit = 655
'''
i = 0
for file_name in file_names:
    print(file_name)
    file = open(path + '/' + file_name + '.json', encoding='utf8')
    data = json.load(file)

    # if (file_name == file_names[-1]):
    #     limit = 676
    # else:
    #     limit = len(data)
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
