import json
import numpy as np
import os
# from extract_ne import extract_names, extract_designations, get_ne_similarity
# from extract_digits import extract_digits, get_digit_similarity
from datetime import datetime
# from doc_to_sentence import doc_to_sentence

start = datetime.now()

en_digits = []
si_digits = []

en_names = []
en_designations = []
en_sent_count = []

print("Start Loading Target Documents")

path = '/content/drive/Shared drives/FYP/1M_db/'

file_names = ['army_200', 'dms_50', 'hiru_200','defence_200',
               'wsws_50', 'embedding_army_1000_1039',
              'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
              'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
              'embedding_dms_pairs_101_150', 'embedding_dms_pairs_151_175', 'embedding_dms_pairs_176_200',
              'embedding_dms_pairs_201_225', 'embedding_dms_pairs_226_237',
              'embedding_dms_non_pairs']

sinhala_data_types = ['embedding_itn_sinhala_', 'embedding_sirasa_local_sinhala_', 'embedding_newslk_sin_',
                      'embedding_hiru_','embedding_hiru_']

end_no = [22000, 30000, 10000, 43000, 87000]
start_no = [0, 0, 0, 0 , 44000]
for i in range(0, 5):
    for j in range(start_no[i], end_no[i], 1000):
        f_name = sinhala_data_types[i] + str(j) + '_' + str(j + 1000)
        file_names.append(f_name)

#limit = 655
paths=[]
for file_name in file_names:
    paths.append(path + '/' + file_name + '.json')

# path = '/content/drive/Shared drives/FYP/Embedded Data'
#
# for i in range(0,200000,10000):
#   paths.append(path+"/para_crawl_"+str(i)+"_"+str(i+10000)+".json")
#
# for i in paths:
#     print(i)

i = 0
for file_name in paths:
    print(file_name)
    file = open(file_name, encoding='utf8')
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
            # save english embeddings
            filename = "./army/" + ind + "/en/" + str(i % 1000)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            b = np.array(data[j]["embed_en"])
            np.save(filename, b)

            '''
            # save english weightings
            filename = "./army/" + ind + "/wen/" + str(i % 1000)
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            b = np.array(data[j]["weight_en"])
            np.save(filename, b)
            '''
            doc_en = data[j]["content_en"]
            # en_digits.append(extract_digits(doc_en, 'en'))
            # en_names.append(extract_names(doc_en, 'en'))
            # en_designations.append(extract_designations(doc_en, 'en'))
            # en_sent_count.append(len(doc_to_sentence(doc_en, 'en')))

            # save english documents
            filename = "./army/" + ind + "/docen/"
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            f = open("./army/" + ind + "/docen/" + str(i % 1000) + ".txt", "a")
            f.write(doc_en)
            f.close()

        # save sinhala embeddings
        filename = "./army/" + ind + "/si/" + str(i % 1000)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        b = np.array(data[j]["embed_si"])
        np.save(filename, b)

        # save sinhala weightings
        # filename = "./army/" + ind + "/wsi/" + str(i % 1000)
        # if not os.path.exists(os.path.dirname(filename)):
        #     try:
        #         os.makedirs(os.path.dirname(filename))
        #     except OSError as exc:  # Guard against race condition
        #         if exc.errno != errno.EEXIST:
        #             raise
        #
        # b = np.array(data[j]["weight_si"])
        # np.save(filename, b)

        doc_si = data[j]["content_si"]
        # si_digits.append(extract_digits(doc_si, 'si'))

        # save sinhala document
        filename = "./army/" + ind + "/docsi/"
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        f = open("./army/" + ind + "/docsi/" + str(i % 1000) + ".txt", "a")
        f.write(doc_si)
        f.close()
        i = i + 1
'''
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
filename = "./army/si_digits"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

b = np.array(si_digits)
np.save(filename, b)

# save english digits
filename = "./army/en_digits"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

b = np.array(en_digits)
np.save(filename, b)

# save english names
filename = "./army/en_names"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

b = np.array(en_names)
np.save(filename, b)

# save english designation
filename = "./army/en_designations"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

b = np.array(en_designations)
np.save(filename, b)

# print(datetime.now() - start)
'''