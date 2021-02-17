def get_file_paths(lang):
    ### ADD Parallel Documents under both en and si blocks
    if lang == 'si':
        ### return paths to sinhala documents
        path = '/content/drive/Shared drives/FYP/1M_db/'
        file_names = ['embedding_army_0_1000', 'embedding_dms_pairs_1_100', 'hiru_200', 'embedding_defence_0_705',
                      'embedding_wsws_new_222', 'embedding_army_1000_1039',
                      'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
                      'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
                      'embedding_dms_pairs_101_150', 'embedding_dms_pairs_151_175', 'embedding_dms_pairs_176_200',
                      'embedding_dms_pairs_201_225', 'embedding_dms_pairs_226_237',
                      'embedding_dms_non_pairs']
        '''
        sinhala_data_types = ['embedding_itn_sinhala_', 'embedding_sirasa_local_sinhala_', 'embedding_newslk_sin_',
                              'embedding_hiru_', 'embedding_hiru_']
    
        end_no = [22000, 30000, 10000, 43000, 87000]
        start_no = [0, 0, 0, 0, 44000]
        for i in range(0, 5):
            for j in range(start_no[i], end_no[i], 1000):
                f_name = sinhala_data_types[i] + str(j) + '_' + str(j + 1000)
                file_names.append(f_name)
        '''

        paths = []
        for file_name in file_names:
            paths.append(path + '/' + file_name + '.json')

        '''
        path = '/content/drive/Shared drives/FYP/Embedded Data'
    
        for i in range(0, 5000000, 10000):
            paths.append(path + "/para_crawl_" + str(i) + "_" + str(i + 10000) + ".json")
        '''
        return paths
    else:
        ### return paths to english documents
        path = '/content/drive/Shared drives/FYP/1M_db/'
        file_names = ['embedding_army_0_1000', 'embedding_dms_pairs_1_100', 'hiru_200', 'embedding_defence_0_705',
                      'embedding_wsws_new_222', 'embedding_army_1000_1039',
                      'embedding_wsws_0_350', 'embedding_wsws_350_700', 'embedding_wsws_700_1000',
                      'embedding_wsws_1000_1350', 'embedding_wsws_1350_1700', 'embedding_wsws_1700_2000',
                      'embedding_dms_pairs_101_150', 'embedding_dms_pairs_151_175', 'embedding_dms_pairs_176_200',
                      'embedding_dms_pairs_201_225', 'embedding_dms_pairs_226_237',
                      'embedding_dms_non_pairs']
        paths = []
        for file_name in file_names:
            paths.append(path + '/' + file_name + '.json')
        return paths