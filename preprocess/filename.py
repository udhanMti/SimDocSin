import pathlib

def get_file_paths(lang):
    # define the root folder
    path_parent = pathlib.Path(__file__).parents[1]

    # define the embedding folder
    path_embeddings = path_parent / 'Embeddings'

    # define the pattern
    pattern = "*.json"

    paths = []

    # define the Parallel embedding folder
    path_parallel = path_embeddings / 'Parallel'

    ### ADD Parallel Documents for both en and si
    for currentFile in path_parallel.glob(pattern):
        paths.append(str(currentFile))

    if lang == 'si':
        # define the Sinhala embedding folder
        path_sinhala = path_embeddings / 'Sinhala'

        ### ADD Sinhala Documents
        for currentFile in path_sinhala.glob(pattern):
            paths.append(str(currentFile))
        return paths

    else:
        # define the English embedding folder
        path_english = path_embeddings / 'English'

        ### ADD English Documents
        for currentFile in path_english.glob(pattern):
            paths.append(str(currentFile))
        return paths




