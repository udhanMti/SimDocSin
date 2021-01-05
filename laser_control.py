import requests
from doc_to_sentence import doc_to_sentence#,doc_to_sentence_si
#import embed_custom
import numpy as np
import os
import sys
from laserembeddings import Laser

laser = Laser()

def sent_embedding(query_in, lang = 'en', address = '127.0.0.1:8050'):

    embeddings = laser.embed_sentences(query_in,lang=lang).tolist()

    return embeddings

def get_embeddig_list(doc,lang = 'en'):
    sentences = doc_to_sentence(doc, lang)
    embedding_list = sent_embedding(sentences,lang)

    return embedding_list
