from sentence_splitter import SentenceSplitter
from splitter.sentence_to_word import sentence_to_word
from splitter.tokenizer_new import SinhalaTokenizer as SinhalaTokenizer_new
import nltk.data
nltk.download('punkt')
# import sys
# sys.path.insert(1, '../../SimDocSin/')

from sinling import SinhalaTokenizer

tokenizer = SinhalaTokenizer()
tokenizer_new = SinhalaTokenizer_new()
splitter = SentenceSplitter(language='en', non_breaking_prefix_file='..\splitter\en.txt')


def doc_to_sentence_new(doc, lang):
    s = []
    if lang == 'en':
        s = splitter.split(text=doc)
    elif lang == 'si':
        s = tokenizer_new.split_sentences(doc)
    else:
        s = splitter.split(text=doc)
    return list(filter(None, s))


def doc_to_sentence_old(doc, lang):
    s = []
    if lang == 'en':
        s = splitter.split(text=doc)
    elif lang == 'si':
        s=tokenizer.split_sentences(doc)
        # s = splitter_s.split(text=doc)
    else:
        s = splitter.split(text=doc)
    return list(filter(None, s))

def doc_to_sentence(doc, lang):
    sentences = doc_to_sentence_new(doc, lang)
    min_tokens = 2 if lang == 'en' else 2
    max_tokens = 7 if lang == 'si' else 7
    total_sent = len(sentences)

    s = []
    join_sent = ''
    for i in range(len(sentences)):
        sent = sentences[i]
        no_tokens = sentence_to_word(sent, lang)
        join_sent = sent if join_sent == '' else join_sent + ' \n' + sent

        if i == total_sent - 1:
            s.append(join_sent)
            break
        if len(no_tokens) <= min_tokens:
            join_tokens = sentence_to_word(join_sent, lang)
            if len(join_tokens) >= max_tokens:
                s.append(join_sent)
                join_sent = ''
        else:
            s.append(join_sent)
            join_sent = ''
    return s