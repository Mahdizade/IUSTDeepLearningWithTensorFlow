import logging

from gensim.models.word2vec import Word2Vec
from hazm import WikipediaReader
from hazm import sent_tokenize, WordTokenizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = WordTokenizer(join_verb_parts=False)


class WikiSentences(object):
    def __init__(self, dump_file):
        self.dump_file = dump_file

    def __iter__(self):
        wiki = WikipediaReader(fawiki_dump=self.dump_file)
        for doc in wiki.docs():
            sentences = sent_tokenize(doc['text'])
            for sentence in sentences:
                # You should apply any preprocess before yield
                yield tokenizer.tokenize(sentence)


def main():
    fawiki_sentences = WikiSentences('fawiki-latest-pages-articles.xml.bz2')
    model = Word2Vec(fawiki_sentences, workers=4)
    model.save('fawiki_vectors.model')
    model.wv.save_word2vec_format('fawiki_vectors_binary.model', fvocab='vocab_binary.txt', binary=True)
    model.wv.save_word2vec_format('fawiki_vectors_non_binary.model', fvocab='vocab_non_binary.txt', binary=False)


if __name__ == '__main__':
    main()
