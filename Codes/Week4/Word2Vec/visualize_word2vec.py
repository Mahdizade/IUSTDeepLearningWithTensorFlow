# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE

import arabic_reshaper


def main():
    num_words = 10000
    words = []
    vectors_list = []
    model = Word2Vec.load('fawiki_vectors.model')
    input_file = open('vocab_non_binary.txt', 'r+')
    words_counter = 0
    while words_counter < num_words:
        line = next(input_file)
        word = line.strip().split()[0].decode('utf8')
        words.append(word)
        vectors_list.append(model[word])
        words_counter += 1

    input_file.close()

    vectors = np.array(vectors_list, dtype=np.float)
    print(vectors.shape)

    tsne_model = TSNE(n_components=2, random_state=0, verbose=1)
    projected = tsne_model.fit_transform(vectors)
    x = projected[:, 0]
    y = projected[:, 1]
    plt.plot(x, y, '.', markersize=0)
    matplotlib.rc('font', family='Arial', size=1)
    for i in range(len(projected)):
        reshaped_text = arabic_reshaper.reshape(words[i])
        text = get_display(reshaped_text)
        plt.annotate(text, xy=(x[i], y[i]), horizontalalignment='center', verticalalignment='center')

    plt.axis('off')
    plt.autoscale(tight=True)
    plt.savefig('words.jpg', format='jpg', bbox_inches='tight', dpi=2000)


if __name__ == '__main__':
    main()
