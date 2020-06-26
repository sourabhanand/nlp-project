import os
import gensim
#from models.lda import do_lda
from datautils.utils import get_processed_data, lazy_load_processed_data

DATA_LOC = os.path.join('data', '2020-06-20')

class TxtCorpus:
    def __init__(self, data_loc):
        self.data_loc = data_loc
        self.dictionary = gensim.corpora.Dictionary(lazy_load_processed_data(data_loc))

    def __iter__(self):
        for tokens in lazy_load_processed_data(self.data_loc):
            yield self.dictionary.doc2bow(tokens)

    def get_dict(self):
        return self.dictionary

def main():
    corpus = TxtCorpus(DATA_LOC)
    print('Training LDA Model')
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           num_topics=20,
                                           id2word=corpus.get_dict(),
                                           passes=1,
                                           workers=2)
    print('Saving LDA Model')
    lda_model.save('lda.model')

    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic: {idx} \nWords: {topic}')

if __name__ == '__main__':
    main()

