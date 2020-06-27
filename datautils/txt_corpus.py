from gensim import corpora
from utils import lazy_load_processed_data


class TxtCorpus:
    def __init__(self, data_loc, use_saved_dict=True):
        self.data_loc = data_loc
        if use_saved_dict:
            self.dictionary = (corpora.Dictionary
                               .load('saved_models/lda.model.id2word'))
        else:
            self.dictionary = corpora.Dictionary(lazy_load_processed_data(data_loc))

    def __iter__(self):
        for tokens in lazy_load_processed_data(self.data_loc):
            yield self.dictionary.doc2bow(tokens)

    def get_dict(self):
        return self.dictionary