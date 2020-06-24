import nltk
import gensim


def do_lda(processed_df):
    processed_df = (processed_df['processed_paper_text'].dropna(axis=0).str.split())
    corpus_dict = gensim.corpora.Dictionary(processed_df)
    corpus_bow = [corpus_dict.doc2bow(doc) for doc in processed_df]
    lda_model = gensim.models.LdaMulticore(corpus_bow, num_topics=10,
                                           id2word=corpus_dict, passes=1,
                                           workers=2)
    for idx, topic in lda_model.print_topics(-1):
        print(f'Topic: {idx} \nWords: {topic}')
