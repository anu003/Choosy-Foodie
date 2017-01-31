from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

class TopicModelling(object):

    def __init__(self, model, vectorizer, data_path):
        self.model = model
        self.vectorizer = vectorizer
        self.data = None
        self.feature_names = None
        self.doctopic = None
        self.topics = None
        self._load_data(data_path)

    def _load_data(self, data_path):
        """
        input : string
        Loads data and returns
        """
        data = pd.read_pickle(data_path)
        docs = [' '.join(sent_words) for review in data['nouns'].values for sent_words in review]
        sentence_list = []
        for sentence in docs:
            word_list = []
            for word in sentence.split():
                if not word.encode('utf-8').isdigit():
                    word_list.append(word)
            sentence_list.append(' '.join(word_list))
        data['nouns'] = np.array(sentence_list)
        self.data = data
        docs = [' '.join(sent_words) for review in data['nouns'] for sent_words in review]
        self._get_topics(docs)

    def _print_top_words(self, feature_names, n_top_words):
        for topic_idx, topic in enumerate(self.model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def _get_topics(self, docs):

        topic_words = []
        sent_topics = []

        tfidf = self.vectorizer.fit_transform(docs)
        tfidf_feature_names = self.vectorizer.get_feature_names()
        self.feature_names = tfidf_feature_names
        doctopic = self.model.fit_transform(tfidf)
        doctopic_orig = doctopic.copy()
        sent_names = np.asarray([str(i) for i in xrange(len(docs))])
        num_groups = len(set(sent_names))
        doctopic_grouped = np.zeros((num_groups, n_topics))

        for topic in self.model.components_:
            word_idx = np.argsort(topic)[::-1][0:n_top_words]
            topic_words.append([tfidf_feature_names[i] for i in word_idx])

        for t in range(len(topic_words)):
            print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))

        for i, name in enumerate(sorted(set(sent_names))):
            doctopic_grouped[i, :] = np.mean(doctopic[sent_names == name, :], axis=0)

        doctopic = doctopic_grouped

        self.doctopic = doctopic
        for i in range(len(doctopic)):
            top_topics = np.argsort(doctopic[i,:])[-1]
            sent_topics.append(top_topics)
            # top_topics_str = ' '.join(str(t) for t in top_topics)
            # print("{}: {}".format(sent_names[i], top_topics_str))
        self.topics = sent_topics

    def assign_topics(self):
        ind = 0
        topics = []
        for review in self.data['sentences']:
            l = len(review)
            topics.append(self.topics[ind: ind+l])
            ind += l
        self.data['topics'] = np.array(topics)


if __name__ == "__main__":
    path = "../data/italian_extracted_review_data.pkl"
    n_topics = 10
    tfidf_vectorizer = TfidfVectorizer(max_df = 0.75, min_df = 50, max_features = 20000,
                                       stop_words='english')
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
    tm = TopicModelling(nmf, tfidf_vectorizer, path)
    tm.assign_topics()
    tm.data.to_pickle("../scored_data.pkl")

    for topic in nmf.components_:
        word_idx = np.argsort(topic)[::-1][0:n_top_words]
        topic_words.append([tfidf_feature_names[i] for i in word_idx])
