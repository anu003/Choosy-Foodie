from spacy.en import English
from pattern.en import lemma, sentiment
from spacy.parts_of_speech import NOUN, VERB, ADV, ADJ
from nltk.corpus import stopwords
import apriori
import pandas as pd
import numpy as np

class FeatureAndOpinionExtractor(object):
    """
    Extracts Features and Opinions from reviews.
    """
    def __init__(self, data, lang):
        self.data = data
        self.nlp = lang()
        self.features = []
        self._preprocess()
        self._get_sentiment()

    def _preprocess(self):
        """
        Preprocesses the data and calls the functions to extract features and its opinions.
        """
        self.data['sentences'] = self.data['text'].apply(self._tokenize_sent)
        self._unlistify('sentences')
        self.data['nouns'] = self.data['sentence'].apply(self._get_nouns)
        self._extract_opinions()

    def _tokenize_sent(self, review):
        """
        input : string
        output : list
        Returns list of sentences of a review.
        """
        return review.decode('ascii','ignore').split('.')

    def _unlistify(self, col):
        """
        input : string
        Unlistifies the list of elements as new rows in a dataframe.
        """
        s = self.data.apply(lambda x: pd.Series(x[col]),axis=1).stack().reset_index(level=1, drop=True)
        s.name = col[:-1]
        self.data = self.data.drop(col, axis = 1).join(s)

    def _get_nouns(self, sentence):
        """
        input : String
        output: list
        Returns features(nouns) for each sentence.
        """
        doc = self.nlp(sentence)
        nouns = [unicode(lemma(str(word).lower())) for word in doc if word.pos == NOUN]
        return nouns

    def _extract_pos(self, sentence, pos):
        """
        input : string, string
        output : list of strings
        Returns the list of words that has parts of speech as given in the input.
        """
        pos_list = []
        stop = stopwords.words('english')
        doc = self.nlp(unicode(sentence))
        pos_list = [unicode(word) for word in doc if word.pos == pos and str(word).lower().encode('utf-8') not in stop]
        return pos_list

    def _extract_opinions(self):
        """
        Extracts adjectives, adverbs, verbs for each sentence of a review.
        """
        self.data['adjectives'] = self.data['sentence'].apply(lambda x: self._extract_pos(x, ADJ))
        self.data['adverbs'] = self.data['sentence'].apply(lambda x: self._extract_pos(x, ADV))
        self.data['verbs'] = self.data['sentence'].apply(lambda x: self._extract_pos(x, VERB))
        self._get_polarity()

    def _get_polarity(self):
        """
        Calculates polarity based on sentiment of a sentence
        """
        self.data['polarity'] = self.data['sentence'].apply(lambda x: [sentiment(i) for i in x])
        polarities = [polarity for sent_polarities in self.data['polarity'].values for polarity in sent_polarities]
        self._get_normalized_score(polarities)

    def _get_normalized_score(self, polarities):
        """
        input : list
        Calculates scores(ranging between 0-5) for each sentence
        """
        normalized_scores = pd.cut(polarities, bins=9, right=True, labels = [1,1.5,2,2.5,3,3.5,4,4.5,5],
                    retbins=False, precision=2, include_lowest=True)
        self.data['scores'] = np.array(normalized_scores)

    def _get_features(self):
        """
        Extracts features from each sentence of a review.
        """
        stop = set(stopwords.words('english'))
        review_features = []
        for features in self.data['nouns'].values:
            sentence_features = []
            for feature in features:
                if feature not in stop:
                    sentence_features.append(feature)
            review_features.append(sentence_features)
        self.features = review_features

if __name__ == "__main__":
    reviews = pd.read_pickle("../data/italian_cleaned_review_data.pkl")
    foe = FeatureAndOpinionExtractor(reviews, English)
    foe.to_pickle('../data/italian_extracted_review_data.pkl')
