import numpy as np
import unicodedata
from spacy.lang.fr import French
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.util import compile_infix_regex, compile_prefix_regex
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


def typecolumns(df):
    df.debit[df.debit == ""] = np.NaN
    df.credit[df.credit == ""] = np.NaN
    df['debit'] = df.debit.astype(float)
    df['credit'] = df.credit.astype(float)
    df['credit_o_n'] = df.credit.isnull()
    df['lib_transaction'] = df.lib_transaction.astype(str)
    return df


# add STOP WORD
stop_words = list(STOP_WORDS) + ['VIR', 'vir', 'PRLV', 'prlv']


def spacy_nlp(nlp):
    customize_add_PUNCT = ['/', '=', '$', '|', '\\', "-"]
    for w in customize_add_PUNCT:
        nlp.vocab[w].is_punct = True

    # modify tokenizer infix patterns
    prefixes = (list(nlp.Defaults.prefixes) + ['/'])
    prefixes_regex = compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefixes_regex.search

    infixes = (list(nlp.Defaults.infixes) + ['(?<=[0-9])[|\/+\\-\\*^](?=[0-9-])'])
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    return nlp


def strip_accents_ascii(s):
    """Transform accentuated unicode symbols into ascii or nothing
    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.
    Parameters
    ----------
    s : string
        The string to strip
    See Also
    --------
    strip_accents_unicode
        Remove accentuated char for any unicode symbol.
    """
    nkfd_form = unicodedata.normalize('NFKD', s)
    return nkfd_form.encode('ASCII', 'ignore').decode('ASCII')


nlp = French()

my_nlp = spacy_nlp(nlp)


def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = my_nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower() for word in mytokens if
                not word.is_punct and not word.like_num and word.text != 'n']

    # Removing stop words
    mytokens = [word for word in mytokens if word not in stop_words]

    # Remove accentuated char for any unicode symbol
    mytokens = [strip_accents_ascii(word) for word in mytokens]

    # return preprocessed list of tokens
    return mytokens


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class MyLabelBinar(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(MyLabelBinar, self).fit(X)

    def transform(self, X, y=None):
        return super(MyLabelBinar, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(MyLabelBinar, self).fit(X).transform(X)


def MyPipeline(C=50, max_iter=200, class_weight=None):
    num_libilisation = 'lib_transaction'
    num_credit_oui_non = 'credit_o_n'
    # tfidf
    tfv = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2))

    features = FeatureUnion([
        (
            'credit_oui_non',
            Pipeline([('selector', DataFrameSelector(num_credit_oui_non)), ('encoder', MyLabelBinar())])),
        ('words', Pipeline([('selector', DataFrameSelector(num_libilisation)), ('wtfidf', tfv)]))])
    # model
    cla = LogisticRegression(C=C, n_jobs=-1, max_iter=max_iter, class_weight=class_weight, solver='newton-cg',
                             multi_class='multinomial')
    model = Pipeline([
        ('features', features), ('clf', cla)])

    return model


def MyPipeline_sous_categorie(C=50, max_iter=200, class_weight=None, n_estimators=500, max_depth=20):
    num_libilisation = 'lib_transaction'
    num_credit_oui_non = 'credit_o_n'
    num_categorie = 'lib_categorie'

    # tfidf
    tfv = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 2))

    features = FeatureUnion([
        (
            'credit_oui_non',
            Pipeline([('selector', DataFrameSelector(num_credit_oui_non)), ('encoder', MyLabelBinar())])),
        ('categorie', Pipeline([('selector_1', DataFrameSelector(num_categorie)), ('encoder_1', MyLabelBinar())])),
        ('words', Pipeline([('selector', DataFrameSelector(num_libilisation)), ('wtfidf', tfv)]))])
    # model
    cla = LogisticRegression(C=C, n_jobs=-1, max_iter=max_iter, class_weight=class_weight, solver='newton-cg',
                             multi_class='multinomial')
    model = Pipeline([
        ('features', features), ('clf', cla)])

    return model
