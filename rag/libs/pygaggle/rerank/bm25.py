from copy import deepcopy
import math
import re

import numpy as np

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from .base import Reranker, Query, Text


__all__ = ['Bm25Reranker']


class Bm25Reranker(Reranker):
    def __init__(self,
                 k1: float = 1.6,
                 b: float = 0.75,
                 tokenize_method: str = 'nlp'):
        self.k1 = k1
        self.b = b
        self.tokenize_method = tokenize_method

    def rescore(self, query: Query, texts: list[Text]) -> list[Text]:
        query_words = self.tokenize(query.text)
        sentences = self.tokenize([t.text for t in texts])
        
        bm25 = BM25Okapi(sentences, k1=self.k1, b=self.b)
        scores = bm25.get_scores(query_words)
        
        texts = deepcopy(texts)
        for score, text in zip(scores, texts):
            text.score = score
        return texts

    def tokenize(self, texts):
        is_single = False
        if isinstance(texts, str):
            is_single = True
            texts = [texts]
            
        if self.tokenize_method == 'nlp':
            stop_blank = set(stopwords.words('english')).union(set([' ']))

            texts = [word_tokenize(t) for t in texts]
            texts = [[re.sub("[^0-9a-zA-Z]", " ", token) for token in t] for t in texts]
            texts = [[token for token in t if token not in stop_blank] for t in texts]
        else:
            raise NotImplemented("Tokenize method not implemented")
        return texts[0] if is_single else texts



from rank_bm25 import BM25Plus
# import re

class Bm25PlusReranker(Reranker):
    def __init__(self,
                 k1: float = 1.6,
                 b: float = 0.75,
                 delta: float = 1.0,
                 tokenize_method: str = 'nlp'):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.tokenize_method = tokenize_method

    def rescore(self, query: Query, texts: list[Text]) -> list[Text]:
        query_words = self.tokenize(query.text)
        sentences = self.tokenize([t.text for t in texts])
        
        bm25plus = BM25Plus(sentences, k1=self.k1, b=self.b, delta=self.delta)
        scores = bm25plus.get_scores(query_words)
        
        texts = deepcopy(texts)
        for score, text in zip(scores, texts):
            text.score = score
        return texts

    def tokenize(self, texts):
        is_single = False
        if isinstance(texts, str):
            is_single = True
            texts = [texts]
            
        if self.tokenize_method == 'nlp':
            stop_blank = set(stopwords.words('english')).union(set([' ']))

            texts = [word_tokenize(t) for t in texts]
            texts = [[re.sub("[^0-9a-zA-Z]", " ", token) for token in t] for t in texts]
            texts = [[token for token in t if token not in stop_blank] for t in texts]
        else:
            raise NotImplemented("Tokenize method not implemented")
        return texts[0] if is_single else texts


class RM3BM25Reranker(Reranker):
    def __init__(self,
                 k1: float = 1.6,
                 b: float = 0.75,
                 top_k: int = 10,
                 tokenize_method: str = 'nlp',
                 alpha: float = 0.5):
        self.k1 = k1
        self.b = b
        self.top_k = top_k
        self.tokenize_method = tokenize_method
        self.alpha = alpha

    def rescore(self, query: Query, texts: list[Text]) -> list[Text]:
        query_words = self.tokenize(query.text)
        sentences = self.tokenize([t.text for t in texts])
        
        bm25 = BM25Okapi(sentences, k1=self.k1, b=self.b)
        initial_scores = bm25.get_scores(query_words)
        
        top_k_texts = [texts[i] for i in np.argsort(initial_scores)[-self.top_k:]]
        top_k_sentences = [sentences[i] for i in np.argsort(initial_scores)[-self.top_k:]]
        
        expanded_query_words = self.expand_query(query_words, top_k_sentences)
        
        final_scores = bm25.get_scores(expanded_query_words)
        
        texts = deepcopy(texts)
        for score, text in zip(final_scores, texts):
            text.score = score
        return texts

    def expand_query(self, query_words, top_k_sentences):
        term_freqs = {}
        for sentence in top_k_sentences:
            for term in sentence:
                term_freqs[term] = term_freqs.get(term, 0) + 1

        total_terms = sum(term_freqs.values())
        term_probs = {term: freq / total_terms for term, freq in term_freqs.items()}
        
        expanded_query = query_words + [term for term, prob in term_probs.items() if prob > self.alpha]
        
        return expanded_query

    def tokenize(self, texts):
        is_single = False
        if isinstance(texts, str):
            is_single = True
            texts = [texts]
            
        if self.tokenize_method == 'nlp':
            stop_blank = set(stopwords.words('english')).union(set([' ']))

            texts = [word_tokenize(t) for t in texts]
            texts = [[re.sub("[^0-9a-zA-Z]", " ", token) for token in t] for t in texts]
            texts = [[token for token in t if token not in stop_blank] for t in texts]
        else:
            raise NotImplemented("Tokenize method not implemented")
        return texts[0] if is_single else texts
    


class QLReranker(Reranker):
    def __init__(self,
                 mu: float = 2000,
                 tokenize_method: str = 'nlp'):
        self.mu = mu
        self.tokenize_method = tokenize_method

    def rescore(self, query: Query, texts: list[Text]) -> list[Text]:
        query_words = self.tokenize(query.text)
        sentences = self.tokenize([t.text for t in texts])
        
        collection_length = sum(len(sentence) for sentence in sentences)
        term_collection_freq = self.compute_term_collection_freq(sentences)
        
        texts = deepcopy(texts)
        for text, sentence in zip(texts, sentences):
            text.score = self.compute_query_likelihood(query_words, sentence, term_collection_freq, collection_length)
        return texts

    def compute_term_collection_freq(self, sentences):
        term_collection_freq = {}
        for sentence in sentences:
            for term in sentence:
                term_collection_freq[term] = term_collection_freq.get(term, 0) + 1
        return term_collection_freq

    def compute_query_likelihood(self, query_words, sentence, term_collection_freq, collection_length):
        score = 0.0
        sentence_length = len(sentence)
        
        for term in query_words:
            term_freq = sentence.count(term)
            collection_term_freq = term_collection_freq.get(term, 0)
            
            # Calculate smoothed probability
            smoothed_prob = (term_freq + self.mu * (collection_term_freq / collection_length)) / (sentence_length + self.mu)
            
            # Ensure smoothed_prob is not zero to avoid log(0)
            if smoothed_prob > 0:
                score += np.log(smoothed_prob)
        
        return score

    def tokenize(self, texts):
        is_single = False
        if isinstance(texts, str):
            is_single = True
            texts = [texts]
            
        if self.tokenize_method == 'nlp':
            stop_blank = set(stopwords.words('english')).union(set([' ']))

            texts = [word_tokenize(t) for t in texts]
            texts = [[re.sub("[^0-9a-zA-Z]", " ", token) for token in t] for t in texts]
            texts = [[token for token in t if token not in stop_blank] for t in texts]
        else:
            raise NotImplemented("Tokenize method not implemented")
        return texts[0] if is_single else texts
