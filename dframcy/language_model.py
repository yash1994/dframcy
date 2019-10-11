# coding: utf-8
from __future__ import unicode_literals

import spacy


class LanguageModel(object):
    """
    Base class for language modelling tasks (annotation, training, matching and evaluation)
    """
    def __init__(self, nlp_model):
        """
        :param nlp_model: name of language model to be used.
        """
        self._nlp = None
        self.nlp_model = nlp_model

    def create_nlp_pipeline(self):
        try:
            nlp = spacy.load(self.nlp_model)
        except IOError:
            nlp = spacy.load("en_core_web_sm")
        return nlp

    @property
    def nlp(self):
        """
        :return: nlp pipeline for linguistic annotations
        """
        if not self._nlp:
            self._nlp = self.create_nlp_pipeline()
        return self._nlp

    def get_nlp(self):
        if not self._nlp:
            self._nlp = self.create_nlp_pipeline()
        return self._nlp
