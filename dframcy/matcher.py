# coding: utf-8
from __future__ import unicode_literals

import pandas as pd
from spacy.matcher import Matcher, PhraseMatcher


class DframCyMatcher(object):
    """
    Dataframe wrapper class over spaCy's Matcher
    https://spacy.io/api/matcher
    """

    def __init__(self, nlp_pipeline):
        """
        :param nlp_pipeline: nlp pipeline to be used (i.e. language model).
        """
        self._nlp = nlp_pipeline
        self._matcher = None

    @property
    def nlp(self):
        return self._nlp

    def get_nlp(self):
        return self._nlp

    def __call__(self, doc):
        """
        To find all token sequences matching the supplied patterns on the Doc
        :param doc: spacy container for linguistic annotations.
        :return: dataframe, containing matched occurrences.
        """
        df_format_json = {}
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            if "match_id" not in df_format_json:
                df_format_json["match_id"] = []
                df_format_json["match_id"].append(match_id)
            else:
                df_format_json["match_id"].append(match_id)
            if "start" not in df_format_json:
                df_format_json["start"] = []
                df_format_json["start"].append(start)
            else:
                df_format_json["start"].append(start)
            if "end" not in df_format_json:
                df_format_json["end"] = []
                df_format_json["end"].append(end)
            else:
                df_format_json["end"].append(end)
            if "string_id" not in df_format_json:
                df_format_json["string_id"] = []
                df_format_json["string_id"].append(self._nlp.vocab.strings[match_id])
            else:
                df_format_json["string_id"].append(self._nlp.vocab.strings[match_id])
            if "span_text" not in df_format_json:
                df_format_json["span_text"] = []
                df_format_json["span_text"].append(doc[start:end].text)
            else:
                df_format_json["span_text"].append(doc[start:end].text)
        matches_dataframe = pd.DataFrame.from_dict(df_format_json)
        matches_dataframe.reindex(matches_dataframe["match_id"])
        matches_dataframe.drop(columns=["match_id"], inplace=True)

        return matches_dataframe

    def get_matcher_object(self):
        """
        To get spaCy's matcher class object (used for testing only).
        :return: matcher object
        """
        return self._matcher

    def get_matcher(self):
        """
        To initialize spaCy's matcher class object.
        :return: Matcher object
        """
        return Matcher(self._nlp.vocab)

    def add(self, pattern_name, callback, *pattern):
        """
        To add patterns to spaCy's matcher object
        :param pattern_name: str, pattern name
        :param callback: function, callback function to be invoked on matched occurrences.
        :param pattern: list of patterns
        """
        if not self._matcher:
            self._matcher = self.get_matcher()
        self._matcher.add(pattern_name, callback, *pattern)

    def remove(self, pattern_name):
        """
        To remove pattern from spaCy's matcher object
        :param pattern_name: str, pattern_name
        """
        if self._matcher:
            self._matcher.remove(pattern_name)

    def reset(self):
        """
        To re-initialize spaCy's matcher object
        """
        self._matcher = self.get_matcher()


class DframCyPhraseMatcher(object):
    """
        Dataframe wrapper class over spaCy's PhraseMatcher
        https://spacy.io/api/phrasematcher
    """

    def __init__(self, nlp_pipeline, attr=None):
        """
        :param nlp_pipeline: nlp pipeline to be used (i.e. language model).
        :param attr: str, token attribute to match on (default: "ORTH")
        """
        self._nlp = nlp_pipeline
        self._phrase_matcher = None
        self.attribute = attr

    @property
    def nlp(self):
        return self._nlp

    def get_nlp(self):
        return self._nlp

    def __call__(self, doc):
        """
        To find all token sequences matching the supplied patterns on the Doc
        :param doc: spacy container for linguistic annotations.
        :return: dataframe, containing matched occurrences.
        """
        df_format_json = {}
        phrase_matches = self._phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            if "match_id" not in df_format_json:
                df_format_json["match_id"] = []
                df_format_json["match_id"].append(match_id)
            else:
                df_format_json["match_id"].append(match_id)
            if "start" not in df_format_json:
                df_format_json["start"] = []
                df_format_json["start"].append(start)
            else:
                df_format_json["start"].append(start)
            if "end" not in df_format_json:
                df_format_json["end"] = []
                df_format_json["end"].append(end)
            else:
                df_format_json["end"].append(end)
            if "span_text" not in df_format_json:
                df_format_json["span_text"] = []
                df_format_json["span_text"].append(doc[start:end].text)
            else:
                df_format_json["span_text"].append(doc[start:end].text)
        phrase_matches_dataframe = pd.DataFrame.from_dict(df_format_json)
        phrase_matches_dataframe.reindex(phrase_matches_dataframe["match_id"])
        phrase_matches_dataframe.drop(columns=["match_id"], inplace=True)

        return phrase_matches_dataframe

    def get_phrase_matcher(self):
        """
        To get spaCy's phrase matcher class object (used for testing only).
        :return: phrase matcher object
        """
        return (
            PhraseMatcher(self._nlp.vocab, attr=self.attribute)
            if self.attribute
            else PhraseMatcher(self._nlp.vocab)
        )

    def get_phrase_matcher_object(self):
        """
        To get spaCy's matcher class object (used for testing only).
        :return: phrase matcher object
        """
        return self._phrase_matcher

    def add(self, pattern_name, callback, *pattern):
        """
        To add patterns to spaCy's phrase matcher object
        :param pattern_name: str, pattern name
        :param callback: function, callback function to be invoked on matched occurrences.
        :param pattern: list of patterns
        """
        if not self._phrase_matcher:
            self._phrase_matcher = self.get_phrase_matcher()
        self._phrase_matcher.add(pattern_name, callback, *pattern)

    def remove(self, pattern_name):
        """
        To remove pattern from spaCy's matcher object
        :param pattern_name: str, pattern_name
        """
        if self._phrase_matcher:
            self._phrase_matcher.remove(pattern_name)

    def reset(self, change_attribute=None):
        """
        To re-initialize spaCy's phrase matcher object
        :param change_attribute: token attribute to match on
        """
        self.attribute = change_attribute
        self._phrase_matcher = self.get_phrase_matcher()
