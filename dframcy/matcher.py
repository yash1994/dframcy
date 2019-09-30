# coding: utf-8
from __future__ import unicode_literals

import pandas as pd
from dframcy.language_model import LanguageModel
from spacy.matcher import Matcher, PhraseMatcher


class DframCyMatcher(LanguageModel):
    def __init__(self, nlp_model):
        super(DframCyMatcher, self).__init__(nlp_model)
        self._matcher = None

    def __call__(self, doc):
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
        return self._matcher

    def get_matcher(self):
        if not self._nlp:
            self._nlp = self.create_nlp_pipeline()
        return Matcher(self._nlp.vocab)

    def add(self, pattern_name, callback, *pattern):
        if not self._matcher:
            self._matcher = self.get_matcher()
        self._matcher.add(pattern_name, callback, *pattern)

    def reset(self):
        self._matcher = self.get_matcher()


class DframCyPhraseMatcher(LanguageModel):
    def __init__(self, nlp_model, attr=None):
        super(DframCyPhraseMatcher, self).__init__(nlp_model)
        self._phrase_matcher = None
        self.attribute = attr

    def __call__(self, doc):
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
        if not self._nlp:
            self._nlp = self.create_nlp_pipeline()
        return PhraseMatcher(self._nlp.vocab, attr=self.attribute) if self.attribute else PhraseMatcher(self._nlp.vocab)

    def get_phrase_matcher_object(self):
        return self._phrase_matcher

    def add(self, pattern_name, callback, *pattern):
        if not self._phrase_matcher:
            self._phrase_matcher = self.get_phrase_matcher()
        self._phrase_matcher.add(pattern_name, callback, *pattern)

    def reset(self, change_attribute=None):
        self.attribute = change_attribute
        self._phrase_matcher = self.get_phrase_matcher()
