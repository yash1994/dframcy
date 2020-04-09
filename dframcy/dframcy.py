# coding: utf-8
from __future__ import unicode_literals

import pandas as pd
from spacy.pipeline import EntityRuler

from dframcy import utils


class DframCy(object):
    """
    Dataframe integration with spaCy's linguistic annotations.
    """

    def __init__(self, nlp_pipeline):
        """
        :param nlp_pipeline: nlp pipeline to be used (i.e. language model).
        """
        self._nlp = nlp_pipeline

    @property
    def nlp(self):
        """
        To get texted nlped
        :return: Spacy's Doc object
        """
        return self._nlp

    @staticmethod
    def get_token_attribute_value(token, attribute_name, _type):
        """
        To get value of specific attribute of spacy's Token class
        :param token: token object of class Token
        :param attribute_name: name attribute for which value is required
        :param _type: type of class attribute (property, attribute)
        :retrun: attribute value
        """
        if _type == "attribute" or _type == "int_format_attribute":
            value = getattr(token, attribute_name)
            if attribute_name in ["head", "left_edge", "right_edge"]:
                return value.text
            else:
                return value
        elif _type == "property":
            value = getattr(token, attribute_name)
            if attribute_name in ["n_lefts", "n_rights", "has_vector", "is_sent_start"]:
                return value
            else:
                return ", ".join([v.text for v in value])
        elif _type == "additional_attribute":
            if attribute_name == "id":
                return getattr(token, "i")
            elif attribute_name == "start":
                return getattr(token, "idx")
            elif attribute_name == "end":
                return getattr(token, "idx") + len(token)

    def get_token_attribute_dict(self, doc, consistent_columns):
        """
        To get attribute dictionary for sequence of Token object in Doc
        :param doc: Doc object
        :param consistent_columns: name attributes required with its type
        :return: python dictionary containing attributes names as keys
                and list of all token values as value.
        """
        token_attribute_dictionary = {}
        for token in doc:
            for column_name in consistent_columns:
                if column_name[0] in token_attribute_dictionary:
                    token_attribute_dictionary[column_name[0]].append(
                        self.get_token_attribute_value(
                            token, column_name[0], column_name[1]
                        )
                    )
                else:
                    token_attribute_dictionary[column_name[0]] = []
                    token_attribute_dictionary[column_name[0]].append(
                        self.get_token_attribute_value(
                            token, column_name[0], column_name[1]
                        )
                    )
        return token_attribute_dictionary

    @staticmethod
    def get_named_entity_dict(doc):
        """
        To get named entities from NLP processed text
        :param doc: spacy container for linguistic annotations.
        :return: dictionary containing entity_text and entity_label
        """
        entity_details_dict = {"ent_text": [], "ent_label": []}
        for ent in doc.ents:
            entity_details_dict["ent_text"].append(ent.text)
            entity_details_dict["ent_label"].append(ent.label_)
        return entity_details_dict

    def to_dataframe(self, doc, columns=None, separate_entity_dframe=False):
        """
        Convert Linguistic annotations for text into pandas dataframe
        :param doc: spacy container for linguistic annotations.
        :param columns: list of str, name of columns to be included in dataframe (default: ["tokens.id", "tokens.text",
        "tokens.start", "tokens.end", "tokens.pos", "tokens.tag", "tokens.dep", "tokens.head", "ents.start",
        "ents.end", "ents.label"])
        :param separate_entity_dframe: bool, for separate entity dataframe (default: False)
        :return: dataframe, dataframe containing linguistic annotations
        """
        if columns is None:
            columns = utils.get_default_columns()

        if "id" not in columns:
            columns = ["id"] + columns

        consistent_columns = utils.check_columns_consistency(columns)

        token_attribute_dictionary = self.get_token_attribute_dict(
            doc, consistent_columns
        )
        tokens_dataframe = pd.DataFrame.from_dict(token_attribute_dictionary)

        new_column_names_map = {i: "token_" + i for i in tokens_dataframe.columns}

        tokens_dataframe.rename(columns=new_column_names_map, inplace=True)

        tokens_dataframe.reindex(tokens_dataframe["token_id"])

        tokens_dataframe.drop(columns=["token_id"], inplace=True)

        if not doc.ents and "token_ent_type_" in tokens_dataframe.columns:
            tokens_dataframe.drop(columns=["token_ent_type_"], inplace=True)

        if separate_entity_dframe:
            entity_dict = self.get_named_entity_dict(doc)
            entity_dataframe = pd.DataFrame.from_dict(entity_dict)

        return (
            tokens_dataframe
            if not separate_entity_dframe
            else (tokens_dataframe, entity_dataframe)
        )

    def add_entity_ruler(self, patterns):
        """
        To add entity ruler in nlp pipeline
        official doc: https://spacy.io/api/entityruler
        :param patterns: list or list of lists of token/phrase based patterns
        """
        ruler = EntityRuler(self._nlp)
        ruler.add_patterns(patterns)
        self._nlp.add_pipe(ruler)
