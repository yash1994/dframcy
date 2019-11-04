# coding: utf-8
from __future__ import unicode_literals

import pandas as pd
from spacy.pipeline import EntityRuler
from cytoolz import merge_with

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
        return self._nlp

    @staticmethod
    def get_additional_token_attributes(attribute, doc):
        """
        To get additional (token information other than doc.json()) token class attributes.
        official doc: https://spacy.io/api/token
        type: 0 for class attribute, 1 for class method, 2 for class property
        :param attribute: tuple(name, type, is_nested) i.e. ("is_punct", 0, False)
        :param doc: spacy container for linguistic annotations.
        :return: list of or list of lists of attribute values
        """
        _name = attribute[0]
        _type = attribute[2]
        _is_nested = attribute[1]

        values = [getattr(token, _name) for token in doc]

        if _is_nested:
            if isinstance(list(values[0]), list):
                flattened_values = []
                for v in values:
                    flattened_values.append([i.text for i in v])
            else:
                flattened_values = [v.text for v in values]
            values = flattened_values
        return values

    @staticmethod
    def get_named_entity_details(doc):
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
        if not columns:
            columns = utils.get_default_columns()
            additional_attributes = False
        else:
            columns, additional_attributes = utils.map_user_columns_names_with_default(columns)

        if "tokens.id" not in columns:
            columns.append("tokens.id")

        json_doc = doc.to_json()

        if "ent" in ", ".join(columns):
            json_doc = utils.merge_entity_details(json_doc)

        if not json_doc["ents"]:
            columns = [cn for cn in columns if "ents" not in cn]

        merged_tokens_dict = merge_with(list, *json_doc["tokens"])
        merged_tokens_dict["text"] = [json_doc["text"][token["start"]:token["end"]] for token in json_doc["tokens"]]

        if additional_attributes:
            for column in columns:
                if column.split(".")[-1] not in merged_tokens_dict:
                    merged_tokens_dict[column.split(".")[-1]] = self.get_additional_token_attributes(
                        utils.additional_attributes_map(column), doc
                    )

        if "tokens.start" in columns and "tokens.end" in columns:
            if "ents.start" in columns:
                columns.remove("ents.start")
            if "ents.end" in columns:
                columns.remove("ents.end")

        columns_filtered_token_dict = {}

        for key in merged_tokens_dict.keys():
            if key in set([i.split(".")[-1] for i in columns]):
                columns_filtered_token_dict["tokens_" + key] = merged_tokens_dict[key]

        tokens_dataframe = pd.DataFrame.from_dict(columns_filtered_token_dict)

        tokens_dataframe.reindex(tokens_dataframe["tokens_id"])

        tokens_dataframe.drop(columns=["tokens_id"], inplace=True)

        if separate_entity_dframe:
            entity_dict = self.get_named_entity_details(doc)
            entity_dataframe = pd.DataFrame.from_dict(entity_dict)
        else:
            entity_dataframe = None

        return tokens_dataframe if not separate_entity_dframe else (tokens_dataframe, entity_dataframe)

    def add_entity_ruler(self, patterns):
        """
        To add entity ruler in nlp pipeline
        official doc: https://spacy.io/api/entityruler
        :param patterns: list or list of lists of token/phrase based patterns
        """
        ruler = EntityRuler(self._nlp)
        ruler.add_patterns(patterns)
        self._nlp.add_pipe(ruler)
