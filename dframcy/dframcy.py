# coding: utf-8
from __future__ import unicode_literals

import warnings
import pandas as pd
from cytoolz import merge_with

from dframcy import utils
from dframcy.language_model import LanguageModel

# reference:
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='numpy.ufunc size changed')


class DframCy(LanguageModel):

    def __init__(self, nlp_model):
        super(DframCy, self).__init__(nlp_model)

    @staticmethod
    def get_additional_token_attributes(attribute, doc):
        _name = attribute[0]
        _type = attribute[2]
        _is_nested = attribute[1]

        if _type == 0 or _type == 2:
            values = [getattr(token, _name) for token in doc]
        else:
            values = [getattr(token, _name)() for token in doc]

        if _is_nested:
            if isinstance(list(values[0]), list):
                flattened_values = []
                for v in values:
                    flattened_values.append([i.text for i in v])
            else:
                flattened_values = [v.text for v in values]
            values = flattened_values
        return values

    def to_dataframe(self, doc, columns=None):

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

        return tokens_dataframe
