# coding: utf-8
from __future__ import unicode_literals

import os
import pytest
import json
import spacy
import pandas as pd
from pandas.util.testing import assert_frame_equal
from io import open
from dframcy.dframcy import DframCy

dframcy = DframCy(spacy.load("en_core_web_sm"))

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = "/" + "/".join(current_dir.split("/")[1:-1])
data_dir = project_root + "/data"


@pytest.mark.parametrize("text", ["I am here in USA."])
def test_nlp_pipeline(text):
    doc = dframcy.nlp(text)
    assert doc[0].text == "I"
    assert doc[0].tag_ == "PRP"
    assert doc[1].lemma_ == "be"


@pytest.mark.parametrize("text", ["I am here in USA."])
def test_default_columns(text):
    doc = dframcy.nlp(text)
    dataframe = dframcy.to_dataframe(doc)
    results = pd.DataFrame(
        {
            "token_text": ["I", "am", "here", "in", "USA", "."],
            "token_start": [0, 2, 5, 10, 13, 16],
            "token_end": [1, 4, 9, 12, 16, 17],
            "token_pos_": ["PRON", "AUX", "ADV", "ADP", "PROPN", "PUNCT"],
            "token_tag_": ["PRP", "VBP", "RB", "IN", "NNP", "."],
            "token_dep_": ["nsubj", "ROOT", "advmod", "prep", "pobj", "punct"],
            "token_head": ["am", "am", "am", "am", "in", "am"],
            "token_ent_type_": ["", "", "", "", "GPE", ""],
        }
    )
    assert_frame_equal(dataframe, results)


@pytest.mark.parametrize("text", ["bright red apples on the tree"])
def test_unknown_column_value(text):
    doc = dframcy.nlp(text)
    dataframe = dframcy.to_dataframe(doc, columns=["id", "start", "end", "apple"])
    results = pd.DataFrame(
        {"token_start": [0, 7, 11, 18, 21, 25], "token_end": [6, 10, 17, 20, 24, 29]}
    )
    assert_frame_equal(dataframe, results)


@pytest.mark.parametrize("text", ["I have an apple"])
def test_custom_attribute(text):
    from spacy.tokens import Token

    fruit_getter = lambda token: token.text in ("apple", "pear", "banana")
    Token.set_extension("is_fruit", getter=fruit_getter)
    doc = dframcy.nlp(text)
    dataframe = dframcy.to_dataframe(
        doc, columns=["id", "start", "end"], custom_attributes=["is_fruit"]
    )
    results = pd.DataFrame(
        {
            "token_start": [0, 2, 7, 10],
            "token_end": [1, 6, 9, 15],
            "token_is_fruit": [False, False, False, True],
        }
    )
    assert_frame_equal(dataframe, results)


def test_all_columns_thoroughly():
    doc = dframcy.nlp(
        "Machine learning is an application of artificial intelligence (AI) that provides systems the "
        "ability to automatically learn and improve from experience without being explicitly "
        "programmed. Machine learning focuses on the development of computer programs that can access "
        "data and use it learn for themselves."
    )
    dataframe = dframcy.to_dataframe(
        doc,
        [
            "id",
            "end",
            "pos",
            "tag",
            "dep",
            "text",
            "head",
            "pos_",
            "tag_",
            "dep_",
            "orth",
            "norm",
            "lang",
            "orth_",
            "norm_",
            "lang_",
            "lefts",
            "start",
            "lower",
            "shape",
            "lemma_",
            "lower_",
            "shape_",
            "is_oov",
            "rights",
            "ent_id",
            "prefix",
            "suffix",
            "ent_id_",
            "prefix_",
            "suffix_",
            "is_stop",
            "n_lefts",
            "subtree",
            "ent_iob",
            "ent_iob_",
            "is_alpha",
            "is_ascii",
            "is_digit",
            "is_lower",
            "is_upper",
            "is_title",
            "is_punct",
            "is_space",
            "is_quote",
            "like_url",
            "like_num",
            "children",
            "n_rights",
            "ent_type",
            "left_edge",
            "ent_type_",
            "ancestors",
            "conjuncts",
            "right_edge",
            "ent_kb_id_",
            "is_bracket",
            "like_email",
            "has_vector",
            "is_currency",
            "is_left_punct",
            "is_sent_start",
            "is_right_punct",
        ],
    )

    assert dataframe.shape == (48, 62)
    assert dataframe["token_ent_type_"][9] == "ORG"
    assert dataframe["token_ancestors"][0] == "learning, is"
    assert (dataframe.token_is_lower).sum() == 41
    assert (~dataframe.token_is_lower).sum() == 7


def test_entity_rule_dataframe():
    dframcy_test_ent = DframCy(spacy.load("en_core_web_sm"))
    patterns = [{"label": "ORG", "pattern": "MyCorp Inc."}]
    dframcy_test_ent.add_entity_ruler(patterns)
    doc = dframcy_test_ent.nlp("MyCorp Inc. is a company in the U.S.")
    _, entity_frame = dframcy_test_ent.to_dataframe(doc, separate_entity_dframe=True)
    results = pd.DataFrame(
        {"ent_text": ["MyCorp Inc.", "U.S."], "ent_label": ["ORG", "GPE"]}
    )
    assert_frame_equal(entity_frame, results)


def test_sentence_without_named_entities():
    doc = dframcy.nlp("Autonomous cars shift insurance liability toward manufacturers.")
    dataframe = dframcy.to_dataframe(doc, ["pos_", "tag_", "ent_type_"])

    assert "token_ent_type_" not in dataframe.columns
