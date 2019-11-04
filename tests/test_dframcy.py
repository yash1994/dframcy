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
data_dir = project_root + '/data'


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
    results = pd.DataFrame({
        "tokens_start": [0, 2, 5, 10, 13, 16],
        "tokens_end": [1, 4, 9, 12, 16, 17],
        "tokens_pos": ["PRON", "AUX", "ADV", "ADP", "PROPN", "PUNCT"],
        "tokens_tag": ["PRP", "VBP", "RB", "IN", "NNP", "."],
        "tokens_dep": ["nsubj", "ROOT", "advmod", "prep", "pobj", "punct"],
        "tokens_head": [1, 1, 1, 1, 3, 1],
        "tokens_label": [None, None, None, None, "GPE", None],
        "tokens_text": ["I", "am", "here", "in", "USA", "."]
    })
    assert_frame_equal(dataframe, results)


def test_all_columns_thoroughly():
    doc = dframcy.nlp("Machine learning is an application of artificial intelligence (AI) that provides systems the "
                      "ability to automatically learn and improve from experience without being explicitly "
                      "programmed. Machine learning focuses on the development of computer programs that can access "
                      "data and use it learn for themselves.")
    dataframe = dframcy.to_dataframe(doc, ["id", "start", "end", "pos", "tag", "dep", "head", "text", "lemma", "lower",
                                           "shape", "prefix", "suffix", "is_alpha", "is_ascii", "is_digit", "is_lower",
                                           "is_upper", "is_title", "is_punct", "is_left_punct", "is_right_punct",
                                           "is_space", "is_bracket", "is_quote", "is_currency", "like_url", "like_num",
                                           "like_email", "is_oov", "is_stop", "ancestors", "conjuncts", "children",
                                           "lefts", "rights", "n_lefts", "n_rights", "is_sent_start", "has_vector",
                                           "ent_start", "ent_end", "ent_label"])
    with open(os.path.join(data_dir, "all_columns_results.json"), "r") as file:
        df_json = json.load(file)
    results = pd.DataFrame(df_json)
    assert_frame_equal(dataframe, results)


def test_entity_rule_dataframe():
    dframcy_test_ent = DframCy(spacy.load("en_core_web_sm"))
    patterns = [{"label": "ORG", "pattern": "MyCorp Inc."}]
    dframcy_test_ent.add_entity_ruler(patterns)
    doc = dframcy_test_ent.nlp("MyCorp Inc. is a company in the U.S.")
    _, entity_frame = dframcy_test_ent.to_dataframe(doc, separate_entity_dframe=True)
    results = pd.DataFrame({
        "ent_text": ["MyCorp Inc.", "U.S."],
        "ent_label": ["ORG", "GPE"]
    })
    assert_frame_equal(entity_frame, results)


def test_sentence_without_named_entities():
    doc = dframcy.nlp("Autonomous cars shift insurance liability toward manufacturers")
    dataframe = dframcy.to_dataframe(doc, ["pos", "tag", "ent_label"])

    assert "tokens_label" not in dataframe.columns
