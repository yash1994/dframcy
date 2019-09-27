# coding: utf-8
from __future__ import unicode_literals

import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
from dframcy.dframcy import DframCy
dframcy = DframCy("en_core_web_sm")


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
        "tokens_pos": ["PRON", "VERB", "ADV", "ADP", "PROPN", "PUNCT"],
        "tokens_tag": ["PRP", "VBP", "RB", "IN", "NNP", "."],
        "tokens_dep": ["nsubj", "ROOT", "advmod", "prep", "pobj", "punct"],
        "tokens_head": [1, 1, 1, 1, 3, 1],
        "tokens_label": [None, None, None, None, "GPE", None],
        "tokens_text": ["I", "am", "here", "in", "USA", "."]
    })
    assert_frame_equal(dataframe, results)
