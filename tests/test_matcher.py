# coding: utf-8
from __future__ import unicode_literals

import os
import pytest
import json
import pandas as pd
from pandas.util.testing import assert_frame_equal
from dframcy.matcher import DframCyMatcher


dframcy_matcher = DframCyMatcher("en_core_web_sm")


def test_matcher():
    dframcy_matcher.reset()
    pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
    dframcy_matcher.add("HelloWorld", None, pattern)
    doc = dframcy_matcher.nlp(u"Hello, world! Hello world!")
    matches = dframcy_matcher.get_matcher_object()(doc)
    assert matches[0][0] == 15578876784678163569
    assert matches[0][1] == 0
    assert matches[0][2] == 3
    assert dframcy_matcher.get_nlp().vocab.strings[matches[0][0]] == "HelloWorld"
    assert doc[matches[0][1]:matches[0][2]].text == "Hello, world"


def test_matcher_dataframe():
    dframcy_matcher.reset()
    pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
    dframcy_matcher.add("HelloWorld", None, pattern)
    doc = dframcy_matcher.nlp(u"Hello, world! Hello world!")
    matches_dataframe = dframcy_matcher(doc)
    results = pd.DataFrame({
        "start": [0],
        "end": [3],
        "string_id": ["HelloWorld"],
        "span_text": ["Hello, world"]
    })
    assert_frame_equal(matches_dataframe, results)


def test_matcher_dataframe_multiple_patterns():
    dframcy_matcher.reset()
    dframcy_matcher.add("Hello_World", None, [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}],
                        [{"LOWER": "hello"}, {"LOWER": "world"}])
    doc = dframcy_matcher.nlp(u"Hello, world! Hello world!")
    matches_dataframe = dframcy_matcher(doc)
    results = pd.DataFrame({
        "start": [0, 4],
        "end": [3, 6],
        "string_id": ["Hello_World", "Hello_World"],
        "span_text": ["Hello, world", "Hello world"]
    })
    assert_frame_equal(matches_dataframe, results)
