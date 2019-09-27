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
    pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
    dframcy_matcher.add("HelloWorld", None, pattern)
    doc = dframcy_matcher.nlp(u"Hello, world! Hello world!")
    matches = dframcy_matcher.get_matcher_object()(doc)
    assert matches[0][0] == 15578876784678163569
    assert matches[0][1] == 0
    assert matches[0][2] == 3
    assert dframcy_matcher.get_nlp().vocab.strings[matches[0][0]] == "HelloWorld"
    assert doc[matches[0][1]:matches[0][2]].text == "Hello, world"

