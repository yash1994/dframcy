# coding: utf-8
from __future__ import unicode_literals

import io
import json
import pytest
import os
import shutil
import operator
import pandas as pd
from dframcy.trainer import (
    DframeConverter,
    DframeTrainer,
    DframeEvaluator,
    DframeTrainClassifier,
)


@pytest.mark.parametrize(
    "input_csv_file, output_json_file",
    [("data/training_data_format.csv", "data/training_data_converted.json")],
)
def test_cli_format_converter_full_pipeline(input_csv_file, output_json_file):
    dframe_converter = DframeConverter(
        train_path=input_csv_file, dev_path=input_csv_file
    )
    json_formatted_file_path, pipeline = dframe_converter.convert(
        input_csv_file, dframe_converter._nlp
    )

    with io.open(json_formatted_file_path, "r") as format_file:
        json_formatted_training_data = json.load(format_file)
    os.remove(json_formatted_file_path)

    assert len(json_formatted_training_data) == 6
    assert (
        len(json_formatted_training_data[0]["paragraphs"][0]["sentences"][0]["tokens"])
        == 8
    )
    assert json_formatted_training_data[0]["paragraphs"][0]["sentences"][0]["tokens"][
        0
    ] == {
        "id": 0,
        "orth": "Uber",
        "tag": "NNP",
        "head": 1,
        "dep": "nsubj",
        "ner": "U-ORG",
    }


@pytest.mark.parametrize("input_csv_file", ["data/training_data_format.csv"])
def test_cli_format_converter_only_tagger(input_csv_file):
    dframe_converter = DframeConverter(
        train_path=input_csv_file, dev_path=input_csv_file, pipeline="tagger"
    )
    json_formatted_file_path, pipeline = dframe_converter.convert(
        input_csv_file, dframe_converter._nlp
    )

    with io.open(json_formatted_file_path, "r") as format_file:
        json_formatted_training_data = json.load(format_file)
    os.remove(json_formatted_file_path)

    assert len(json_formatted_training_data) == 6
    assert (
        len(json_formatted_training_data[0]["paragraphs"][0]["sentences"][0]["tokens"])
        == 8
    )
    assert json_formatted_training_data[0]["paragraphs"][0]["sentences"][0]["tokens"][
        1
    ] == {"id": 1, "orth": "blew", "tag": "VBD"}
    assert (
        "dep"
        not in json_formatted_training_data[0]["paragraphs"][0]["sentences"][0][
            "tokens"
        ][0]
    )


@pytest.mark.parametrize("input_csv_file", ["data/training_data_format.csv"])
def test_cli_format_converter_only_parser(input_csv_file):
    dframe_converter = DframeConverter(
        train_path=input_csv_file, dev_path=input_csv_file, pipeline="parser"
    )
    json_formatted_file_path, pipeline = dframe_converter.convert(
        input_csv_file, dframe_converter._nlp
    )

    with io.open(json_formatted_file_path, "r") as format_file:
        json_formatted_training_data = json.load(format_file)
    os.remove(json_formatted_file_path)

    assert len(json_formatted_training_data) == 6
    assert (
        len(json_formatted_training_data[0]["paragraphs"][0]["sentences"][0]["tokens"])
        == 8
    )
    assert json_formatted_training_data[-1]["paragraphs"][0]["sentences"][0]["tokens"][
        0
    ] == {"id": 0, "orth": "look", "tag": "VB", "head": 0, "dep": "ROOT"}
    assert (
        "ner"
        not in json_formatted_training_data[0]["paragraphs"][0]["sentences"][0][
            "tokens"
        ][0]
    )


@pytest.mark.parametrize("input_csv_file", ["data/training_data_format.csv"])
def test_cli_format_converter_only_ner(input_csv_file):
    dframe_converter = DframeConverter(
        train_path=input_csv_file, dev_path=input_csv_file, pipeline="ner"
    )
    json_formatted_file_path, pipeline = dframe_converter.convert(
        input_csv_file, dframe_converter._nlp
    )

    with io.open(json_formatted_file_path, "r") as format_file:
        json_formatted_training_data = json.load(format_file)
    os.remove(json_formatted_file_path)

    assert len(json_formatted_training_data) == 6
    assert (
        len(json_formatted_training_data[0]["paragraphs"][0]["sentences"][0]["tokens"])
        == 8
    )
    assert (
        json_formatted_training_data[0]["paragraphs"][0]["sentences"][0]["tokens"][0][
            "ner"
        ]
        == "U-ORG"
    )


@pytest.mark.parametrize("input_csv_file", ["data/training_data_format.csv"])
def test_cli_training(input_csv_file):
    dframe_trainer = DframeTrainer(
        "en", "/tmp/", input_csv_file, input_csv_file, debug_data_first=False
    )
    assert dframe_trainer.pipeline == "tagger,parser,ner"


@pytest.mark.parametrize("input_csv_file", ["data/training_data_format.csv"])
def test_cli_evaluation(input_csv_file):
    dframe_evaluator = DframeEvaluator("en_core_web_sm", input_csv_file)
    assert dframe_evaluator.pipeline == "tagger,parser,ner"


@pytest.mark.parametrize(
    "input_csv_file, dev_xls_file",
    [("data/training_data_format.csv", "data/training_data_format_xls.xls")],
)
def test_ods_training_file_format(input_csv_file, dev_xls_file):
    dframe_trainer = DframeTrainer("en", "/tmp/", input_csv_file, dev_xls_file)
    os.remove(dframe_trainer.train_path)
    os.remove(dframe_trainer.dev_path)
    assert dframe_trainer.pipeline == "tagger,parser,ner"


@pytest.mark.parametrize(
    "input_csv_file, dev_xls_file",
    [("data/training_data_format.csv", "data/training_data_format_xls.xls")],
)
def test_training_from_directory(input_csv_file, dev_xls_file):
    if not os.path.exists("/tmp/dframcy_test"):
        os.mkdir("/tmp/dframcy_test/")

    file_name, file_extension = os.path.splitext(input_csv_file)
    file_name = str(file_name.split("/")[-1])
    for i in range(10):
        shutil.copy(
            input_csv_file,
            "/tmp/dframcy_test/" + file_name + "_" + str(i) + file_extension,
        )

    file_name, file_extension = os.path.splitext(dev_xls_file)
    file_name = str(file_name.split("/")[-1])
    for i in range(10):
        shutil.copy(
            dev_xls_file,
            "/tmp/dframcy_test/" + file_name + "_" + str(i) + file_extension,
        )

    dframe_trainer = DframeTrainer(
        "en", "/tmp/", "/tmp/dframcy_test/", "/tmp/dframcy_test/"
    )
    shutil.rmtree("/tmp/dframcy_test/")
    assert dframe_trainer.pipeline == "tagger,parser,ner"


@pytest.mark.xfail  # test case supposed to fail due to very low number of training instances
@pytest.mark.parametrize("input_csv_file", ["data/training_data_format.csv"])
def test_data_debugging(input_csv_file):
    dframe_trainer = DframeTrainer(
        "en", "/tmp/", input_csv_file, input_csv_file, debug_data_first=True
    )
    dframe_trainer.begin_training()


@pytest.mark.parametrize("input_csv_file", ["data/textcat_training.csv"])
def test_cli_textcat_training(input_csv_file):
    dframe_textcat_classifier = DframeTrainClassifier(
        "/tmp/", input_csv_file, input_csv_file, n_iter=1
    )
    test_text = "This movie sucked"
    dframe_textcat_classifier.begin_training()
    doc = dframe_textcat_classifier.nlp(test_text)
    assert max(doc.cats.items(), key=operator.itemgetter(1))[0] in ["NEG", "POS"]


@pytest.mark.parametrize("input_csv_file", ["data/textcat_training.csv"])
def test_cli_textcat_training_multiclass(input_csv_file):
    if not os.path.exists("/tmp/dframcy_test"):
        os.mkdir("/tmp/dframcy_test/")
    training_csv = pd.read_csv(input_csv_file)
    random_update = training_csv.sample(frac=0.15)
    random_update["labels"] = "NEUTRAL"
    training_csv.update(random_update)
    training_csv.to_csv("/tmp/dframcy_test/textcat_multiclass_training.csv")
    dframe_textcat_classifier = DframeTrainClassifier(
        "/tmp/",
        "/tmp/dframcy_test/textcat_multiclass_training.csv",
        "/tmp/dframcy_test/textcat_multiclass_training.csv",
        n_iter=1,
    )
    test_text = "This movie sucked"
    dframe_textcat_classifier.begin_training()
    doc = dframe_textcat_classifier.nlp(test_text)
    shutil.rmtree("/tmp/dframcy_test/")
    assert max(doc.cats.items(), key=operator.itemgetter(1))[0] in [
        "NEG",
        "POS",
        "NEUTRAL",
    ]
