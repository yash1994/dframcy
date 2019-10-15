# coding: utf-8
from __future__ import unicode_literals

import os
import io
import json
import spacy
import pandas as pd
from wasabi import Printer
from pathlib import Path
from spacy.cli.train import train
from spacy.cli.debug_data import debug_data
from spacy.cli.evaluate import evaluate

from dframcy.dframcy import utils
from dframcy.language_model import LanguageModel

messenger = Printer()


class DframeConverter(object):
    """
    Base class to convert xls/csv training data format to spaCy's JSON format
    (https://spacy.io/api/annotation#json-input).
    """

    def __init__(self,
                 train_path,
                 dev_path,
                 language_model="en_core_web_sm",
                 pipeline="tagger,parser,ner"):
        """
        initialize JSON formatter.
        :param train_path: str, file path or directory path containing multiple training data files
        :param dev_path: str, file path or directory path containing multiple validation data files
        :param language_model: str, language model to be used (default: "em_core_web_sm")
        :param pipeline: str, training pipline (default: "tagger,parser,ner")
        """
        self.train_path = train_path
        self.dev_path = dev_path
        self.dev_path = self.dev_path
        self._nlp = LanguageModel(language_model).get_nlp()
        self.pipeline = pipeline

    def convert(self, data_path, nlp, data_type='training'):
        """
        To convert xls/csv training data to JSON format.
        :param data_path: str, single file or multiple files (directory) to be converted
        :param nlp: nlp pipeline object
        :param data_type: str, type of data "training"/"validation" (default: "training")
        :return: new JSON formatted file path, pipeline inferred from data
        """
        if os.path.exists(data_path):
            if os.path.isfile(data_path):
                if data_path.endswith(".csv"):
                    training_data = pd.read_csv(data_path)
                elif data_path.endswith(".xls") or data_path.endswith(".xlsx"):
                    excel_file = pd.ExcelFile(data_path)
                    training_data = excel_file.parse("Sheet1")
                else:
                    training_data = None
                    messenger.fail("Unknown file format for {} data file:'{}'".format(data_type, data_path), exits=-1)
            elif os.path.isdir(data_path):
                dataframe_list = []
                for file_name in os.listdir(data_path):
                    file_path = os.path.join(data_path, file_name)
                    if file_path.endswith(".csv"):
                        dataframe_list.append(pd.read_csv(file_path))
                    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
                        excel_file = pd.ExcelFile(file_path)
                        dataframe_list.append(excel_file.parse("Sheet1"))
                    else:
                        messenger.warn("Unknown file format for {} data file:{}, skipping".format(data_type, file_path))
                training_data = pd.concat(dataframe_list, join='inner', ignore_index=True)

            training_pipeline = utils.get_training_pipeline_from_column_names(training_data.columns)
            training_pipeline = training_pipeline if len(training_pipeline.split(",")) <= len(self.pipeline.split(",")) \
                else self.pipeline

            json_format = utils.dataframe_to_spacy_training_json_format(
                training_data,
                nlp,
                training_pipeline)
            json_formatted_file_path = data_path.rstrip(".csv").rstrip(".xls").rstrip(".xlsx") + ".json"

            with io.open(json_formatted_file_path, "w") as file:
                json.dump(json_format, file)
            return json_formatted_file_path, training_pipeline
        else:
            messenger.fail("{} file/directory path does not exist".format(data_type), exits=-1)


class DframeTrainer(DframeConverter):
    """
    Wrapper class over spaCy's CLI training from CSV/XLS files.
    """

    def __init__(self,
                 lang,
                 output_path,
                 train_path,
                 dev_path,
                 debug_data_first=True,
                 raw_text=None,
                 base_model=None,
                 pipeline="tagger,parser,ner",
                 vectors=None,
                 n_iter=30,
                 n_early_stopping=None,
                 n_examples=0,
                 use_gpu=-1,
                 version="0.0.0",
                 meta_path=None,
                 init_tok2vec=None,
                 parser_multitasks="",
                 entity_multitasks="",
                 noise_level=0.0,
                 orth_variant_level=0.0,
                 eval_beam_widths="",
                 gold_preproc=False,
                 learn_tokens=False,
                 textcat_multilabel=False,
                 textcat_arch="bow",
                 textcat_positive_label=None,
                 verbose=False):
        """
        for parameters refer to: https://spacy.io/api/cli#train
        """
        super(DframeTrainer, self).__init__(train_path, dev_path, pipeline=pipeline)

        self.lang = lang
        self.output_path = output_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.debug_data_first = debug_data_first
        self.raw_text = raw_text
        self.base_model = base_model
        self.pipeline = pipeline
        self.vectors = vectors
        self.n_iter = n_iter
        self.n_early_stopping = n_early_stopping
        self.n_examples = n_examples
        self.use_gpu = use_gpu
        self.version = version
        self.meta_path = meta_path
        self.init_tok2vec = init_tok2vec
        self.parser_multitasks = parser_multitasks
        self.entity_multitasks = entity_multitasks
        self.noise_level = noise_level
        self.orth_variant_level = orth_variant_level
        self.eval_beam_widths = eval_beam_widths
        self.gold_preproc = gold_preproc
        self.learn_tokens = learn_tokens
        self.textcat_multilabel = textcat_multilabel
        self.textcat_arch = textcat_arch
        self.textcat_positive_label = textcat_positive_label
        self.verbose = verbose

        if self.train_path != self.dev_path:
            self.train_path, training_pipeline = self.convert(self.train_path, self._nlp, "training")
            self.dev_path, evaluation_pipeline = self.convert(self.dev_path, self._nlp, "validation")
        else:
            messenger.warn("Same Training and validation data")
            self.train_path, training_pipeline = self.convert(self.train_path, self._nlp, "training")
            self.dev_path = self.train_path
            evaluation_pipeline = training_pipeline

        self.pipeline = training_pipeline if training_pipeline else self.pipeline
        assert training_pipeline == evaluation_pipeline, messenger.fail("Training({}) and Evaluation({}) pipeline "
                                                                        "does not "
                                                                        "match".format(training_pipeline,
                                                                                       evaluation_pipeline), exits=-1)

    def begin_training(self):
        """
        To initiate training.
        """
        if self.debug_data_first:
            debug_data(
                self.lang,
                Path(self.train_path),
                Path(self.dev_path),
                self.base_model,
                self.pipeline
            )

        train(
            self.lang,
            self.output_path,
            self.train_path,
            self.dev_path,
            self.raw_text,
            self.base_model,
            self.pipeline,
            self.vectors,
            self.n_iter,
            self.n_early_stopping,
            self.n_examples,
            self.use_gpu,
            self.version,
            self.meta_path,
            self.init_tok2vec,
            self.parser_multitasks,
            self.entity_multitasks,
            self.noise_level,
            self.orth_variant_level,
            self.eval_beam_widths,
            self.gold_preproc,
            self.learn_tokens,
            self.textcat_multilabel,
            self.textcat_arch,
            self.textcat_positive_label,
            self.verbose)


class DframeEvaluator(DframeConverter):
    """
    Wrapper class over spaCy's CLI model evaluation from CSV/XLS files.
    """

    def __init__(self,
                 model,
                 data_path,
                 gpu_id=-1,
                 gold_preproc=False,
                 displacy_path=None,
                 displacy_limit=25,
                 return_scores=False):
        """
        for parameters refer to: https://spacy.io/api/cli#evaluate
        """
        super(DframeEvaluator, self).__init__(data_path, data_path)

        self.model = model
        self.data_path = data_path
        self.gpu_id = gpu_id
        self.gold_preproc = gold_preproc
        self.displacy_path = displacy_path
        self.displacy_limit = displacy_limit
        self.return_scores = return_scores

        self.data_path, pipeline = self.convert(self.data_path, self._nlp, "evaluation")

    def begin_evaluation(self):
        """
        To initiate evaluation.
        """
        evaluate(
            self.model,
            self.data_path,
            self.gpu_id,
            self.gold_preproc,
            self.displacy_path,
            self.displacy_limit,
            self.return_scores
        )


class DframeTrainClassifier(object):
    """
    To train text classifier from CSV file.
    """

    def __init__(self,
                 output_path,
                 train_path,
                 dev_path,
                 model=None,
                 n_iter=20,
                 init_tok2vec=None,
                 exclusive_classes=False,
                 architecture="ensemble",
                 train_split=0.8):

        self.output_path = output_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.model = model
        self.n_iter = n_iter
        self.init_tok2vec = init_tok2vec
        self.exclusive_classes = exclusive_classes
        self.architecture = architecture
        self.train_split = train_split

        if self.model is not None:
            self.nlp = spacy.load(self.model)
        else:
            self.nlp = spacy.blank("en")

        if "textcat" not in self.nlp.pipe_names:
            self.textcat = self.nlp.create_pipe(
                "textcat",
                config={"exclusive_classes": self.exclusive_classes, "architecture": self.architecture}
            )
            self.nlp.add_pipe(self.textcat, last=True)
        else:
            self.textcat = self.nlp.get_pipe("textcat")

    def load_dataset(self):

        if self.train_path != self.dev_path:
            training_dataframe = pd.read_csv(self.train_path)
            testing_dataframe = pd.read_csv(self.dev_path)
        else:
            messenger.warn("Same Training and validation data")
            messenger.info("Train and test data will be split in ratio:{}".format(self.train_split))
            dataset = pd.read_csv(self.train_path)
            dataset = dataset.sample(frac=1).reset_index(drop=True)  # shuffle data
            self.train_split = int(dataset.shape[0] * self.train_split)
            training_dataframe, testing_dataframe = dataset.iloc[:, :self.train_split], dataset.iloc[:,
                                                                                        self.train_split:]

        unique_train_labels = pd.unique(training_dataframe["labels"])
        unique_test_labels = pd.unique(testing_dataframe["labels"])

        if unique_train_labels != unique_test_labels:
            additional_labels = set(unique_train_labels) - set(unique_test_labels)
            messenger.warn("Found following additional labels in test set: {}".
                           format(", ".join(list(additional_labels))))
            messenger.info("Removing test instances having additional labels from test set")
            testing_dataframe = testing_dataframe[testing_dataframe.label not in additional_labels]

        for label in unique_train_labels:
            self.textcat.add_label(label)

        training_text, training_label = training_dataframe["text"].tolist(), training_dataframe["labels"].tolist()
        testing_text, testing_label = testing_dataframe["text"].tolist(), testing_dataframe["labels"].tolist()

        training_cats = [{label: 1 if label == _label else 0 for label in unique_train_labels} for _label in
                         training_label]
        testing_cats = [{label: 1 if label == _label else 0 for label in unique_train_labels} for _label in
                        testing_label]
        return training_text, training_cats, testing_text, testing_cats
