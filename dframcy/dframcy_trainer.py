# coding: utf-8
from __future__ import unicode_literals

import os
import io
import magic
import json
import pandas as pd
from wasabi import Printer
from pathlib import Path
from spacy.cli.train import train
from spacy.cli.debug_data import debug_data

from dframcy.dframcy import utils
from dframcy.language_model import LanguageModel
messenger = Printer()


class DframeTrainer(object):
    def __init__(self,
                 lang,
                 output_path,
                 train_path,
                 dev_path,
                 debug_data_first=True,
                 language_model="en_core_web_sm",
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
        self.lang = lang
        self.output_path = output_path
        self.train_path = train_path
        self.dev_path = dev_path
        self.debug_data_first = debug_data_first
        self.language_model = language_model
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
        self._nlp = LanguageModel(self.language_model).get_nlp()
        self.dataframe_shape = None

    def convert(self):
        if os.path.exists(self.train_path):
            if magic.from_file(self.train_path, mime=True) == 'text/plain' and self.train_path.endswith(".csv"):
                training_data = pd.read_csv(self.train_path)
            elif "application/vnd" in magic.from_file(self.train_path, mime=True) and \
                    (self.train_path.endswith(".xls") or self.train_path.endswith(".ods")):
                training_data = pd.ExcelFile(self.train_path)
            else:
                training_data = None
                messenger.fail("Unknown file format for input file:'{}'".format(self.train_path), exits=-1)

            self.dataframe_shape = training_data.shape
            training_pipeline = utils.get_training_pipeline_from_column_names(training_data.columns)
            self.pipeline = training_pipeline if training_pipeline is not None else self.pipeline

            json_format = utils.dataframe_to_spacy_training_json_format(
                training_data,
                self._nlp,
                self.pipeline)

            json_training_file_path = self.train_path.strip(".csv").strip(".xls") + ".json"

            with io.open(json_training_file_path, "w") as file:
                json.dump(json_format, file)

            self.train_path = json_training_file_path
            self.dev_path = json_training_file_path
        else:
            messenger.fail("Training file path does not exist.", exits=-1)

    def begin_training(self):

        self.convert()
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
