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
from spacy.cli.evaluate import evaluate

from dframcy.dframcy import utils
from dframcy.language_model import LanguageModel

messenger = Printer()


class DframeConverter(object):
    def __init__(self,
                 train_path,
                 dev_path,
                 language_model="en_core_web_sm",
                 pipeline="tagger,parser,ner"):

        self.train_path = train_path
        self.dev_path = dev_path
        self.dev_path = self.dev_path
        self._nlp = LanguageModel(language_model).get_nlp()
        self.pipeline = pipeline

    @staticmethod
    def convert(data_path, nlp, data_type='training'):
        if os.path.exists(data_path):
            if magic.from_file(data_path, mime=True) == 'text/plain' and data_path.endswith(".csv"):
                training_data = pd.read_csv(data_path)
            elif "application/vnd" in magic.from_file(data_path, mime=True) and \
                    (data_path.endswith(".xls") or data_path.endswith(".ods")):
                training_data = pd.ExcelFile(data_path)
            else:
                training_data = None
                messenger.fail("Unknown file format for {} data file:'{}'".format(data_type, data_path), exits=-1)
            training_pipeline = utils.get_training_pipeline_from_column_names(training_data.columns)
            json_format = utils.dataframe_to_spacy_training_json_format(
                training_data,
                nlp,
                training_pipeline)
            json_formatted_file_path = data_path.strip(".csv").strip(".xls") + ".json"

            with io.open(json_formatted_file_path, "w") as file:
                json.dump(json_format, file)
            return json_formatted_file_path, training_pipeline
        else:
            messenger.fail("{} file path does not exist".format(data_type), exits=-1)


class DframeTrainer(DframeConverter):
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
    def __init__(self,
                 model,
                 data_path,
                 gpu_id=-1,
                 gold_preproc=False,
                 displacy_path=None,
                 displacy_limit=25,
                 return_scores=False, ):
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
        evaluate(
            self.model,
            self.data_path,
            self.gpu_id,
            self.gold_preproc,
            self.displacy_path,
            self.displacy_limit,
            self.return_scores
        )
