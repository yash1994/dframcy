# coding: utf-8
from __future__ import unicode_literals

import json
import click
from pathlib import Path
from wasabi import Printer
from io import open
from .dframcy import DframCy
from .utils import get_default_columns
from .trainer import DframeTrainer, DframeEvaluator

messenger = Printer()
DEFAULT_COLUMNS = ",".join(get_default_columns())


@click.group()
def main():
    pass


@main.command()
@click.option("--input_file", "-i", required=True, type=Path, help="Input text file path.")
@click.option("--output_file", "-o", required=True, type=Path, help="Output file path/name")
@click.option("--convert_type", "-t", default="csv", show_default=True, type=str, help="Output file format (JSON/CSV)")
@click.option("--language_model", "-l", default="en_core_web_sm", show_default=True, type=str, help="Language model "
                                                                                                    "to be used.")
@click.option("--columns", "-c", default=DEFAULT_COLUMNS, show_default=True, type=str, help="Annotations to be "
                                                                                            "included in dataframe.")
@click.option("--separate_entity_frame", "-s", default=False, show_default=True, type=bool, help="Save separate "
                                                                                                 "entity dataframe.")
def convert(input_file, output_file, convert_type, language_model, columns, separate_entity_frame):
    if output_file.is_dir():
        output_file = output_file.joinpath(input_file.stem + "." + str(convert_type))
    if input_file.exists():
        with open(input_file, "r") as infile:
            text = infile.read()
            dframcy = DframCy(language_model)
            doc = dframcy.nlp(text)
            if columns == DEFAULT_COLUMNS:
                annotation_dataframe = dframcy.to_dataframe(doc, separate_entity_dframe=separate_entity_frame)
            else:
                annotation_dataframe = dframcy.to_dataframe(doc, columns=columns.split(", "),
                                                            separate_entity_dframe=separate_entity_frame)
            if separate_entity_frame:
                token_annotation_dataframe, entity_dataframe = annotation_dataframe
            else:
                token_annotation_dataframe = annotation_dataframe
                entity_dataframe = None

            if convert_type == "csv":
                token_annotation_dataframe.to_csv(output_file)
                if separate_entity_frame:
                    entity_output_file = Path(str(output_file).strip(".csv") + "_entity.csv")
                    entity_dataframe.to_csv(entity_output_file)
            elif convert_type == "json":
                annotation_json = token_annotation_dataframe.to_json(orient="columns")
                with open(output_file, "w") as outfile:
                    json.dump(annotation_json, outfile)
                if separate_entity_frame:
                    entity_output_file = Path(str(output_file).strip(".json") + "_entity.json")
                    with open(entity_output_file, "w") as ent_outfile:
                        json.dump(entity_dataframe, ent_outfile)
            else:
                messenger.fail("Unknown output file format '{}'".format(convert_type), exits=-1)
    else:
        messenger.fail("input path {} does not exist".format(input_file), exits=-1)


@main.command()
@click.option("--lang", "-l", required=True, type=str, help="Model language.")
@click.option("--output_path", "-o", required=True, type=str, help="Output directory to store mode in.")
@click.option("--train_path", "-t", required=True, type=str, help="Path of CSV containing training data.")
@click.option("--dev_path", "-d", required=True, type=str, help="Path to CSV containing validation data")
@click.option("--debug_data_first", "-debug", default=True, show_default=True, type=bool, help="Run spaCy's training "
                                                                                               "data debugger before "
                                                                                               "training")
@click.option("--raw_text", "-rt", default=None, show_default=True, type=str, help="Path to jsonl file with unlabelled "
                                                                                   "text documents.")
@click.option("--base_model", "-b", default=None, show_default=True, type=str, help="Name of model to update")
@click.option("--pipeline", "-p", default="tagger,parser,ner", show_default=True, type=str, help="Comma-separated "
                                                                                                 "names of pipeline "
                                                                                                 "components")
@click.option("--vectors", "-v", default=None, show_default=True, type=str, help="Model to load vectors from")
@click.option("--n_iter", "-n", default=30, show_default=True, type=int, help="Number of iterations")
@click.option("--n_early_stopping", "-ne", default=None, show_default=True, type=int, help="Maximum number of "
                                                                                           "training epochs without "
                                                                                           "dev accuracy improvement")
@click.option("--n_examples", "-ns", default=0, show_default=True, type=int, help="Number of examples")
@click.option("--use_gpu", "-g", default=-1, show_default=True, type=int, help="Use GPU")
@click.option("--version", "-v", default="0.0.0", show_default=True, type=str, help="Model version")
@click.option("--meta_path", "-m", default=None, show_default=True, type=Path, help="Optional path to meta.json to "
                                                                                    "use as base.")
@click.option("--init_tok2vec", "-t2v", default=None, show_default=True, type=str, help="Path to pretrained weights "
                                                                                        "for the token-to-vector "
                                                                                        "parts of the models")
@click.option("--parser_multitasks", "-pm", default="", show_default=True, type=str, help="Side objectives for parser "
                                                                                          "CNN, e.g. 'dep' or 'dep,"
                                                                                          "tag'")
@click.option("--entity_multitasks", "-em", default="", show_default=True, type=str, help="Side objectives for NER "
                                                                                          "CNN, e.g. 'dep' or 'dep,"
                                                                                          "tag'")
@click.option("--noise_level", "-n", default=0.0, show_default=True, type=float, help="Amount of corruption for data "
                                                                                      "augmentation")
@click.option("--orth_variant_level", "-vl", default=0.0, show_default=True, type=float, help="Amount of orthography "
                                                                                              "variation for data "
                                                                                              "augmentation")
@click.option("--eval_beam_widths", "-bw", default="", show_default=True, type=str, help="Beam widths to evaluate, "
                                                                                         "e.g. 4,8")
@click.option("--gold_preproc", "-G", default=False, show_default=True, type=bool, help="Use gold preprocessing")
@click.option("--learn_tokens", "-T", default=False, show_default=True, type=bool, help="Make parser learn "
                                                                                        "gold-standard tokenization")
@click.option("--textcat_multilabel", "-TML", default=False, show_default=True, type=bool, help="Textcat classes "
                                                                                                "aren't mutually "
                                                                                                "exclusive ("
                                                                                                "multilabel)")
@click.option("--textcat_arch", "-ta", default="bow", show_default=True, type=str, help="Textcat model architecture")
@click.option("--textcat_positive_label", "-tpl", default=None, show_default=True, type=str, help="Textcat positive "
                                                                                                  "label for binary "
                                                                                                  "classes with two "
                                                                                                  "labels")
@click.option("--verbose", "-VV", default=False, show_default=True, type=bool, help="verbosity")
def train(
        lang,
        output_path,
        train_path,
        dev_path,
        debug_data_first,
        raw_text,
        base_model,
        pipeline,
        vectors,
        n_iter,
        n_early_stopping,
        n_examples,
        use_gpu,
        version,
        meta_path,
        init_tok2vec,
        parser_multitasks,
        entity_multitasks,
        noise_level,
        orth_variant_level,
        eval_beam_widths,
        gold_preproc,
        learn_tokens,
        textcat_multilabel,
        textcat_arch,
        textcat_positive_label,
        verbose):
    dframe_trainer = DframeTrainer(lang,
                                   output_path,
                                   train_path,
                                   dev_path,
                                   debug_data_first,
                                   raw_text,
                                   base_model,
                                   pipeline,
                                   vectors,
                                   n_iter,
                                   n_early_stopping,
                                   n_examples,
                                   use_gpu,
                                   version,
                                   meta_path,
                                   init_tok2vec,
                                   parser_multitasks,
                                   entity_multitasks,
                                   noise_level,
                                   orth_variant_level,
                                   eval_beam_widths,
                                   gold_preproc,
                                   learn_tokens,
                                   textcat_multilabel,
                                   textcat_arch,
                                   textcat_positive_label,
                                   verbose)
    dframe_trainer.begin_training()


@main.command()
@click.option("--model", "-m", required=True, type=str, help="Model name or path")
@click.option("--data_path", "-d", required=True, type=str, help="Path of CSV containing validation data")
@click.option("--gpu_id", "-g", default=-1, show_default=True, type=bool, help="Use GPU")
@click.option("--gold_preproc", "-G", default=False, show_default=True, type=bool, help="Use gold preprocessing")
@click.option("--displacy_path", "-dp", default=None, show_default=True, type=str, help="Directory to output rendered "
                                                                                        "parses as HTML")
@click.option("--displacy_limit", "-dl", default=25, show_default=True, type=int, help="Limit of parses to render as "
                                                                                       "HTML")
@click.option("--return_scores", "-R", default=False, show_default=True, type=bool, help="Return dict containing "
                                                                                         "model scores")
def evaluate(
        model,
        data_path,
        gpu_id,
        gold_preproc,
        displacy_path,
        displacy_limit,
        return_scores):
    dframe_evaluator = DframeEvaluator(
        model,
        data_path,
        gpu_id,
        gold_preproc,
        displacy_path,
        displacy_limit,
        return_scores
    )
    dframe_evaluator.begin_evaluation()


if __name__ == '__main__':
    main()
