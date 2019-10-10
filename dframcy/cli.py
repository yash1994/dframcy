# coding: utf-8
from __future__ import unicode_literals

import json
import click
from pathlib import Path
from wasabi import Printer
from io import open
from .dframcy import DframCy
from .utils import get_default_columns

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


if __name__ == '__main__':
    main()
