# coding: utf-8
from __future__ import unicode_literals

import json
import click
from pathlib import Path
from wasabi import Printer
from io import open
from .dframcy import DframCy

messenger = Printer()


@click.group()
def main():
    pass


@main.command()
@click.option("--input_path", "-i", required=True, type=Path)
@click.option("--output_path", "-o", required=True, type=Path)
@click.option("--convert_type", "-t", default="csv", show_default=True, type=str)
@click.option("--language_model", "-l", default="en_core_web_sm", show_default=True, type=str)
def convert(input_path, output_path, convert_type, language_model):
    if input_path.exists():
        with open(input_path, "r") as infile:
            text = infile.read()
            dframcy = DframCy(language_model)
            doc = dframcy.nlp(text)
            annotation_dataframe = dframcy.to_dataframe(doc)
            if convert_type == "csv":
                annotation_dataframe.to_csv(output_path)
            elif convert_type == "json":
                annotation_json = annotation_dataframe.to_json(orient="columns")
                with open(output_path, "w") as outfile:
                    json.dump(annotation_json, outfile)
            else:
                messenger.fail("Unknown output file format '{}'".format(convert_type), exits=-1)
    else:
        messenger.fail("input path {} does not exist".format(input_path), exits=-1)


if __name__ == '__main__':
    main()
