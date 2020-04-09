# DframCy

[![Package Version](https://img.shields.io/pypi/v/dframcy.svg?&service=github)](https://pypi.python.org/pypi/dframcy/)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) 
[![Build Status](https://travis-ci.org/yash1994/dframcy.svg?branch=master)](https://travis-ci.org/yash1994/dframcy) 
[![Coverage Status](https://coveralls.io/repos/github/yash1994/dframcy/badge.svg?branch=master)](https://coveralls.io/github/yash1994/dframcy?branch=development)

DframCy is a light-weight utility module to integrate Pandas Dataframe to spaCy's linguistic annotation and training tasks. DframCy provides clean APIs to convert spaCy's linguistic annotations, Matcher and PhraseMatcher information to Pandas dataframe, also supports training and evaluation of NLP pipeline from CSV/XLXS/XLS without any changes to spaCy's underlying APIs.

## Getting Started

DframCy can be easily installed. Just need to the following:

### Requirements

* Python 3.5 or later
* Pandas
* spaCy >= 2.2.0

Also need to download spaCy's language model:

```bash
python -m spacy download en_core_web_sm
```

For more information refer to: [Models & Languages](https://spacy.io/usage/models)

### Installation:

This package can be installed from [PyPi](https://pypi.org/project/dframcy/) by running:

```bash
pip install dframcy
```

To build from source:

```bash
git clone https://github.com/yash1994/dframcy.git
cd dframcy
python setup.py install
```

## Usage

### Linguistic Annotations

Get linguistic annotation in the dataframe. For linguistic annotations (dataframe column names) refer to [spaCy's Token API](https://spacy.io/api/token) document.

```python
import spacy
from dframcy import DframCy

nlp = spacy.load("en_core_web_sm")

dframcy = DframCy(nlp)
doc = dframcy.nlp(u"Apple is looking at buying U.K. startup for $1 billion")

# default columns: ["id", "text", "start", "end", "pos_", "tag_", "dep_", "head", "ent_type_"]
annotation_dataframe = dframcy.to_dataframe(doc)

# can also pass columns names (spaCy's linguistic annotation attributes)
annotation_dataframe = dframcy.to_dataframe(doc, columns=["text", "lemma_", "lower_", "is_punct"])

# for separate entity dataframe
token_annotation_dataframe, entity_dataframe = dframcy.to_dataframe(doc, separate_entity_dframe=True)

# custom attributes can also be included
from spacy.tokens import Token
fruit_getter = lambda token: token.text in ("apple", "pear", "banana")
Token.set_extension("is_fruit", getter=fruit_getter)
doc = dframcy.nlp(u"I have an apple")

annotation_dataframe = dframcy.to_dataframe(doc, custom_attributes=["is_fruit"])
```

### Rule-Based Matching

```python
# Token-based Matching
import spacy

nlp = spacy.load("en_core_web_sm")

from dframcy.matcher import DframCyMatcher, DframCyPhraseMatcher
dframcy_matcher = DframCyMatcher(nlp)
pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
dframcy_matcher.add("HelloWorld", None, pattern)
doc = dframcy_matcher.nlp("Hello, world! Hello world!")
matches_dataframe = dframcy_matcher(doc)

# Phrase Matching
dframcy_phrase_matcher = DframCyPhraseMatcher(nlp)
terms = [u"Barack Obama", u"Angela Merkel",u"Washington, D.C."]
patterns = [dframcy_phrase_matcher.get_nlp().make_doc(text) for text in terms]
dframcy_phrase_matcher.add("TerminologyList", None, *patterns)
doc = dframcy_phrase_matcher.nlp(u"German Chancellor Angela Merkel and US President Barack Obama "
                                u"converse in the Oval Office inside the White House in Washington, D.C.")
phrase_matches_dataframe = dframcy_phrase_matcher(doc)
```

### Command Line Interface

Dframcy supports command line arguments for conversion of plain text file to linguistically annotated text in CSV/JSON format, training and evaluation of language models from CSV/XLS formatted training data.
[Training data example](https://github.com/yash1994/dframcy/blob/master/data/training_data_format.csv). CLI arguments for training and evaluation are exactly same as [spaCy's CLI](https://spacy.io/api/cli), only difference is the format of training data.

```bash
# convert
dframcy convert -i plain_text.txt -o annotations.csv -t CSV

# train
dframcy train -l en -o spacy_models -t train.csv -d test.csv

# evaluate
dframcy evaluate -m spacy_model/ -d test.csv

# train text classifier
dframcy textcat -o spacy_model/ -t data/textcat_training.csv -d data/textcat_training.csv
```
