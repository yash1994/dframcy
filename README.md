# DframCy
DframCy is a light-weight utility module to integrate Pandas Dataframe to spaCy's linguistic annotation and training tasks. DframCy provides clean APIs to convert spaCy's linguistic annotations, Matcher and PhraseMatcher information to Pandas dataframe, also supports training and evaluation of NLP pipeline from CSV/XLXS/XLS without any changes to spaCy's underlying APIs.

## Getting Started
DframCy can be easily installed. Just need to the following:
### Requirements
* Python 3.6
* Pandas
* spaCy 2.2.0
* python-magic

Also need to download spaCy's language model:
```bash
python -m spacy download en
```
For more information refer to: [Models & Languages](https://spacy.io/usage/models)

For installation:
```bash
git clone https://github.com/yash1994/dframcy.git
cd dframcy
python setup.py install
```

## Usage

#### Linguistic Annotations
Get linguistic annotation in the dataframe. For linguistic annotations (dataframe column names) refer to [spaCy's Token API](https://spacy.io/api/token) document.
```python
from dframcy import DframCy
dframcy = DframCy("en_core_web_sm")
doc = dframcy.nlp(u"Apple is looking at buying U.K. startup for $1 billion")

# default columns: ['id', 'text', 'start', 'end', 'pos', 'tag', 'dep', 'head', 'label'] 
annotation_dataframe = dframcy.to_dataframe(doc)

# can also pass columns names (spaCy's linguistic annotation attributes)
annotation_dataframe = dframcy.to_dataframe(doc, columns=["text", "lemma", "lower", "is_punct"])

# for separate entity dataframe
token_annotation_dataframe, entity_dataframe = dframcy.to_dataframe(doc, separate_entity_dframe=True) 
```
#### Rule-Based Matching
```python
# Token-based Matching
from dframcy.matcher import DframCyMatcher, DframCyPhraseMatcher
dframcy_matcher = DframCyMatcher("en_core_web_sm")
pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
dframcy_matcher.add("HelloWorld", None, pattern)
doc = dframcy_matcher.nlp("Hello, world! Hello world!")
matches_dataframe = dframcy_matcher(doc)

# Phrase Matching
dframcy_phrase_matcher = DframCyPhraseMatcher("en_core_web_sm")
terms = [u"Barack Obama", u"Angela Merkel",u"Washington, D.C."]
patterns = [dframcy_phrase_matcher.get_nlp().make_doc(text) for text in terms]
dframcy_phrase_matcher.add("TerminologyList", None, *patterns)
doc = dframcy_phrase_matcher.nlp(u"German Chancellor Angela Merkel and US President Barack Obama "
                                u"converse in the Oval Office inside the White House in Washington, D.C.")
phrase_matches_dataframe = dframcy_phrase_matcher(doc)
```
#### Command Line Interface
Dframcy supports command line arguments for conversion of plain text file to linguistically annotated text in CSV/JSON format, training and evaluation of language models from CSV/XLS formatted training data.
[Training data example](https://github.com/yash1994/dframcy/blob/master/data/training_data_format.csv). CLI arguments for training and evaluation are exactly same as [spaCy's CLI](https://spacy.io/api/cli), only difference is the format of training data.
```bash
# convert command
dframcy convert -i plain_text.txt -o annotations.csv -t CSV

# train command
dframcy train -l en -o spacy_models -t train.csv -d test.csv

# evaluate command
dframcy evaluate -m spacy_model/ -d test.csv
```
