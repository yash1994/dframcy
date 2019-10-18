import numpy as np
from ast import literal_eval
from spacy.gold import biluo_tags_from_offsets, tags_to_entities
from wasabi import Printer
from scipy.sparse import coo_matrix
messenger = Printer()


def get_default_columns():
    """
    To get list of columns names to be default in annotation dataframe.
    :return: list of columns names.
    """
    return [
        'tokens.id',
        'tokens.text',
        'tokens.start',
        'tokens.end',
        'tokens.pos',
        'tokens.tag',
        'tokens.dep',
        'tokens.head',
        'ents.start',
        'ents.end',
        'ents.label']


def map_user_columns_names_with_default(user_columns):
    """
    To map user provided column names to specific format
    :param user_columns: list, of user provided columns
    :return: list, of specifically formatted column names
    """
    column_map = {
        "id": "tokens.id",
        "start": "tokens.start",
        "end": "tokens.end",
        "pos": "tokens.pos",
        "tag": "tokens.tag",
        "dep": "tokens.tag",
        "head": "tokens.head",
        "text": "tokens.text",
        "lemma": "tokens.lemma",
        "lower": "tokens.lower",
        "shape": "tokens.shape",
        "prefix": "tokens.prefix",
        "suffix": "tokens.suffix",
        "is_alpha": "tokens.is_alpha",
        "is_ascii": "tokens.is_ascii",
        "is_digit": "tokens.is_digit",
        "is_lower": "tokens.is_lower",
        "is_upper": "tokens.is_upper",
        "is_title": "tokens.is_title",
        "is_punct": "tokens.is_punct",
        "is_left_punct": "tokens.is_left_punct",
        "is_right_punct": "tokens.is_right_punct",
        "is_space": "tokens.is_space",
        "is_bracket": "tokens.is_bracket",
        "is_quote": "tokens.is_quote",
        "is_currency": "tokens.is_currency",
        "like_url": "tokens.like_url",
        "like_num": "tokens.like_num",
        "like_email": "tokens.like_email",
        "is_oov": "tokens.is_oov",
        "is_stop": "tokens.is_stop",
        "ancestors": "tokens.ancestors",
        "conjuncts": "tokens.conjuncts",
        "children": "tokens.children",
        "lefts": "tokens.lefts",
        "rights": "tokens.rights",
        "n_lefts": "tokens.n_lefts",
        "n_rights": "tokens.n_rights",
        "is_sent_start": "tokens.is_sent_start",
        "has_vector": "tokens.has_vector",
        "vector": "tokens.vector",
        "ent_start": "ents.start",
        "ent_end": "ents.end",
        "ent_label": "ents.label"}

    user_defined_columns = []

    for uc in user_columns:
        try:
            user_defined_columns.append(column_map[uc])
        except KeyError:
            messenger.fail("could not recognize given column name:'{}' so skipping it".format(uc))
            continue

    default_columns = get_default_columns()

    additional_attributes = any([True if nc not in default_columns else False for nc in user_defined_columns])

    return user_defined_columns, additional_attributes


def additional_attributes_map(column):
    """
    To map column names of additional attributes to match spaCy's Token API.
    :param column: list, of column names
    :return: tuple, containing API info
    """
    attributes_map = {
        "tokens.lemma": ("lemma_", False, 0),
        "tokens.lower": ("lower_", False, 0),
        "tokens.shape": ("shape_", False, 0),
        "tokens.prefix": ("prefix_", False, 0),
        "tokens.suffix": ("suffix_", False, 0),
        "tokens.is_alpha": ("is_alpha", False, 0),
        "tokens.is_ascii": ("is_ascii", False, 0),
        "tokens.is_digit": ("is_digit", False, 0),
        "tokens.is_lower": ("is_lower", False, 0),
        "tokens.is_upper": ("is_upper", False, 0),
        "tokens.is_title": ("is_title", False, 0),
        "tokens.is_punct": ("is_punct", False, 0),
        "tokens.is_left_punct": ("is_left_punct", False, 0),
        "tokens.is_right_punct": ("is_right_punct", False, 0),
        "tokens.is_space": ("is_space", False, 0),
        "tokens.is_bracket": ("is_bracket", False, 0),
        "tokens.is_quote": ("is_quote", False, 0),
        "tokens.is_currency": ("is_currency", False, 0),
        "tokens.like_url": ("like_url", False, 0),
        "tokens.like_num": ("like_num", False, 0),
        "tokens.like_email": ("like_email", False, 0),
        "tokens.is_oov": ("is_oov", False, 0),
        "tokens.is_stop": ("is_stop", False, 0),
        "tokens.ancestors": ("ancestors", True, 2),
        "tokens.conjuncts": ("conjuncts", True, 2),
        "tokens.children": ("children", True, 2),
        "tokens.lefts": ("lefts", True, 2),
        "tokens.rights": ("rights", True, 2),
        "tokens.n_lefts": ("n_lefts", False, 2),
        "tokens.n_rights": ("n_rights", False, 2),
        "tokens.is_sent_start": ("is_sent_start", False, 2),
        "tokens.has_vector": ("has_vector", False, 2),
        "tokens.vector": ("vector", False, 2)
    }

    return attributes_map[column]


def merge_entity_details(json_doc):
    """
    To merge entity and token details provided by Doc.to_json() method.
    :param json_doc: dictionary, returned by Doc.to_json()
    :return: new dictionary
    """
    if "ents" in json_doc and json_doc["ents"]:
        ents_dict = {str(ent["start"]) + "_" + str(ent["end"]): ent for ent in json_doc["ents"]}

        new_tokens_data_list = []

        for token_data in json_doc["tokens"]:
            if str(token_data["start"]) + "_" + str(token_data["end"]) in ents_dict:
                token_data["label"] = ents_dict[str(token_data["start"]) + "_" + str(token_data["end"])]["label"]
                new_tokens_data_list.append(token_data)
            else:
                token_data["label"] = None
                new_tokens_data_list.append(token_data)
        json_doc["tokens"] = new_tokens_data_list
        json_doc["ents"] = True
        return json_doc
    else:
        json_doc["ents"] = False
        return json_doc


def get_training_pipeline_from_column_names(columns):
    """
    To infer training pipeline from column names provided by training CSV/XLS.
    :param columns: list, of input dataframe column names
    :return: str, training pipeline
    """
    columns = set(columns)
    _all = {"text", "token_orth", "token_tag", "token_head", "token_dep", "entities"}
    _only_tagger = {"text", "token_orth"}
    _only_parser = {"text", "token_head"}
    _only_ner = {"text", "entities"}
    _tagger_and_parser = {"text", "token_orth", "token_tag", "token_head", "token_dep"}

    if len(columns & _all) == len(_all):
        return "tagger,parser,ner"
    elif len(columns & _only_tagger) == len(_only_tagger):
        return "tagger"
    elif len(columns & _only_parser) == len(_only_tagger):
        return "parser"
    elif len(columns & _only_ner) == len(_only_ner):
        return "ner"
    else:
        return None


def entity_offset_to_biluo_format(nlp, rows, pipline):
    """
    To convert entity offset (start, end) into BILUO format.
    :param nlp: nlp pipeline
    :param rows: dataframe rows (text)
    :param pipline: list, training pipline inferred from data
    :return: list of tuples, containing annotated text and entity tag info
    """
    biluo_rows = []
    default_pipline = ["tagger", "ner"]
    disabled_components = tuple(set(default_pipline) & set(pipline))
    with nlp.disable_pipes(*disabled_components):
        for row in rows.iterrows():
            doc = nlp(row[1]["text"])
            if "ner" in pipline:
                entities = literal_eval(row[1]["entities"])
                tags = biluo_tags_from_offsets(doc, entities)
                if entities:
                    for start, end, label in entities:
                        span = doc.char_span(start, end, label=label)
                        if span and span not in doc.ents:
                            doc.ents = list(doc.ents) + [span]
                if doc.ents:
                    biluo_rows.append((doc, tags))
            else:
                biluo_rows.append((doc, None))
    return biluo_rows


def dataframe_to_spacy_training_json_format(dataframe, nlp, pipline):
    """
    To convert dataframe into spaCy's CLI training JSON format.
    :param dataframe: dataframe, training/validation data
    :param nlp: nlp pipeline
    :param pipline: str, training pipeline
    :return: JSON object, containing training data
    """
    pipline = pipline.split(",")
    list_of_documents = []
    biluo_rows = entity_offset_to_biluo_format(nlp, dataframe, pipline)

    for _id, biluo_row in enumerate(biluo_rows):
        doc, tags = biluo_row
        doc_sentences = []
        if "tagger" in pipline and "parser" in pipline and "ner" in pipline:
            document_row = dataframe.iloc[[_id]]
            token_orth = document_row["token_orth"].iloc[0].replace("'", "").split(", ")
            token_tag = document_row["token_tag"].iloc[0].replace("'", "").split(", ")
            token_head = document_row["token_head"].iloc[0].replace("'", "").split(", ")
            token_dep = document_row["token_dep"].iloc[0].replace("'", "").split(", ")
            tags_to_entities(tags)
            for sentence in doc.sents:
                sentence_tokens = []

                assert len(sentence) <= len(token_orth), messenger.fail("number of token and token_orth field mismatch")
                assert len(sentence) <= len(token_tag), messenger.fail("number of token and token_tag field mismatch")
                assert len(sentence) <= len(token_head), messenger.fail("number of token and token_head field mismatch")
                assert len(sentence) <= len(token_dep), messenger.fail("number of token and token_dep field mismatch")

                for token_id, token in enumerate(sentence):
                    token_data = {
                        "id": token_id,
                        "orth": token_orth[token_id],
                        "tag": token_tag[token_id],
                        "head": int(token_head[token_id]),
                        "dep": token_dep[token_id],
                        "ner": tags[token_id]
                    }
                    sentence_tokens.append(token_data)
                doc_sentences.append({"tokens": sentence_tokens})
        elif "tagger" in pipline:
            document_row = dataframe.iloc[[_id]]
            token_orth = document_row["token_orth"].iloc[0].replace("'", "").split(", ")
            token_tag = document_row["token_tag"].iloc[0].replace("'", "").split(", ")
            for sentence in doc.sents:
                sentence_tokens = []

                assert len(sentence) <= len(token_orth), messenger.fail("number of token and token_orth field mismatch")
                assert len(sentence) <= len(token_tag), messenger.fail("number of token and token_tag field mismatch")

                for token_id, token in enumerate(sentence):
                    token_data = {
                        "id": token_id,
                        "orth": token_orth[token_id],
                        "tag": token_tag[token_id]
                    }
                    sentence_tokens.append(token_data)
                doc_sentences.append({"tokens": sentence_tokens})
        elif "parser" in pipline:
            document_row = dataframe.iloc[[_id]]
            token_head = document_row["token_head"].iloc[0].replace("'", "").split(", ")
            token_dep = document_row["token_dep"].iloc[0].replace("'", "").split(", ")
            for sentence in doc.sents:
                sentence_tokens = []

                assert len(sentence) <= len(token_head), messenger.fail("number of token and token_head field mismatch")
                assert len(sentence) <= len(token_dep), messenger.fail("number of token and token_dep field mismatch")

                for token_id, token in enumerate(sentence):
                    token_data = {
                        "id": token.i,
                        "orth": token.orth_,
                        "tag": token.tag_,
                        "head": int(token_head[token_id]),
                        "dep": token_dep[token_id]
                    }
                    sentence_tokens.append(token_data)
                doc_sentences.append({"tokens": sentence_tokens})
        elif "ner" in pipline:
            for sentence in doc.sents:
                sentence_tokens = []
                for token in sentence:
                    token_data = {
                        "id": token.i,
                        "orth": token.orth_,
                        "tag": token.tag_,
                        "head": token.head.i - token.i,
                        "dep": token.dep_,
                        "ner": tags[token.i]
                    }
                    sentence_tokens.append(token_data)
                doc_sentences.append({"tokens": sentence_tokens})

        list_of_documents.append({
            "id": _id,
            "paragraphs": [
                {
                    "raw": doc.text,
                    "sentences": doc_sentences
                }
            ]
        })
    return list_of_documents


def confusion_matrix(prediction, target, label_map):
    inverse_label_map = {v: k for k, v in label_map.items()}
    prediction = np.asarray([inverse_label_map[i] for i in prediction])
    target = np.asarray([inverse_label_map[i] for i in target])

    if len(label_map) == 2:
        return coo_matrix((np.ones(target.shape[0], dtype=np.int64), (target, prediction)),
                          shape=(len(label_map), len(label_map))).toarray()

    true_positives = prediction == target
    true_positives_bins = target[true_positives]

    if len(true_positives_bins):
        tp_sum = np.bincount(true_positives_bins, minlength=len(label_map))
    else:
        true_sum = prediction_sum = tp_sum = np.zeros(len(label_map))

    if len(prediction):
        prediction_sum = np.bincount(prediction, minlength=len(label_map))
    else:
        prediction_sum = np.zeros(len(label_map))

    if len(target):
        true_sum = np.bincount(target, minlength=len(label_map))
    else:
        true_sum = np.zeros(len(label_map))

    fp = prediction_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    tn = target.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def precision_recall_fscore(_confusion_matrix):

    if _confusion_matrix.ndim == 2: # binary classification
        tp = _confusion_matrix[1][1]
        tn = _confusion_matrix[0][0]
        fp = _confusion_matrix[0][1]
        fn = _confusion_matrix[1][0]

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f_score
    else:
        tp_sum = _confusion_matrix[:, 1, 1]
        pred_sum = tp_sum + _confusion_matrix[:, 0, 1]
        true_sum = tp_sum + _confusion_matrix[:, 1, 0]

        # avoid nan/inf
        pred_sum[pred_sum == 0.0] = 1
        true_sum[true_sum == 0.0] = 1

        precision = tp_sum / pred_sum

        recall = tp_sum / true_sum

        f_denom = precision + recall
        f_denom[f_denom == 0.0] = 1
        f_score = 2 * (precision * recall) / f_denom

        # weighted (true labels) average
        precision = np.average(precision, weights=true_sum)
        recall = np.average(recall, weights=true_sum)
        f_score = np.average(f_score, weights=true_sum)

        return precision, recall, f_score
