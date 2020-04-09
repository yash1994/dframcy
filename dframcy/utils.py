import numpy as np
from ast import literal_eval
from spacy.gold import biluo_tags_from_offsets, tags_to_entities
from wasabi import Printer
from scipy.sparse import coo_matrix

messenger = Printer()


def get_default_columns():
    """
    Default columns for dataframe
    :return: list of default attributes
    """
    return ["id", "text", "start", "end", "pos_", "tag_", "dep_", "head", "ent_type_"]


def get_spacy_token_class_config():
    """
    Configuration of spacy's Token class attribute
    :return: config dictionary of attributes/properties
    """
    token_config = {
        "PROPERTIES": [
            "lefts",
            "rights",
            "n_lefts",
            "subtree",
            "children",
            "n_rights",
            "ancestors",
            "conjuncts",
            "has_vector",
            "is_sent_start",
        ],
        "ATTRIBUTES": [
            "text",
            "head",
            "pos_",
            "tag_",
            "dep_",
            "orth_",
            "norm_",
            "lang_",
            "lemma_",
            "lower_",
            "shape_",
            "is_oov",
            "ent_id_",
            "prefix_",
            "suffix_",
            "is_stop",
            "ent_iob_",
            "is_alpha",
            "is_ascii",
            "is_digit",
            "is_lower",
            "is_upper",
            "is_title",
            "is_punct",
            "is_space",
            "is_quote",
            "like_url",
            "like_num",
            "left_edge",
            "ent_type_",
            "right_edge",
            "ent_kb_id_",
            "is_bracket",
            "like_email",
            "is_currency",
            "is_left_punct",
            "is_right_punct",
        ],
        "ADDITIONAL_ATTRIBUTES": ["id", "start", "end"],
        "INT_FORMAT_ATTRIBUTES": [
            "pos",
            "tag",
            "dep",
            "orth",
            "norm",
            "lang",
            "lower",
            "shape",
            "ent_id",
            "prefix",
            "suffix",
            "ent_iob",
            "ent_type",
        ],
    }

    return token_config


def check_columns_consistency(columns):
    """
    Checks consistency of column names passed by users
    with spacy's Token class.
    :param columns: list of column names
    :return: list of consistent column names
    """
    spacy_token_config = get_spacy_token_class_config()
    consistent_column_names = []
    for column_name in columns:
        if column_name in spacy_token_config["PROPERTIES"]:
            consistent_column_names.append((column_name, "property"))
        elif column_name in spacy_token_config["ATTRIBUTES"]:
            consistent_column_names.append((column_name, "attribute"))
        elif column_name in spacy_token_config["ADDITIONAL_ATTRIBUTES"]:
            consistent_column_names.append((column_name, "additional_attribute"))
        elif column_name in spacy_token_config["INT_FORMAT_ATTRIBUTES"]:
            consistent_column_names.append((column_name, "int_format_attribute"))
        else:
            messenger.warn("Column name '{}' not consistent with spacy's Token class".format(column_name))    

    return consistent_column_names


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

                assert len(sentence) <= len(token_orth), messenger.fail(
                    "number of token and token_orth field mismatch"
                )
                assert len(sentence) <= len(token_tag), messenger.fail(
                    "number of token and token_tag field mismatch"
                )
                assert len(sentence) <= len(token_head), messenger.fail(
                    "number of token and token_head field mismatch"
                )
                assert len(sentence) <= len(token_dep), messenger.fail(
                    "number of token and token_dep field mismatch"
                )

                for token_id, token in enumerate(sentence):
                    token_data = {
                        "id": token_id,
                        "orth": token_orth[token_id],
                        "tag": token_tag[token_id],
                        "head": int(token_head[token_id]),
                        "dep": token_dep[token_id],
                        "ner": tags[token_id],
                    }
                    sentence_tokens.append(token_data)
                doc_sentences.append({"tokens": sentence_tokens})
        elif "tagger" in pipline:
            document_row = dataframe.iloc[[_id]]
            token_orth = document_row["token_orth"].iloc[0].replace("'", "").split(", ")
            token_tag = document_row["token_tag"].iloc[0].replace("'", "").split(", ")
            for sentence in doc.sents:
                sentence_tokens = []

                assert len(sentence) <= len(token_orth), messenger.fail(
                    "number of token and token_orth field mismatch"
                )
                assert len(sentence) <= len(token_tag), messenger.fail(
                    "number of token and token_tag field mismatch"
                )

                for token_id, token in enumerate(sentence):
                    token_data = {
                        "id": token_id,
                        "orth": token_orth[token_id],
                        "tag": token_tag[token_id],
                    }
                    sentence_tokens.append(token_data)
                doc_sentences.append({"tokens": sentence_tokens})
        elif "parser" in pipline:
            document_row = dataframe.iloc[[_id]]
            token_head = document_row["token_head"].iloc[0].replace("'", "").split(", ")
            token_dep = document_row["token_dep"].iloc[0].replace("'", "").split(", ")
            for sentence in doc.sents:
                sentence_tokens = []

                assert len(sentence) <= len(token_head), messenger.fail(
                    "number of token and token_head field mismatch"
                )
                assert len(sentence) <= len(token_dep), messenger.fail(
                    "number of token and token_dep field mismatch"
                )

                for token_id, token in enumerate(sentence):
                    token_data = {
                        "id": token.i,
                        "orth": token.orth_,
                        "tag": token.tag_,
                        "head": int(token_head[token_id]),
                        "dep": token_dep[token_id],
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
                        "ner": tags[token.i],
                    }
                    sentence_tokens.append(token_data)
                doc_sentences.append({"tokens": sentence_tokens})

        list_of_documents.append(
            {"id": _id, "paragraphs": [{"raw": doc.text, "sentences": doc_sentences}]}
        )
    return list_of_documents


def confusion_matrix(prediction, target, label_map):
    inverse_label_map = {v: k for k, v in label_map.items()}
    prediction = np.asarray([inverse_label_map[i] for i in prediction])
    target = np.asarray([inverse_label_map[i] for i in target])

    if len(label_map) == 2:
        return coo_matrix(
            (np.ones(target.shape[0], dtype=np.int64), (target, prediction)),
            shape=(len(label_map), len(label_map)),
        ).toarray()

    true_positives = prediction == target
    true_positives_bins = target[true_positives]

    if len(true_positives_bins):
        tp_sum = np.bincount(true_positives_bins, minlength=len(label_map))
    else:
        tp_sum = np.zeros(len(label_map))

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

    if _confusion_matrix.ndim == 2:  # binary classification
        tp = _confusion_matrix[1][1]
        fp = _confusion_matrix[0][1]
        fn = _confusion_matrix[1][0]

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

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
