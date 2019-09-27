def get_default_columns():
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
        "nbor": "tokens.nbor",
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
            continue

    default_columns = get_default_columns()

    additional_attributes = any([True if nc not in default_columns else False for nc in user_defined_columns])

    return user_defined_columns, additional_attributes


def additional_attributes_map(column):

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
        "tokens.nbor": ("nbor", True, 1),
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
        del json_doc["ents"]
        return json_doc
    else:
        return json_doc
