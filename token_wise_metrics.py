import os

import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk import pos_tag


def test_some_dummy_code():
    lemmatizer = WordNetLemmatizer()
    list_of_words = ["changing", "changed", "changes", "activated", "activates", "depends", "depended", "dependent",
                     "dependence",
                     "dogs", "regulon", "interaction", "interacting", "controlled", "inhibits", "drives",
                     "negative regulation", "indirectly activate", "promoter elements", "driven",
                     "produced", "production", "regulon", "regulated", "induction", "inducible", "induced", "regulons"]

    list_of_lemmas = [lemmatizer.lemmatize(i, "v") for i in list_of_words]
    # list_of_stems = [stemmer.stem(i) for i in list_of_words]
    print(list_of_lemmas)

    normalized = apply_lemma_normalization(list_of_lemmas)
    print(normalized)

    # Perform part-of-speech tagging
    list_of_words = ["negative regulation", "indirectly activate", "promoter elements", "induction", "inducible",
                     "induced", "regulons"]
    pos_tags = pos_tag(list_of_words)
    print(pos_tags)

    print(lemmatizer.lemmatize("induced", "v"))

    # for i in range(len(list_of_words)):
    #     tag = pos_tags[i][1]
    #     print(pos_tags[i][0])
    #     pt = ''
    #     if "NN" in tag:
    #         pt = "n"
    #     elif "VB" in tag:
    #         pt = "v"
    #     elif "JJ" in tag:
    #         pt = "a"
    #     lemma = lemmatizer.lemmatize(pos_tags[i][0], pt)
    #     print(lemma)


def read_from_txt(file_path):
    f = open(file_path)
    with f as file:
        lines = [line.rstrip() for line in file]

    response_list = []
    for x in range(len(lines) - 6):
        if "### Response:" in lines[x]:
            if "INFO" in lines[x + 6]:
                response_list.append([lines[x + 1]])
    print(len(response_list))


def read_from_excel(file_path):
    df = pd.DataFrame(pd.read_excel(file_path))
    return df


def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, "v")
    return lemma


def convert_to_lower_case(word_list):
    # Convert tokens to lowercase
    lowered = [
        [token.lower() if token is not None else "NoNe" for token in sublist]
        for sublist in word_list
    ]
    return lowered


def get_lemma_list(tokens):
    lemmas = []
    for x in tokens:
        values = []
        for y in [x[0].split(",")]:
            if len(y) > 1:
                for z in y:
                    values.append(lemmatize_word(z.strip()))
            else:
                values.append(lemmatize_word(y[0]))
        lemmas.append(values)
    return lemmas


def apply_lemma_normalization(lemma_list):
    for i in range(len(lemma_list)):
        if lemma_list[i] == "inducible":
            lemma_list[i] = "induce"
        elif lemma_list[i] == "member":
            lemma_list[i] = "member of"
        elif lemma_list[i] == "dependent genes":
            lemma_list[i] = "dependent"
        elif lemma_list[i] == "indirectly activate":
            lemma_list[i] = "activate"
        elif lemma_list[i] == "promoter elements":
            lemma_list[i] = "promoter"
        elif lemma_list[i] == "under control":
            lemma_list[i] = "control"
        elif lemma_list[i] == "under the control":
            lemma_list[i] = "control"
        elif lemma_list[i] == "under control of":
            lemma_list[i] = "control"
        elif lemma_list[i] == "negative regulate":
            lemma_list[i] = "negatively regulate"
        elif lemma_list[i] == "negatively regulates":
            lemma_list[i] = "negatively regulate"
        elif lemma_list[i] == "driven by":
            lemma_list[i] = "driven"
        elif lemma_list[i] == "produce":
            lemma_list[i] = "production"
        elif lemma_list[i] == "dependence":
            lemma_list[i] = "depend"
        elif lemma_list[i] == "combined action":
            lemma_list[i] = "action"
        elif lemma_list[i] == "inducer":
            lemma_list[i] = "induce"

    return lemma_list


def calculate_lemma_based_metric(predicted_words, actual_words, is_normalized):
    scores = []
    for i in range(len(actual_words)):
        if is_normalized:
            pred_sublist = apply_lemma_normalization(predicted_words[i])
            actual_sublist = apply_lemma_normalization(actual_words[i])
        else:
            pred_sublist = predicted_words[i]
            actual_sublist = actual_words[i]

        TP = sum(1 for token in actual_sublist if token in pred_sublist)
        FP = sum(1 for token in pred_sublist if token not in actual_sublist)
        FN = sum(1 for token in actual_sublist if token not in pred_sublist)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        scores.append({"precision": precision, "recall": recall, "f1": f1})

    return scores


def read_ino_file_into_dict(path):
    d = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split("\t")
            d[key] = val.strip("\n")
    return d


def find_ino_id_or_apply_norm_lemmatize(token, ino_dictionary):
    values = []
    if token in ino_dictionary:
        ino_id = ino_dict[token]
        values.append(ino_id)
    else:
        lemmas = [lemmatize_word(token)]
        norm_lemma = apply_lemma_normalization(lemmas)
        for i in norm_lemma:
            if i in ino_dictionary:
                values.append(ino_dictionary[i])
            else:
                values.append(i)
    return values


def get_ino_id_list(tokens, ino_dictionary):
    ino_ids = []
    for x in tokens:
        values = []
        for y in [x[0].split(",")]:
            if len(y) > 1:
                for z in y:
                    ino_id_or_lemma_list = find_ino_id_or_apply_norm_lemmatize(z.strip(), ino_dictionary)
                    for i in ino_id_or_lemma_list:
                        values.append(i)
            else:
                ino_id_or_lemma_list = find_ino_id_or_apply_norm_lemmatize(y[0].strip(), ino_dictionary)
                for i in ino_id_or_lemma_list:
                    values.append(i)
        ino_ids.append(values)
    return ino_ids


def calculate_ino_id_based_metric(predicted_words, actual_words):
    scores = []
    for i in range(len(actual_words)):
        pred_sublist = predicted_words[i]
        actual_sublist = actual_words[i]

        TP = 0
        for pred_token in pred_sublist:
            already_counted = False
            for token in actual_sublist:
                if already_counted is False:
                    if pred_token == token:
                        TP += 1
                        already_counted = True
                else:
                    break

        copy_pred_sublist = [i for i in pred_sublist]
        for item2 in actual_sublist:
            if item2 in copy_pred_sublist:
                copy_pred_sublist.remove(item2)
        FP = len(copy_pred_sublist)

        copy_actual_sublist = [i for i in actual_sublist]
        for item2 in pred_sublist:
            if item2 in copy_actual_sublist:
                copy_actual_sublist.remove(item2)
        FN = len(copy_actual_sublist)


        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        scores.append({"precision": precision, "recall": recall, "f1": f1})

    return scores


if __name__ == '__main__':

    # all the reported experiments done using STRATEGY=INO
    STRATEGY = "INO"
    pth_to_file = "./finetuned_13b_chat.xlsx"
    df = read_from_excel(pth_to_file)
    replaced_df = df.replace({np.nan: None})

    predicted_tokens = [[value] for value in replaced_df["Predicted"]]
    actual_tokens = [[value] for value in replaced_df["True Label"]]

    # lower the words
    lowered_pred_tokens = convert_to_lower_case(predicted_tokens)
    lowered_act_tokens = convert_to_lower_case(actual_tokens)

    if STRATEGY == "LEMMA":
        # apply lemmatization to each keyword
        predicted = get_lemma_list(lowered_pred_tokens)
        actual = get_lemma_list(lowered_act_tokens)
        # calculate lemma based metric, Normalized and Normal way
        scores_list = calculate_lemma_based_metric(predicted, actual, is_normalized=False)

    elif STRATEGY == "NORM_LEMMA":
        # apply lemmatization to each keyword
        predicted = get_lemma_list(lowered_pred_tokens)
        actual = get_lemma_list(lowered_act_tokens)
        # calculate lemma based metric, Normalized and Normal way
        scores_list = calculate_lemma_based_metric(predicted, actual, is_normalized=True)

    elif STRATEGY == "INO":
        # read ino dictionary
        ino_dict = read_ino_file_into_dict("./INO.txt")
        # calculate ino id based metric
        predicted = get_ino_id_list(lowered_pred_tokens, ino_dictionary=ino_dict)
        actual = get_ino_id_list(lowered_act_tokens, ino_dictionary=ino_dict)
        # calculate ino based metric
        scores_list = calculate_ino_id_based_metric(predicted, actual)
    else:
        print("Please select STRATEGY from LEMMA, NORM_LEMMA or INO")
        exit()

    # get the scores
    pred_lemma_column = [predicted[i] for i in range(len(scores_list))]
    act_lemma_column = [actual[i] for i in range(len(scores_list))]
    precision_column = [scores_list[i]["precision"] for i in range(len(scores_list))]
    recall_column = [scores_list[i]["recall"] for i in range(len(scores_list))]
    f1_column = [scores_list[i]["f1"] for i in range(len(scores_list))]

    # add them to df column
    df["Predicted Lemma"] = pred_lemma_column
    df["True Lemma"] = act_lemma_column
    df["Precision"] = precision_column
    df["Recall"] = recall_column
    df["F1"] = f1_column

    ave_precision = df["Precision"].mean()
    ave_recall = df["Recall"].mean()
    ave_f1 = df["F1"].mean()

    df.loc[-1] = ["Average Scores", None, None, None, None, ave_precision, ave_recall, ave_f1]

    print("Average Precision: ", ave_precision)
    print("Average Recall: ", ave_recall)
    print("Average F1: ", ave_f1)

    pth_to_output_file = "./metric-13b-chat.xlsx"
    df.to_excel(pth_to_output_file)
