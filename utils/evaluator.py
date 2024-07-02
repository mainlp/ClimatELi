import io, os, re
from glob import glob
import json, pickle
from sklearn.metrics import accuracy_score, cohen_kappa_score
from collections import Counter, defaultdict
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import reduce

def read_tsv(filename):
    file_basename = os.path.basename(filename)
    if not os.path.exists(filename):
        print("!!! File %s does not exist !!!" % (filename))
        return {"token_annos": [], "entity_annos": []}
    with io.open(filename, "r", encoding="utf-8") as f:
        tmp_text = re.sub(r"\n\t+\n", r"\n\n", f.read().strip())
        sentences = [x.split("\n") for x in tmp_text.split("\n\n")]
    token_annos = []
    entity_annos = []
    for sentence_id, sentence in enumerate(sentences):
        for token_id, token in enumerate(sentence):
            fields = token.split("\t")
            token_annos.append([file_basename, sentence_id, token_id,  "-".join(fields[2:])])
            if fields[2] == "B":
                entity_annos.append([file_basename, sentence_id, token_id, token_id, [fields[0]], [fields[1]], fields[3]])
            elif fields[2] == "I":
                assert entity_annos[-1][3] == token_id - 1
                entity_annos[-1][3] = token_id
                entity_annos[-1][4].append(fields[0])
                entity_annos[-1][5].append(fields[1])

    # Change entity_annos list to tuple
    for entity_anno_id in range(len(entity_annos)):
        entity_annos[entity_anno_id][4] = tuple(entity_annos[entity_anno_id][4])
        entity_annos[entity_anno_id][5] = tuple(entity_annos[entity_anno_id][5])
    annotations = {"token_annos": token_annos, "entity_annos": entity_annos}
    return annotations

def compare_annos(gold_annos, pred_annos):
    gold_token_annos, gold_entity_annos = gold_annos["token_annos"], gold_annos["entity_annos"]
    pred_token_annos, pred_entity_annos = pred_annos["token_annos"], pred_annos["entity_annos"]
    assert len(gold_token_annos) == len(pred_token_annos)

    score_dict = {}

    ### Token-level
    gold_token_untyped = [x[-1].split("-")[0] for x in gold_token_annos]
    gold_token_typed = [x[-1] for x in gold_token_annos]
    pred_token_untyped = [x[-1].split("-")[0] for x in pred_token_annos]
    pred_token_typed = [x[-1] for x in pred_token_annos]

    score_dict["untyped_token_accuracy"] = 100.0 * accuracy_score(gold_token_untyped, pred_token_untyped)
    score_dict["typed_token_accuracy"] = 100.0 * accuracy_score(gold_token_typed, pred_token_typed)
    score_dict["untyped_token_kappa"] =  100.0 *cohen_kappa_score(gold_token_untyped, pred_token_untyped)
    score_dict["typed_token_kappa"] = 100.0 * cohen_kappa_score(gold_token_typed, pred_token_typed)

    ### Entity level
    gold_entity_untyped = set([tuple(x[:-1]) for x in gold_entity_annos])
    pred_entity_untyped = set([tuple(x[:-1]) for x in pred_entity_annos])
    gold_entity_typed = set([tuple(x) for x in gold_entity_annos])
    pred_entity_typed = set([tuple(x) for x in pred_entity_annos])
    shared_entity_untyped = gold_entity_untyped.intersection(pred_entity_untyped)
    shared_entity_typed = gold_entity_typed.intersection(pred_entity_typed)

    score_dict["untyped_entity_precision"] =  100.0 * len(shared_entity_untyped) / len(pred_entity_untyped)
    score_dict["untyped_entity_recall"] = 100.0 * len(shared_entity_untyped) / len(gold_entity_untyped)
    if (score_dict["untyped_entity_precision"] + score_dict["untyped_entity_recall"]) != 0:
        score_dict["untyped_entity_f1"] = 2 * score_dict["untyped_entity_precision"] * score_dict["untyped_entity_recall"] \
                                      / (score_dict["untyped_entity_precision"] + score_dict["untyped_entity_recall"])
    else:
        score_dict["untyped_entity_f1"] = 0


    score_dict["typed_entity_precision"] =  100.0 * len(shared_entity_typed) / len(pred_entity_typed)
    score_dict["typed_entity_recall"] = 100.0 * len(shared_entity_typed) / len(gold_entity_typed)
    if (score_dict["typed_entity_precision"] + score_dict["typed_entity_recall"]) != 0:
        score_dict["typed_entity_f1"] = 2 * score_dict["typed_entity_precision"] * score_dict["typed_entity_recall"] \
                                      / (score_dict["typed_entity_precision"] + score_dict["typed_entity_recall"])
    else:
        score_dict["typed_entity_f1"] = 0
    return score_dict

def scorer_wrapper(gold_annos, pred_annos, filters=None):
    assert filters==None or set(filters).issubset(possible_filters)
    assert len(gold_annos.keys()) == len(pred_annos.keys()) # same amount of  documents
    concat_gold_annos = {"token_annos":[], "entity_annos":[]}
    concat_pred_annos = {"token_annos": [], "entity_annos": []}
    for goldkey in gold_annos.keys():
        assert len(gold_annos[goldkey]["token_annos"]) == len(pred_annos[goldkey]["token_annos"])  # same amount of tokens
        assert [x[:3] for x in gold_annos[goldkey]["token_annos"]] == [x[:3] for x in
                                                              pred_annos[goldkey]["token_annos"]]  # same token ids
        concat_gold_annos["token_annos"] += gold_annos[goldkey]["token_annos"]
        concat_gold_annos["entity_annos"] += gold_annos[goldkey]["entity_annos"]
        concat_pred_annos["token_annos"] += pred_annos[goldkey]["token_annos"]
        concat_pred_annos["entity_annos"] += pred_annos[goldkey]["entity_annos"]
    score_dict = compare_annos(concat_gold_annos, concat_pred_annos)
    return score_dict

def scorer_printer(scenario_name, score_dict):
    print("\n\no This is evaluation on %s:" % scenario_name)
    score_orders = ["token_accuracy", "token_kappa", "entity_precision", "entity_recall", "entity_f1"]
    typed_scores = ["untyped", "typed"]
    print("& " + " & ".join(score_orders), end="  \\\\\n")
    for typed_score in typed_scores:
        print("%s & " % typed_score)
        print(" & ".join(["%.2f" % (score_dict[typed_score + "_" + score_order]) for score_order in score_orders]), end="  \\\\\n")

def check_cc_related(title):
    if title in link_metadata_dict.keys() and "climate_text" in link_metadata_dict[title].keys() and "climate_link" in link_metadata_dict[title].keys():
        return link_metadata_dict[title]["climate_text"], link_metadata_dict[title]["climate_link"]
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        contains_climate_text = 'climate' in soup.get_text().lower()
        contains_climate_change_link = False
        for link in soup.find_all('a', href=True):
            if 'climate_change' in link['href']:
                contains_climate_change_link = True
                break
        link_metadata_dict[title]["climate_text"] = contains_climate_text
        link_metadata_dict[title]["climate_link"] = contains_climate_change_link
        print(title, len(link_metadata_dict))
        return contains_climate_text, contains_climate_change_link
    else:
        link_metadata_dict[title]["climate_text"] = False
        link_metadata_dict[title]["climate_link"] = False
        return False, False

def check_nominal_entity(pos_list):
    return "NOUN" in pos_list or "PROPN" in pos_list

def check_valid_link(title):
    if title in link_metadata_dict.keys() and "valid_link" in link_metadata_dict[title].keys():
        return link_metadata_dict[title]["valid_link"][0]
    link = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(link)
    blank_page_text = "Wikipedia does not have an article with this exact name"
    disambiguation_page_text = "you may wish to change the link to point directly to the intended article"
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        if disambiguation_page_text in soup.get_text():
            link_metadata_dict[title]["valid_link"] = [False, disambiguation_page_text]
            return False
        elif blank_page_text in soup.get_text():
            link_metadata_dict[title]["valid_link"] = [False, blank_page_text]
            return False
        else:
            link_metadata_dict[title]["valid_link"] = [True]
            return True
    else:
        link_metadata_dict[title]["valid_link"] = [False, "page not found or other error (not 200)"]
        return False

def apply_annotation_filter(all_annotations_dict, new_filter, from_versions="all"):
    assert new_filter in possible_filters
    if from_versions == "all":
        from_versions = list(all_annotations_dict.keys())
    elif isinstance(from_versions, str):
        from_versions = [x for x in all_annotations_dict.keys() if x.endswith(from_versions)]
    for from_version in from_versions:
        all_annotations_dict[from_version + "_" + new_filter] = defaultdict(dict)
        for doc_name in all_annotations_dict[from_version].keys():
            all_annotations_dict[from_version + "_" + new_filter][doc_name] = {"token_annos": [], "entity_annos": []}
            doc_token_annos = all_annotations_dict[from_version][doc_name]["token_annos"]
            doc_entity_annos = all_annotations_dict[from_version][doc_name]["entity_annos"]

            # apply filter to entity-level annotations
            non_nominal_entities = []
            for doc_entity_anno in doc_entity_annos:
                contains_climate_text, contains_climate_link = check_cc_related(doc_entity_anno[-1])
                is_valid_link = check_valid_link(doc_entity_anno[-1])
                contains_nominal = check_nominal_entity(doc_entity_anno[5])
                if new_filter == "valid_link" and is_valid_link:
                    all_annotations_dict[from_version + "_" + new_filter][doc_name]["entity_annos"].append(doc_entity_anno)
                elif new_filter == "climate_text" and contains_climate_text:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["entity_annos"].append(doc_entity_anno)
                elif new_filter == "climate_link" and contains_climate_link:
                    all_annotations_dict[from_version + "_" + new_filter][doc_name]["entity_annos"].append(doc_entity_anno)
                elif new_filter == "nominal_entity":
                    if contains_nominal:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["entity_annos"].append(doc_entity_anno)
                    else:
                        non_nominal_entities.append(doc_entity_anno)


            # apply filter to token-level annotations
            for doc_token_anno in doc_token_annos:
                if new_filter == "valid_link":
                    if "-" in doc_token_anno[-1] and not check_valid_link(doc_token_anno[-1].split("-")[1]):
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(
                            doc_token_anno[:-1] + ["O"])
                    else:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(
                            doc_token_anno)
                elif new_filter == "climate_text":
                    if "-" in doc_token_anno[-1] and not check_cc_related(doc_token_anno[-1].split("-")[1])[0]:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(doc_token_anno[:-1] + ["O"])
                    else:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(doc_token_anno)
                elif new_filter == "climate_link":
                    if "-" in doc_token_anno[-1] and not check_cc_related(doc_token_anno[-1].split("-")[1])[1]:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(
                            doc_token_anno[:-1] + ["O"])
                    else:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(
                            doc_token_anno)
                elif new_filter == "nominal_entity":
                    found_encompassing_non_nominal_entity = False
                    for non_nominal_entity in non_nominal_entities:
                        if non_nominal_entity[0] == doc_token_anno[0] \
                            and  non_nominal_entity[1] == doc_token_anno[1] \
                            and non_nominal_entity[2] <= doc_token_anno[2] <= non_nominal_entity[3]:
                            all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(
                                doc_token_anno[:-1] + ["O"])
                            found_encompassing_non_nominal_entity = True
                            break
                    # if not in any of the non_nominal_entities (not breaked)
                    if not found_encompassing_non_nominal_entity:
                        all_annotations_dict[from_version + "_" + new_filter][doc_name]["token_annos"].append(
                                doc_token_anno)


    return all_annotations_dict

def plot_threshold(wikifier_thres_score_dict=None, tagme_thres_score_dict=None):
    ### Get data
    thres_range_dict = {"wikifier": range(10, 0, -1), "tagme": range(0, 10, 1)} # TODO: SZ update tagme include thres 1
    plot_thres_dict = {}

    if wikifier_thres_score_dict:
        plot_thres_dict.update({
            "wikifier_P" : [wikifier_thres_score_dict[x]["typed_entity_precision"] for x in thres_range_dict["wikifier"]],
            "wikifier_R": [wikifier_thres_score_dict[x]["typed_entity_recall"] for x in thres_range_dict["wikifier"]],
            "wikifier_F1": [wikifier_thres_score_dict[x]["typed_entity_f1"] for x in thres_range_dict["wikifier"]],
        })
    if tagme_thres_score_dict:
        plot_thres_dict.update({
            "tagme_P": [tagme_thres_score_dict[x]["typed_entity_precision"] for x in thres_range_dict["tagme"]],
            "tagme_R": [tagme_thres_score_dict[x]["typed_entity_recall"] for x in thres_range_dict["tagme"]],
            "tagme_F1": [tagme_thres_score_dict[x]["typed_entity_f1"] for x in thres_range_dict["tagme"]],
        })

    ### Plotting
    colors = {"wikifier": "b", "tagme": "g"}
    shapes = {"P": "+-", "R": "^-", "F1": "o-"}
    line_dict = {}
    plot_thres_range = [0.1 * x for x in thres_range_dict["wikifier"]]
    for plot_thres_dict_key in plot_thres_dict.keys():
        color = colors[plot_thres_dict_key.split("_")[0]]
        shape = shapes[plot_thres_dict_key.split("_")[1]]
        plt.plot(plot_thres_range, plot_thres_dict[plot_thres_dict_key], color+"--")
        plt.plot(plot_thres_range, plot_thres_dict[plot_thres_dict_key], color +shape, label=plot_thres_dict_key)
    plt.legend(loc='upper left')
    plt.savefig("../figs/lineplot_threshold.png")

def get_token_entity_count(version):
    token_count = sum([len(x["token_annos"]) for x in all_annotations_dict[version].values()])
    concat_entity_annos = list(reduce(lambda x,y: x+y, [x["entity_annos"] for x in all_annotations_dict[version].values()]))
    entity_count = len(concat_entity_annos)
    unique_entity_count = len(set([x[-1] for x in concat_entity_annos]))
    return token_count, entity_count, unique_entity_count

def convert_version_to_ding(version_str):
    ding_str = ""
    for possible_filter in possible_filters:
        if possible_filter in version_str:
            ding_str += "\\yes &"
        else:
            ding_str += " & "
    return ding_str



if __name__ == '__main__':
    option_iaa = True
    option_wikifier = True
    option_tagme = True
    option_genre = True


    gold_files = sorted(glob("../logan_gold_tsv/main/*.tsv"))

    ### Load manual and predicted annotations
    all_annotations_dict = defaultdict(dict)
    # gold
    for gold_file in gold_files:
        gold_file_basename = os.path.basename(gold_file)
        all_annotations_dict["gold"][gold_file_basename] = read_tsv(gold_file)
    # IAA gold + secondary
    secondary_iaa_file = "../logan_gold_tsv/double/en_wiki_ParisAgreement.tsv"
    gold_iaa_file = "../logan_gold_tsv/main/en_wiki_ParisAgreement.tsv"
    all_annotations_dict["gold_iaa"]["en_wiki_ParisAgreement.tsv"] = read_tsv(gold_iaa_file)
    all_annotations_dict["secondary_iaa"]["en_wiki_ParisAgreement.tsv"] = read_tsv(secondary_iaa_file)
    # wikifier
    if option_wikifier:
        for thres_num in range(10, 0, -1):  # range(1, 11):
            for gold_file in gold_files:
                gold_file_basename = os.path.basename(gold_file)
                wikifier_thres_file = "../machine_annotation/wikifier/tsv/threshold_%02d/" % thres_num + gold_file_basename
                all_annotations_dict["wikifier_%02d" % thres_num][gold_file_basename] = read_tsv(wikifier_thres_file)
    # tagme
    if option_tagme:
        for thres_num in range(0, 10, 1):  # range(1, 11):
            for gold_file in gold_files:
                gold_file_basename = os.path.basename(gold_file)
                tagme_thres_file = "../machine_annotation/tagme/tsv/threshold_%02d/" % thres_num + gold_file_basename
                all_annotations_dict["tagme_%02d" % thres_num][gold_file_basename] = read_tsv(tagme_thres_file)
    # genre
    if option_genre:
        for gold_file in gold_files:
            gold_file_basename = os.path.basename(gold_file)
            genre_file = "../machine_annotation/genre/tsv/" + gold_file_basename
            all_annotations_dict["genre"][gold_file_basename] = read_tsv(genre_file)

    ### Apply filterings
    possible_filters = ["valid_link", "nominal_entity", "climate_text", "climate_link"]
    link_metadata_dict = defaultdict(dict)
    previous_link_metadata_files = list(glob("../link_metadata_v*.json"))
    if previous_link_metadata_files != []:
        latest_link_metadata_verion = max([int(x[-7:-5]) for x in previous_link_metadata_files])
        with io.open("../link_metadata_v%02d.json" % latest_link_metadata_verion, "r", encoding="utf8") as fp:
            latest_link_metadata_dict = json.load(fp)
            link_metadata_dict.update(latest_link_metadata_dict)
    else:
        latest_link_metadata_verion = 0

    ### Filter steps
    # Step 1: remove valid links
    all_annotations_dict = apply_annotation_filter(all_annotations_dict, new_filter="valid_link",
                                                   from_versions="all")
    # Step 2: apply climate_text, climate_link, nominal_text individually to output of step 1
    all_annotations_dict = apply_annotation_filter(all_annotations_dict, new_filter="climate_text",
                                                   from_versions="_valid_link")
    all_annotations_dict = apply_annotation_filter(all_annotations_dict, new_filter="climate_link",
                                                   from_versions="_valid_link")
    all_annotations_dict = apply_annotation_filter(all_annotations_dict, new_filter="nominal_entity",
                                                   from_versions="_valid_link")
    # Step 3
    all_annotations_dict = apply_annotation_filter(all_annotations_dict, new_filter="climate_text",
                                                   from_versions="_valid_link_nominal_entity")
    all_annotations_dict = apply_annotation_filter(all_annotations_dict, new_filter="climate_link",
                                                   from_versions="_valid_link_nominal_entity")


    ### save link_metadata
    latest_link_metadata_verion += 1
    if len(link_metadata_dict) > len(latest_link_metadata_dict):
        with io.open("../link_metadata_v%02d.json" % latest_link_metadata_verion, "w", encoding="utf8") as fp:
            json.dump(link_metadata_dict, fp)
        print("o Updated link metadata version %02d dumped to json file" % latest_link_metadata_verion)

    ### save filtered annotations
    with io.open("../annotations.pkl", "wb") as fp:
        pickle.dump(all_annotations_dict, fp)
    print("o Saved original and filtered annotations")


    ### Print linker + filter versions statistics
    linker_keys = ["gold", "wikifier_10",  "tagme_00", "genre"]
    filter_keys = ["", "_valid_link",
               "_valid_link_nominal_entity", "_valid_link_climate_text", "_valid_link_climate_link",
               "_valid_link_nominal_entity_climate_text", "_valid_link_nominal_entity_climate_link"]
    print("\n\n & ", " & ".join(linker_keys), end="  \\\\\n")
    prev_token_count = None
    for filter_key in filter_keys:
        filtered_linker_token_counts = [get_token_entity_count(linker_key + filter_key)[0] for linker_key in linker_keys]
        filtered_linker_entity_counts = [get_token_entity_count(linker_key + filter_key)[1] for linker_key in linker_keys]
        filtered_linker_unique_entity_counts = [get_token_entity_count(linker_key + filter_key)[2] for linker_key in
                                         linker_keys]
        filtered_linker_entity_counts_and_unique_to_print = [
            " %d & %d " % (filtered_linker_entity_counts[idx], filtered_linker_unique_entity_counts[idx])
            for idx in range(len(filtered_linker_entity_counts))]
        assert len(set(filtered_linker_token_counts)) == 1
        print(convert_version_to_ding(filter_key), " & ".join(filtered_linker_entity_counts_and_unique_to_print), end="  \\\\\n")



    ### Comparisons between annotations
    ### Inter-annotator agreement
    if option_iaa:
        score_dict = scorer_wrapper(all_annotations_dict["gold_iaa"],
                                    all_annotations_dict["secondary_iaa"],
                                    filters=None)
        scorer_printer("IAA on en_wiki_ParisAgreement", score_dict)

    if option_wikifier:
        wikifier_thres_score_dict = {}
        for thres_num in range(10, 0, -1): # range(1, 11):
            wikifier_thres_score_dict[thres_num] = scorer_wrapper(all_annotations_dict["gold"],
                                                                  all_annotations_dict["wikifier_%02d" % thres_num],
                                                                  filters=None)
            scorer_printer("o Wikifier score with threshold %d" % thres_num, wikifier_thres_score_dict[thres_num])

    if option_tagme:
        tagme_thres_score_dict = {}
        for thres_num in range(0, 10, 1):  # range(1, 11):
            tagme_thres_score_dict[thres_num] = scorer_wrapper(all_annotations_dict["gold"],
                                                                  all_annotations_dict["tagme_%02d" % thres_num],
                                                                  filters=None)
            scorer_printer("o Tagme score with threshold %d" % thres_num, tagme_thres_score_dict[thres_num])

    if option_genre:
        genre_score_dict = scorer_wrapper(all_annotations_dict["gold"],
                                          all_annotations_dict["genre"],
                                          filters=None)
        scorer_printer("o GENRE score", genre_score_dict)


    ### Analyses
    if option_wikifier and option_tagme:
        plot_threshold(wikifier_thres_score_dict=wikifier_thres_score_dict, tagme_thres_score_dict=tagme_thres_score_dict)


    print("o Done!")