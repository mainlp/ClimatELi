import io, re, os
from glob import glob
from functools import reduce

import stanza
stanza_nlp = stanza.Pipeline(lang="en", processors="tokenize,pos")

short_list = io.open("../raw_document/short_list.txt", "r", encoding="utf-8").read().split("\n")

def stanza_output(doc_str):
    doc = stanza_nlp(doc_str)
    tokens = []
    uposes = []
    for sentence in doc.sentences:
        tokens.append([x.text for x in sentence.tokens])
        uposes.append([x.words[0].upos for x in sentence.tokens])
    return tokens, uposes


def extract_tokenized(filename):
    with io.open(filename, "r", encoding="utf-8") as f:
        tokenized_str = f.read().strip().split("\n")
    tokens = []
    uposes = []
    for tokenized_sent in tokenized_str:
        tokens.append([x.split("|")[0] for x in tokenized_sent.split(" ")])
        uposes.append([x.split("|")[1] for x in tokenized_sent.split(" ")])
    return tokens, uposes

def get_index_long_short(long, short, min_i):
    for curr_i in range(min_i, len(long)-len(short)+1):
        if long[curr_i:curr_i+len(short)] == short:
            return curr_i, curr_i+len(short)
    print("o ERROR: cannot find short in long", short, long, min_i, sep="\n")
    return

def extract_markdown(filename, gold_tokens):
    with io.open(filename, "r", encoding="utf-8") as f:
        tokenized_str = re.sub(r"\n+", "\n", f.read().strip()).split("\n")

    links = []
    for tokenized_sent_id, tokenized_sent in enumerate(tokenized_str):
        regex_results = re.findall(r"\[([^\]]+)\]\((\S+)\)", tokenized_sent)

        ### PART 1: remove link and validate stanza parse
        link_removed_sent = tokenized_sent
        for regex_result in regex_results:
            link_removed_sent = link_removed_sent.replace("[%s](%s)" % regex_result, regex_result[0])
        link_removed_tokenized = reduce(lambda x,y: x+y, stanza_output(link_removed_sent)[0])

        ### Token match validation between gold_tokenized_raw and gold entity annotation
        if link_removed_tokenized != gold_tokens[tokenized_sent_id]:
            link_removed_tokenized_highlighted = [re.sub(".", "_", x)
                                                  if len(gold_tokens[tokenized_sent_id])>idx and
                                                     link_removed_tokenized[idx]==gold_tokens[tokenized_sent_id][idx]
                                                  else x for idx, x in enumerate(link_removed_tokenized)]
            gold_tokens_highlighted = [
                re.sub(".", "_", x) if len(link_removed_tokenized)>idx and link_removed_tokenized[idx] == gold_tokens[tokenized_sent_id][idx] else x for
                idx, x in enumerate(gold_tokens[tokenized_sent_id])]
            print(filename, tokenized_sent_id, " ".join(link_removed_tokenized_highlighted), " ".join(gold_tokens_highlighted), sep="\n", end="\n\n")

        ### PART 2: allocate link to token ID
        min_i = 0
        link_start_ends = []
        for regex_result in regex_results:
            regex_tokenized = reduce(lambda x,y: x+y, stanza_output(regex_result[0])[0])
            curr_i, min_i = get_index_long_short(gold_tokens[tokenized_sent_id], regex_tokenized, min_i)
            link_start_ends.append([regex_result[1], curr_i,  curr_i+len(regex_tokenized)])
        sent_links = ["O"]*len(gold_tokens[tokenized_sent_id])
        for link_start_end in link_start_ends:
            sent_links[link_start_end[1]] = "B\t" + link_start_end[0].split("/")[-1]
            for link_tmp_id in range(link_start_end[1]+1, link_start_end[2]):
                sent_links[link_tmp_id] =  "I\t" + link_start_end[0].split("/")[-1]
        links.append(sent_links)

    print("o %s validated!" % filename)
    return links

def write_tsv(tsvfile, tokens, uposes, links):
    with io.open(tsvfile, "w", encoding="utf-8") as f:
        assert len(tokens) == len(uposes) == len(links)
        for sent_id in range(len(tokens)):
            assert len(tokens[sent_id]) == len(uposes[sent_id]) == len(links[sent_id])
            f.write("\n".join(["%s\t%s\t%s" % (tokens[sent_id][idx], uposes[sent_id][idx], links[sent_id][idx])
                              for idx in range(len(tokens[sent_id]))]) + "\n\n")


def compare_tsv_tokens(gold_tsv_file, pred_tsv_file):
    gold_tsv_file_basename = os.path.basename(gold_tsv_file)
    with io.open(gold_tsv_file, "r", encoding="utf-8") as f:
        gold_tsv_sents = [x.split("\n") for x in f.read().strip().split("\n\n")]
    if not os.path.exists(pred_tsv_file):
        print("o Predicted files does not exist -- skipped: ", pred_tsv_file)
        return
    with io.open(pred_tsv_file, "r", encoding="utf-8") as f:
        pred_tsv_sent_str = re.split(r"\n\s*\n", f.read().strip())
        pred_tsv_sents = [x.split("\n") for x in pred_tsv_sent_str]

    for sent_id in range(len(gold_tsv_sents)):
        gold_tsv_sent_toks = [x.split("\t")[0] for x in gold_tsv_sents[sent_id]]
        pred_tsv_sent_toks = [x.split("\t")[0] for x in pred_tsv_sents[sent_id]]
        gold_tsv_sent_toks_highlighted = " ".join([re.sub(".", "_", gold_tsv_sent_toks[idx])
                                          if len(pred_tsv_sent_toks) > idx and
                                             gold_tsv_sent_toks[idx]== pred_tsv_sent_toks[idx]
                                          else gold_tsv_sent_toks[idx]
                                          for idx in range(len(gold_tsv_sent_toks))])
        pred_tsv_sent_toks_highlighted = " ".join([re.sub(".", "_", pred_tsv_sent_toks[idx])
                                          if len(gold_tsv_sent_toks) > idx and
                                             gold_tsv_sent_toks[idx] == pred_tsv_sent_toks[idx]
                                          else pred_tsv_sent_toks[idx]
                                          for idx in range(len(pred_tsv_sent_toks))])
        if gold_tsv_sent_toks_highlighted.replace("_", "").replace(" ", "") != "" \
            or pred_tsv_sent_toks_highlighted.replace("_", "").replace(" ", "") != "":
            print(gold_tsv_file_basename, sent_id, gold_tsv_sent_toks_highlighted, pred_tsv_sent_toks_highlighted, sep="\n", end="\n\n")

    return


if __name__ == '__main__':
    option_generate_stanza_tokenized = False
    option_validate_gold_annotations = True
    option_validate_predicted_tokenization = False


    ### Generate stanza-tokenized version
    if option_generate_stanza_tokenized:
        print("o NOW EXECUTING: option_generate_stanza_tokenized ")
        raw_files = sorted(glob("../raw_document/raw/*.txt"))
        for raw_file in raw_files:
            raw_basename = os.path.basename(raw_file)
            if raw_basename in short_list:
                raw_text = io.open(raw_file, "r", encoding="utf-8").read()
                tokens, uposes = stanza_output(raw_text)
                concatenated_output ="\n".join([" ".join(["|".join(tokens[i][j], uposes[i][j]) for j in len(i)]) for i in len(tokens)])
                with io.open(raw_file.replace("/raw/", "/stanza_tokenized/"), "w", encoding="utf-8") as stanza_f:
                    stanza_f.write(concatenated_output)


    ### Validate human gold annotations
    if option_validate_gold_annotations:
        print("o NOW EXECUTING: option_validate_gold_annotations ")
        gold_markdown_files = sorted(glob("../human_annotation/double/*.txt")) # ** # SZ # LP # adjudicated # last three # threshold_10
        for gold_markdown_file in gold_markdown_files:
            gold_basename = os.path.basename(gold_markdown_file)
            gold_tokenized_file = "../raw_document/gold_tokenized/" + gold_basename
            gold_tokenized_tokens, gold_tokenized_upos = extract_tokenized(gold_tokenized_file)
            entity_links = extract_markdown(gold_markdown_file, gold_tokenized_tokens)
            write_tsv(gold_markdown_file.replace("/human_annotation/", "/logan_gold_tsv/").replace(".txt", ".tsv"), gold_tokenized_tokens, gold_tokenized_upos, entity_links)

    ### Validate predicted annotations using gold TSV
    if option_validate_predicted_tokenization:
        print("o NOW EXECUTING: option_validate_predicted_tokenization ")
        gold_tsv_files = sorted(glob("../logan_gold_tsv/**/*.tsv"))  # ** # SZ # LP # adjudicated
        for gold_tsv_file in gold_tsv_files:
            gold_tsv_basename = os.path.basename(gold_tsv_file)
            # pred_tsv_file = "../machine_annotation/wikifier/tsv/threshold_10/" + gold_tsv_basename  # wikifier threshold = 10
            pred_tsv_file = "../machine_annotation/tagme/tsv/" + gold_tsv_basename  # tagme
            # pred_tsv_file = "../machine_annotation/genre/tsv/" + gold_tsv_basename  # genre
            compare_tsv_tokens(gold_tsv_file, pred_tsv_file)


    print("o Done!")





