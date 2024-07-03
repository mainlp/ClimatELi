import os
import urllib.parse, urllib.request, json

def CallWikifier(text, lang="en", threshold=1):
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "dvabwuiiztnbxarrbwdhetvzafxwxp"),
        ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
        ])
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout = 60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))

    entities = []

    for ann in response["annotations"]:
        for mention in ann["support"]:
            # print(tokens[mention['wFrom']], mention['wFrom'])
            
            entities.append((mention['chFrom'], \
                             mention['chTo']+1, \
                             text[mention['chFrom']:mention['chTo']+1], \
                             ann["url"].split("/")[-1]))
    # Output the annotations
    return sorted(entities, key=lambda x: x[1])

def annotation_extractor(dir_path, file_name, threshold=1):

    file_path = dir_path + file_name

    document = ''

    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        for sentence in file:
            for pair in sentence.strip().split():
                if pair:
                    document += pair.split("|")[0] + " "
                    # tokens.append(pair.split("|")[0])
            document += "\n\n"

    entities = CallWikifier(document, threshold=threshold)
    print(entities)

    deduplicate_entities = []
    current_entity = (-1, -1, -1) # start, end, span
    for s, e, m, t in entities:
        # print(s, e, m, t)
        if s >= current_entity[1]:
            # print(1)
            deduplicate_entities.append((s, e, m, t))
            current_entity = (s, e, e-s)
        else:
            if e-s > current_entity[2]:
                # print(2)
                deduplicate_entities.pop()
                deduplicate_entities.append((s, e, m, t))
                current_entity = (s, e, e-s)
            
    return deduplicate_entities, document

def alignment_extraction(entities, document, tsv_dir, txt_file):
    data = []  # ("token1", "POS", "B/I/O", "Title1")
    start = 0
    if entities:
        s, e, _, t = entities.pop(0)
        flag = False

        with open(dir_path + txt_file, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                pairs = line.split()
                for pair in pairs:
                    token = pair.split("|")[0]
                    pos_tag = pair.split("|")[1]
        
                    if start < s:
                        data.append(f"{token}\t{pos_tag}\tO\n")
                        flag = False
                        
                    elif start == s:
                        print("B", document[s:e], s, start, token)
                        data.append(f"{token}\t{pos_tag}\tB\t{t}\n")
                        flag = True
                        if start+len(token) >= e:
                            try:
                                s, e, _, t = entities.pop(0)
                                flag = False
                            except:
                                pass

                    elif s < start < e:
                        if flag:
                            print("I", document[s:e], s, start, token)
                            data.append(f"{token}\t{pos_tag}\tI\t{t}\n")
                            if start+len(token) >= e:
                                try:
                                    s, e, _, t = entities.pop(0)
                                    flag = False
                                except:
                                    pass
                        else:
                            data.append(f"{token}\t{pos_tag}\tO\n")
                            
                    elif start >= e:
                        try:
                            s, e, _, t = entities.pop(0)
                            flag = False
                            data.append(f"{token}\t{pos_tag}\tO\n")
                        except:
                            data.append(f"{token}\t{pos_tag}\tO\n")

                            
                    start += len(token) + 1
                start += 2              
                data.append("\n")
    
    else:
        with open(dir_path+txt_file, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                pairs = line.split()
                for pair in pairs:
                    token = pair.split("|")[0]
                    pos_tag = pair.split("|")[1]
                    data.append(f"{token}\t{pos_tag}\tO\n")
                data.append("\n")

    with open(tsv_dir+txt_file[:-4]+".tsv", 'w', newline='', encoding='utf-8') as tsvfile:
        for row in data:
            tsvfile.write(row)
    print(f"{txt_file[:-4]} done!")


if __name__ == "__main__":
    dir_path = "../dataset/gold_tokenized/"
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

    threshold_map = { i/10: "{:0>2}".format(i) for i in range(1, 11) } # different thresholds

    for k, v in sorted(threshold_map.items(), reverse=True):
        tsv_dir = f'../machine_annotation/wikifier/tsv/threshold_{v}/'
        if not os.path.exists(tsv_dir):
            os.makedirs(tsv_dir)
        
        document = ""

        for txt_file in txt_files:
            entities, document = annotation_extractor(dir_path, txt_file, threshold=k)
            alignment_extraction(entities, document, tsv_dir, txt_file)
            

