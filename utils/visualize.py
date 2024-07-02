import pickle
from collections import OrderedDict, defaultdict
from evaluator import compare_annos, scorer_wrapper
from sklearn.metrics import accuracy_score, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter


def split_genre(annos):
    sub_annos = {}
    for key, value in annos.items():
        sub_key = key.split("_")[1]
        if sub_key not in sub_annos:
            sub_annos[sub_key] = {}
        sub_annos[sub_key][key] = value
    return sub_annos

def plot_threshold(ax, wikifier_thres_score_dict=None, tagme_thres_score_dict=None, save_file=None):
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
    colors = {"wikifier": "green", "tagme": "b"}
    shapes = {"P": "-", "R": "-", "F1": "-"}
    markers = {"P": "o", "R": "s", "F1": "D"}  
    line_dict = {}
    plot_thres_range = [0.1 * x for x in thres_range_dict["wikifier"]]
    for plot_thres_dict_key in plot_thres_dict.keys():
        if plot_thres_dict_key.split("_")[0] == "wikifier": label = "Wikifier" 
        else: label = "TagMe" 
        color = colors[plot_thres_dict_key.split("_")[0]]
        shape = shapes[plot_thres_dict_key.split("_")[1]]
        marker = markers[plot_thres_dict_key.split("_")[1]]  
        ax.plot(plot_thres_range, plot_thres_dict[plot_thres_dict_key], linestyle=shape, color=color, marker=marker, label=label+"_"+plot_thres_dict_key.split("_")[1])  
    ax.legend(loc='upper left')


    if save_file:
        plt.savefig(f"../figs/{save_file}.tif")

def frequency_distribution(lst):
    modified_lst = [num if num <= 4 else 4 for num in lst]
    freq_dist = dict(Counter(modified_lst))
    return freq_dist

def plot_multiple_histograms(data, title):
    bar_width = 0.15
    x = np.arange(1, 5)  # 1, 2, 3, and 4 (including the >=4 category)
    distributions = [frequency_distribution(lst) for lst in data.values()]
    labels = data.keys()

    for i, (freq_dist, label) in enumerate(zip(distributions, labels)):
        if label=="academic": label="aca."
        total_count = sum(freq_dist.values())
        counts = [freq_dist.get(num, 0) for num in range(1, 4)] + [freq_dist.get(4, 0)]
        frequencies = [count / total_count * 100 for count in counts]  
        positions = x + (i - len(distributions) / 2) * bar_width
        plt.bar(positions, frequencies, bar_width, label=label)

        for j in range(len(counts)):
            plt.text(positions[j], frequencies[j]+1, f'{counts[j]} | {frequencies[j]:.2f}%', ha='center', va='bottom', rotation=90, fontsize=11)
            
    plt.xlabel('Entity Length (token)')
    plt.ylabel('Frequency | Proportion')
    plt.title(title)
    plt.xticks(ticks=x, labels=['1', '2', '3', '>=4'])
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 91, 10), [f'{i}%' for i in np.arange(0, 91, 10)])
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"../figs/NC_length_distribution.png")
    plt.show()

if __name__ == "__main__":
    with open('../annotations.pkl', 'rb') as file:
        annotations = pickle.load(file)

    iaa_result = dict()
    machine_result = dict()

    c_only = [file for file in annotations.keys() if file.endswith("_climate_text")]
    nc_only = [file for file in c_only if file.endswith("_nominal_entity_climate_text")]
    c_only = list(set(c_only) - set(nc_only))
    n_only = [file for file in annotations.keys() if file.endswith("_nominal_entity")]
    n_only.sort()
    c_only.sort()
    nc_only.sort()

    ### evaluation without any filter

    w_evaluation = dict()
    wikifier_w_dict = defaultdict(dict)
    tagme_w_dict = defaultdict(dict)

    without_filter = ['gold', 'wikifier_10', 'wikifier_09', 'wikifier_08', 'wikifier_07', 'wikifier_06', 'wikifier_05', 'wikifier_04', 'wikifier_03', 'wikifier_02', 'wikifier_01', 
    'tagme_00', 'tagme_01', 'tagme_02', 'tagme_03', 'tagme_04', 'tagme_05', 'tagme_06', 'tagme_07', 'tagme_08', 'tagme_09', 'genre']

    gold_annos = "gold"
    for pred_annos in without_filter:
        if pred_annos.split("_")[0] in ["genre", "tagme", "wikifier"]:
            w_evaluation[f"{pred_annos}"] = scorer_wrapper(annotations[gold_annos], annotations[pred_annos])
            if pred_annos.split("_")[0]=="tagme":
                x = int(pred_annos.split("_")[1])
                tagme_w_dict[x]["typed_entity_precision"] = w_evaluation[f"{pred_annos}"]["typed_entity_precision"]
                tagme_w_dict[x]["typed_entity_recall"] = w_evaluation[f"{pred_annos}"]["typed_entity_recall"]
                tagme_w_dict[x]["typed_entity_f1"] = w_evaluation[f"{pred_annos}"]["typed_entity_f1"]
            if pred_annos.split("_")[0]=="wikifier":
                x = int(pred_annos.split("_")[1])
                wikifier_w_dict[x]["typed_entity_precision"] = w_evaluation[f"{pred_annos}"]["typed_entity_precision"]
                wikifier_w_dict[x]["typed_entity_recall"] = w_evaluation[f"{pred_annos}"]["typed_entity_recall"]
                wikifier_w_dict[x]["typed_entity_f1"] = w_evaluation[f"{pred_annos}"]["typed_entity_f1"]

    w_evaluation = OrderedDict(sorted(w_evaluation.items()))
    df = pd.DataFrame.from_dict(w_evaluation, orient='index')
    pd.options.display.float_format = '{:.2f}'.format
    df = df.applymap(lambda x: f"{x:.2f}")
    df.to_excel('../results/machine_without_filter.xlsx')

    ### for 5 genres

    genre_evaluations = defaultdict(dict)

    sub_gold = split_genre(annotations["gold"])
        
    for pred_annos in without_filter:
        if pred_annos.split("_")[0] == "genre":
            sub_genre = split_genre(annotations[pred_annos])
            for g in sub_genre:
                genre_evaluations[f"{pred_annos}"][g] = scorer_wrapper(sub_gold[g], sub_genre[g])
            
        if pred_annos.split("_")[0] == "tagme":
            x = int(pred_annos.split("_")[1])
            if x == 0:
                sub_tagme = split_genre(annotations[pred_annos])
                for g in sub_tagme:
                    genre_evaluations[f"{pred_annos}"][g] = scorer_wrapper(sub_gold[g], sub_tagme[g])
                
        if pred_annos.split("_")[0] == "wikifier":
            x = int(pred_annos.split("_")[1])
            if x == 10:
                sub_wiki = split_genre(annotations[pred_annos])
                for g in sub_wiki:
                    genre_evaluations[f"{pred_annos}"][g] = scorer_wrapper(sub_gold[g], sub_wiki[g])

    writer = pd.ExcelWriter('../results/machine_all_5_genre.xlsx')

    for key, value in genre_evaluations.items():
        df = pd.DataFrame(value)
        df = df.round(2)
        df.to_excel(writer, sheet_name=key, index=True)

    writer.close()