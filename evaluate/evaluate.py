import ast
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import sys
import os

from sklearn import metrics
from scipy.stats import norm


sys.path.append(os.getcwd())

import config as cfg
from utils.utils import latin_to_common, latin_to_dutch
from analyzers import Analyzer, BirdNETAnalyzer, NaturalisAnalyzer, OwnAnalyzer, PerchAnalyzer, LDAAnalyzer, AquilaAnalyzer


def get_sample_targets(classes):
    if cfg.TEST_SOUNDSCAPES:
        df = pd.read_csv(cfg.DATASET_DIR + "test/labels.csv")
        labels = df['labels'].values
        labels = [ast.literal_eval(label) for label in labels]
    else:
        df = pd.read_csv(cfg.DATASET_DIR + "val/labels.csv")
        
        labels = []
        for i, row in df.iterrows():
            rec_primary = row['primary']
            rec_secondary = ast.literal_eval(row['secondary'])
            labels.append([rec_primary] + rec_secondary)

    n_samples = len(labels)
    n_classes = len(classes)

    sample_targets = np.zeros(shape=(n_samples, n_classes))
    for i in range(n_samples):
        for label in labels[i]:
            if label in classes:
                class_index = classes.index(label)
                sample_targets[i][class_index] = 1

    return sample_targets


def print_mean_aucs(analyzers: list[Analyzer], classes: list[str]):
    sample_targets = get_sample_targets(classes)

    for analyzer in analyzers:
        sample_scores = analyzer.sample_scores

        aucs = []
        for i in range(len(classes)):
            class_targets = sample_targets[:,i]
            class_scores = sample_scores[:,i]

            if np.sum(class_targets) > 0:
                auc = metrics.roc_auc_score(class_targets, class_scores)
                aucs.append(auc)

        print(type(analyzer))
        print(sum(aucs) / len(aucs))


def plot_binary_det_curves(analyzers: list[Analyzer], classes: list[str], models: list[str]):
    sample_targets = get_sample_targets(classes).flatten()
    p_target = sum(sample_targets) / len(sample_targets)

    fig, ax = plt.subplots()
    for i, analyzer in enumerate(analyzers):
        sample_scores = analyzer.sample_scores.flatten()

        fpr, fnr, thresholds = metrics.det_curve(sample_targets, sample_scores)
        ax.plot(norm.ppf(fpr), norm.ppf(fnr), label=models[i])

        cfp = (1-p_target) * fpr * 1
        cfn =   p_target   * fnr * 5
        
        # minimum detection cost - based on detection cost function (DCF)
        #mdc_index = np.nanargmin(cfp + cfn)#np.absolute((cfp - cfn)))
        #ax.plot(norm.ppf(fpr)[mdc_index], norm.ppf(fnr)[mdc_index],'x')#'ro') 

        # threshold at minimum detection cost
        #print(models[i], thresholds[edc_index])
        
    intervals = np.array([0.5, 1, 2, 5, 10, 20, 40, 60])
    labels = list(map(lambda x: int(x) if x.is_integer() else x, intervals))
    
    ax.set_xlabel("false alarm probability (%)")
    ax.set_ylabel("miss probability (%)")

    min = -2.7; max = 0.5
    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.plot([min, max], [min, max], linestyle='dotted', color='darkgray', linewidth=1)

    ax.grid(linestyle='dotted')
    ax.set_xticks(ticks=norm.ppf(intervals / 100.), labels=labels)
    ax.set_yticks(ticks=norm.ppf(intervals / 100.), labels=labels)

    ax.set_title("DET plot")
    ax.set_box_aspect(1)
    ax.legend()

    fig.set_figwidth(4)
    fig.set_figheight(4)
    fig.set_dpi(200)
    fig.tight_layout()
    fig.show()


def plot_roc_curves_class(ax: Axes, analyzers: list[Analyzer], models: list[str], class_targets: np.ndarray, class_index: int, class_name: str):
    for i, analyzer in enumerate(analyzers):
        class_scores = analyzer.sample_scores[:,class_index]

        fpr, tpr, thresholds = metrics.roc_curve(class_targets, class_scores)
        auc = metrics.roc_auc_score(class_targets, class_scores)

        ax.plot(fpr, tpr, label="{}: {}%".format(models[i], round(auc * 100)))

    ax.set_xlabel("false positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle='dotted')
    ax.legend()
    ax.set_title(latin_to_common(class_name))


def plot_roc_curves_classes(analyzers: list[Analyzer], classes, models, classes_to_plot: list[str]):
    sample_targets = get_sample_targets(classes)
    p_targets = np.sum(sample_targets, axis=0) / sample_targets.shape[0]
    n_classes = len(classes)
    
    i = 0
    fig, axs = plt.subplots(1, 4)
    for class_index in range(n_classes):
        if p_targets[class_index] > 0.4 and p_targets[class_index] < 0.6:
            class_targets = sample_targets[:,class_index]
            plot_roc_curves_class(axs[i], analyzers, models, class_targets, class_index, classes[class_index])

            if i == 0:
                axs[i].set_ylabel("true positive rate")
            i += 1
    
    plt.show()


def xor(a: bool, b: bool) -> bool:
    return (a or b) and not (a and b)


def plot_det_curve_one_vs_one(analyzers: list[Analyzer], classes, models, class1, class2):
    sample_targets = get_sample_targets(classes)
    n_samples = sample_targets.shape[0]
    class1_targets = sample_targets[:,classes.index(class1)]
    class2_targets = sample_targets[:,classes.index(class2)]
    samples_to_consider = [xor(class1_targets[i] == 1, class2_targets[i] == 1) for i in range(n_samples)]

    for analyzer in analyzers:
        class1_scores = analyzer.sample_scores[:,classes.index(class1)][samples_to_consider]
        class2_scores = analyzer.sample_scores[:,classes.index(class2)][samples_to_consider]
        class_scores = (class1_scores-class2_scores) / 2.0 + 0.5
        
        class_targets = class1_targets[samples_to_consider]

        auc = metrics.roc_auc_score(class_targets, class_scores)
        print(auc)


def get_aucs_class(analyzers: list[Analyzer], models, class_targets, class_index, class_name):
    result = {
        "Scientific name": class_name,
        "Common name": latin_to_common(class_name),
        "targets": sum(class_targets)
    }
    
    for i, analyzer in enumerate(analyzers):
        class_scores = analyzer.sample_scores[:,class_index]

        auc = metrics.roc_auc_score(class_targets, class_scores)
        result[models[i]] = round(auc, 2)

    return result    


def store_result_classes(analyzers, classes, models):
    sample_targets = get_sample_targets(classes)
    p_targets = np.sum(sample_targets, axis=0) / sample_targets.shape[0]
    n_classes = len(classes)
    
    results = []
    for class_index in range(n_classes):
        if p_targets[class_index] > 0:
            class_targets = sample_targets[:,class_index]

            result = get_aucs_class(analyzers, models, class_targets, class_index, classes[class_index])
            results.append(result)

    df = pd.DataFrame.from_dict(results)
    df.to_excel("results.xlsx")            



def evaluate_samples(threshold, sample_scores: np.ndarray, sample_targets: np.ndarray):
    predictions = np.where(sample_scores >= threshold, 1, 0)

    tps = np.sum(sample_targets * predictions)
    fps = np.sum(predictions) - tps
    fns = np.sum(sample_targets) - tps

    return tps, fps, fns


def print_n_errors(analyzers: list[Analyzer], classes, models, thresholds):
    sample_targets = get_sample_targets(classes)

    for i, analyzer in enumerate(analyzers):
        sample_scores = analyzer.sample_scores

        tps, fps, fns = evaluate_samples(thresholds[i], sample_scores, sample_targets)
        print(models[i], tps + fps, tps, fps, fns)


def store_n_errors_per_class(analyzers: list[Analyzer], classes, models, thresholds):
    sample_targets = get_sample_targets(classes)
    n_classes = len(classes)
    n_samples = sample_targets.shape[0]

    results = [{ "scientific-name": classes[i], "common-name": latin_to_dutch(classes[i]), "targets": sum(sample_targets[:,i]) } for i in range(n_classes)]

    for model_index, analyzer in enumerate(analyzers):
        model = models[model_index]
        sample_scores = analyzer.sample_scores

        for class_index in range(n_classes):
            class_targets = sample_targets[:,class_index]
            class_scores = sample_scores[:,class_index]

            if np.sum(class_targets) > 0:
                auc = metrics.roc_auc_score(class_targets, class_scores)
                results[class_index]["auc {}".format(model)] = auc

                fpr, fnr, _ = metrics.det_curve(class_targets, class_scores)
                eer_index = np.nanargmin(np.absolute((fpr - fnr)))

            _, fps, fns = evaluate_samples(thresholds[model_index], class_scores, class_targets)
            results[class_index]["fps {}".format(model)] = fps
            results[class_index]["fns {}".format(model)] = fns
            results[class_index]["acc {}".format(model)] = (n_samples - fps - fns) / n_samples
            results[class_index]["threshold {}".format(model)] = thresholds[model_index]
                

    df = pd.DataFrame.from_dict(results)
    
    threshold_cols = []
    auc_cols = []; acc_cols = []
    fps_cols = []; fns_cols = []
    for model in models:
        threshold_cols.append("threshold {}".format(model))
        auc_cols.append("auc {}".format(model))
        acc_cols.append("acc {}".format(model))
        fps_cols.append("fps {}".format(model))
        fns_cols.append("fns {}".format(model))

    cols = df.columns.tolist()[:3] + threshold_cols + auc_cols + acc_cols + fps_cols + fns_cols 
    df = df[cols]
    
    df.to_excel("errors_at_t.xlsx")        


def plot_aucs_classes(analyzers: list[Analyzer], classes, models):
    sample_targets = get_sample_targets(classes)

    classes_to_plot = {}
    for class_index, class_name in enumerate((classes)):
        class_targets = sample_targets[:,class_index]

        if abs(35 - sum(class_targets)) <= 28:
            classes_to_plot[latin_to_common(class_name)] = class_index

    data = np.empty(shape=(len(analyzers), len(classes_to_plot)), dtype=np.float32)
    for model_index, analyzer in enumerate(analyzers):
        
        for i in range(len(classes_to_plot)):
            class_index = list(classes_to_plot.values())[i]
            class_targets = sample_targets[:,[class_index]]
            class_scores = analyzer.sample_scores[:,class_index]

            auc = metrics.roc_auc_score(class_targets, class_scores)
            data[model_index][i] = auc

    mean_aucs = np.mean(data, axis=0)
    order = np.argsort(mean_aucs)[::-1]

    class_names = [list(classes_to_plot.keys())[i] for i in order]
    plt.axhline(y=0.5, color='r', linestyle='dotted')
    plt.plot(class_names, mean_aucs[order], color='darkgray')

    for i in range(len(models)):
        plt.plot(class_names, data[i][order], linestyle='--', marker='o', label=models[i])
        
    plt.xticks(rotation=90)
    plt.yticks(np.arange(start=0.45, stop=1.04, step=0.05))
    plt.legend()
    plt.grid(linestyle='dotted')
    plt.ylabel("Area under ROC curve")
    plt.title("Model performance for the 37 most frequent species")

    plt.show()
    

            
def print_accuracies(analyzers: list[Analyzer], classes, models):
    sample_targets = get_sample_targets(classes)

    thresholds = [0.1921737, 0.34639397, 0.2389975190162658, 0.3335171341896057]
    for model_index, analyzer in enumerate(analyzers):
        sample_predictions = np.where(analyzer.sample_scores >= thresholds[model_index], 1, 0)

        mean_acc = []
        for class_index in range(len(classes)):
            class_targets = sample_targets[:,class_index]

            if np.sum(class_targets) > 0:
                class_predictions = sample_predictions[:,class_index]
                class_acc = np.sum(np.where(class_predictions == class_targets, 1, 0)) / len(class_predictions)
                mean_acc.append(class_acc)
        
        print(models[model_index])
        print(sum(mean_acc) / len(mean_acc))


def evaluate_dataset():
    species_list = pd.read_csv(cfg.SPECIES_FILE_PATH)
    classes = list(species_list['latin_name'].values)
    #classes.append("Noise")
    print("Nr classes:", len(classes))

    #aquila = AquilaAnalyzer(classes, species_list)
    birdnet = BirdNETAnalyzer(classes, species_list)
    perch = PerchAnalyzer(classes, species_list)
    #naturalis = NaturalisAnalyzer(classes, species_list)
    sovon = OwnAnalyzer(classes, species_list)
    sovon2 = OwnAnalyzer(classes, species_list, folder="nlc-predictions")

    models = ['GBVC', 'NLC', 'NLC-NoSec']#['BirdNET', 'GBVC', 'AvesEcho', 'Aquila']
    analyzers = [perch, sovon2, sovon]#[birdnet, perch, naturalis, aquila]
    
    print_mean_aucs(analyzers, classes)
    #print_accuracies(analyzers, classes, models)
    plot_aucs_classes(analyzers, classes, models)
    plot_binary_det_curves(analyzers, classes, models)
    #store_n_errors_per_class(analyzers, classes, models, thresholds)
    #store_result_classes(analyzers, classes, models)
    #plot_det_curve_one_vs_one(analyzers, classes, models, "Acanthis flammea", "Acanthis cabaret")
    plot_roc_curves_classes(analyzers, classes, models, ["Fringilla montifringilla", "Delichon urbicum", "Erithacus rubecula", "Sturnus vulgaris"])
    #print_n_errors(analyzers, classes, models, thresholds)


if __name__ == '__main__':
    evaluate_dataset()