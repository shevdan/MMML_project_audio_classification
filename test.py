import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
sys.path.insert(1, os.path.join(sys.path[0], './audioset_tagging_cnn/pytorch'))
sys.path.insert(2, os.path.join(sys.path[0], './audioset_tagging_cnn/utils'))

from inference import audio_tagging

# CHECKPOINT_PATH="./transformed3/checkpoints_1/main/1500_iterations.pth"
# STATISTICS_PATH = 'transformed3'
CHECKPOINT_PATH = "./dataset_10_1/checkpoints_1/main/450_iterations.pth"
STATISTICS_PATH = 'dataset_10_1'
DATASET_DIR = "./dataset_10"
MODEL_TYPE="Cnn14"

class Params:

    def __init__(self, model_type, checkpoint_path, audio_path, cuda):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.audio_path = audio_path
        self.cuda = cuda

        self.fmax = 14000
        self.fmin = 50
        self.mel_bins = 64
        self.hop_size = 320
        self.window_size = 1024
        self.sample_rate = 22050
        self.verbose = False


def test(mode, verbose):
    if mode == 'test':
        path = f'{DATASET_DIR}/audios/test'
        true_path = f'{DATASET_DIR}/metadata/test.csv'
    else:
        path = f'{DATASET_DIR}/audios/eval'
        true_path = f'{DATASET_DIR}/metadata/eval.csv'

    true_data = pd.read_csv(true_path, skiprows=2)

    results = []
    y_true = []
    for file in os.listdir(path):
        
        print(os.fsdecode(file))
        true_label = true_data[true_data['# YTID'] == os.fsdecode(file)[:-4]][' positive_labels'].tolist()[0]
        y_true.append(true_label[2:-1])

        if verbose:
            print(f'Test for {os.fsdecode(file)}:')
            os.system(f'python pytorch/inference.py audio_tagging \
            --model_type={MODEL_TYPE} \
            --checkpoint_path={CHECKPOINT_PATH} \
            --audio_path={path + "/" + os.fsdecode(file)} \
            --cuda')
        else:
            args = Params(MODEL_TYPE, CHECKPOINT_PATH, path + '/' + os.fsdecode(file), False)
            clipwise_output, labels, codes = audio_tagging(args)
            results.append([clipwise_output.tolist(), codes])
    return results, y_true
            
        
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)  # type: ignore

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)

def create_confusion_matrix(mode):
    data, y_true = test(mode, False)
    y_predicted = []

    for vals, codes in data:
        max_idx = vals.index(max(vals))
        y_predicted.append(codes[max_idx])
    print(y_true)
    print(y_predicted)
    lst = []
    lst.extend(y_predicted)
    lst.extend(y_true)
    all_classes = list(set(lst))
    # print(all_classes)
    matrix = confusion_matrix(y_true, y_predicted, labels=all_classes)
    # print(matrix.size)
    img = make_confusion_matrix(matrix, categories=all_classes)
    
    plt.savefig(f'{STATISTICS_PATH}/confusion_matrix.png')
    plt.show()
    


if __name__ == '__main__':
    # test('eval', False)
    create_confusion_matrix('eval')
    