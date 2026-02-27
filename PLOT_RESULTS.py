import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import seaborn as sn

def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out

def Plot_Activation():
    Eval = np.load('Eval_Act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']

    learn = [1, 2, 3, 4, 5]
    X = np.arange(5)
    Graph_Term = [0,1,2, 3, 4, 5,6,7,8,9]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
        plt.plot(learn, Graph[:, 0], '-.', color='#65fe08', linewidth=5, marker='*', markerfacecolor='blue',
                 markersize=10,
                 label="DMO-O-DNN")
        plt.plot(learn, Graph[:, 1], '-.', color='#4e0550', linewidth=5, marker='*', markerfacecolor='red',
                 markersize=10,
                 label="GSOA-O-DNN")
        plt.plot(learn, Graph[:, 2], '-.', color='#f70ffa', linewidth=5, marker='*', markerfacecolor='green',
                 markersize=10,
                 label="LOA-O-DNN")
        plt.plot(learn, Graph[:, 3], '-.', color='#a8a495', linewidth=5, marker='*', markerfacecolor='yellow',
                 markersize=10,
                 label="NRO-O-DNN")
        plt.plot(learn, Graph[:, 4], '-.', color='#004577', linewidth=5, marker='*', markerfacecolor='cyan',
                 markersize=10,
                 label="E-NRO-O-DNN")
        plt.xlabel('Activation Function', fontsize=12)
        plt.ylabel(Terms[Graph_Term[j]], fontsize=12)
        plt.xticks(X + 1, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
        plt.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),
                   ncol=3, fancybox=True, shadow=True, fontsize='12')
        path1 = "./Cit_Results/line_Act_%s.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

    eval = np.load('Eval_Act.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0,1,2, 3, 4, 5,6,7,8,9]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, j + 4]
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.7, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='#f97306', edgecolor='k', width=0.15, hatch="+", label="DRL")
        ax.bar(X + 0.15, Graph[:, 6], color='#f10c45', edgecolor='k', width=0.15, hatch="+", label="DTWN")
        ax.bar(X + 0.30, Graph[:, 7], color='#ddd618', edgecolor='k', width=0.15, hatch="+", label="CHAINFL")
        ax.bar(X + 0.45, Graph[:, 8], color='#6ba353', edgecolor='k', width=0.15, hatch="+", label="DNN")
        ax.bar(X + 0.60, Graph[:, 9], color='b', edgecolor='k', width=0.15, hatch="*", label="E-NRO-O-DNN")
        plt.xticks(X + 0.15, (
            'Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'), fontsize=12)
        plt.ylabel(Terms[Graph_Term[j]], fontsize=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=12, fancybox=True, shadow=True)
        path1 = "./Cit_Results/Learn_%s_bar_Act.png" % (Terms[Graph_Term[j]])
        plt.xlabel('Activation Function', fontsize=12)
        plt.savefig(path1)
        plt.show()

def Confusion_matrix():
    # Confusion Matrix
    Eval = np.load('Eval_Act.npy', allow_pickle=True)
    value = Eval[3, 4, :5]
    val = np.asarray([0, 1, 1])
    data = {'y_Actual': [val.ravel()],
            'y_Predicted': [np.asarray(val).ravel()]
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'], colnames=['Predicted'])
    value = value.astype('int')

    confusion_matrix.values[0, 0] = value[1]
    confusion_matrix.values[0, 1] = value[3]
    confusion_matrix.values[1, 0] = value[2]
    confusion_matrix.values[1, 1] = value[0]

    sn.heatmap(confusion_matrix, annot=True).set(title='Accuracy = ' + str(Eval[4, 4, 4] )[:5] + '%')
    sn.plotting_context()
    path1 = './Cit_Results/Confusion_smart.png'
    plt.savefig(path1)
    plt.show()
def Plot_Fitness():
    conv = np.load('Fitness.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['DMO-O-DNN', 'GSOA-O-DNN', 'LOA-O-DNN', 'NRO-O-DNN', 'E-NRO-O-DNN']

    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("ALGORITHMS", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print(
        '------------------------------------------------------Statistical Analysis--------------------------------------------------')
    print(Table)

    iteration = np.arange(conv.shape[1])
    plt.plot(iteration, conv[0, :], color='r', linewidth=5, marker='>', markerfacecolor='blue', markersize=8,
             label="DMO-O-DNN")
    plt.plot(iteration, conv[1, :], color='g', linewidth=5, marker='>', markerfacecolor='red', markersize=8,
             label="GSOA-O-DNN")
    plt.plot(iteration, conv[2, :], color='b', linewidth=5, marker='>', markerfacecolor='green', markersize=8,
             label="LOA-O-DNN")
    plt.plot(iteration, conv[3, :], color='m', linewidth=5, marker='>', markerfacecolor='yellow', markersize=8,
             label="NRO-O-DNN")
    plt.plot(iteration, conv[4, :], color='k', linewidth=5, marker='>', markerfacecolor='cyan', markersize=8,
             label="E-NRO-O-DNN")
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost Function', fontsize=12)
    plt.legend(loc=1, fontsize=12)
    path1 = "./Cit_Results/conv.png"
    plt.savefig(path1)
    plt.show()


def Plot_Kfold():
    Eval = np.load('Eval_kFOLD.npy', allow_pickle=True)
    Terms = np.asarray(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC'])
    Algorithm = ['TERMS', 'DMO-O-DNN', 'GSOA-O-DNNA', 'LOA-O-DNN', 'NRO-O-DNN',
                 'E-NRO-O-DNN']
    Classifier = ['TERMS', 'DRL', 'DTWN', 'CHAINFL', 'DNN', 'E-NRO-O-DNN']

    value = Eval[:, :, 4:14]
    Graph_Term = np.array([0]).astype(int)
    Graph_Term_Array = np.array(Graph_Term)
    variation = ['1', '2', '3', '4', '5']
    # value = Eval[:, :, 4:]
    Table = PrettyTable()
    Table.add_column(Algorithm[0], variation[0:])
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term])
    print('---------------------------------------------------KFold - Algorithm Comparison -',
          Terms[Graph_Term],
          '--------------------------------------------------')
    print(Table)

    Table = PrettyTable()
    Table.add_column(Classifier[0], variation[0:])
    for j in range(len(Classifier) - 1):
        Table.add_column(Classifier[j + 1], value[:, j+5, Graph_Term])
    print('---------------------------------------------------KFold - Classifier Comparison -',
          Terms[Graph_Term],
          '--------------------------------------------------')
    print(Table)

# Table = PrettyTable()
#     Table.add_column(Classifier[0], Terms)
#     for j in range(len(Classifier) - 1):
#         Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
#     print('--------------------------------------------------KFold', 'Classifier Comparison - ',
#           '--------------------------------------------------')
#     print(Table)


def Plot_federated_learning():
    Eval = np.load('Federated_Learning.npy', allow_pickle=True)
    Terms = ['Attack Rate', 'Leakage Rate', 'Anonymization Effectiveness', 'Authentication Success Rate', 'Detection Rate']

    learn = [1, 2, 3, 4, 5]
    X = np.arange(5)
    Graph_Term = [0,1,2, 3, 4]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
        for k in range(Eval.shape[0]):
            for l in range(Eval.shape[1]):
                Graph[k, l] = Eval[k, l, j]
        plt.plot(learn, Graph[:, 0], '-.', color='#65fe08', linewidth=7, marker='*', markerfacecolor='blue',markersize=10,
                 label="DMO-O-DNN")
        plt.plot(learn, Graph[:, 1], '-.', color='#4e0550', linewidth=7, marker='*', markerfacecolor='red',
                 markersize=10,
                 label="GSOA-O-DNN")
        plt.plot(learn, Graph[:, 2], '-.', color='#f70ffa', linewidth=7, marker='*', markerfacecolor='green',
                 markersize=10,
                 label="LOA-O-DNN")
        plt.plot(learn, Graph[:, 3], '-.', color='#a8a495', linewidth=7, marker='*', markerfacecolor='yellow',
                 markersize=10,
                 label="NRO-O-DNN")
        plt.plot(learn, Graph[:, 4], '-.', color='#004577', linewidth=7, marker='*', markerfacecolor='cyan',
                 markersize=10,
                 label="E-NRO-O-DNN")
        plt.xlabel('Activation Function', fontsize=12)
        plt.ylabel(Terms[j], fontsize=12)
        plt.xticks(X + 1, ('Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'))
        plt.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),
                   ncol=3, fancybox=True, shadow=True, fontsize='12')
        path1 = "./Cit_Results/Federated_Act_%s.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

    eval = np.load('Federated_Learning.npy', allow_pickle=True)
    Terms = ['Attack Rate', 'Leakage Rate', 'Anonymization Effectiveness', 'Authentication Success Rate',
             'Detection Rate']
    Graph_Term = [0,1,2, 3, 4]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, j ]
        fig = plt.figure()
        ax = fig.add_axes([0.12, 0.1, 0.7, 0.8])
        X = np.arange(5)
        ax.bar(X + 0.00, Graph[:, 5], color='#f97306', edgecolor='k', width=0.15, hatch="+", label="DRL")
        ax.bar(X + 0.15, Graph[:, 6], color='#f10c45', edgecolor='k', width=0.15, hatch="+", label="DTWN")
        ax.bar(X + 0.30, Graph[:, 7], color='#ddd618', edgecolor='k', width=0.15, hatch="+", label="CHAINFL")
        ax.bar(X + 0.45, Graph[:, 8], color='#6ba353', edgecolor='k', width=0.15, hatch="+", label="DNN")
        ax.bar(X + 0.60, Graph[:, 9], color='b', edgecolor='k', width=0.15, hatch="*", label="E-NRO-O-DNN")
        plt.xticks(X + 0.15, (
            'Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'), fontsize=12)
        plt.ylabel(Terms[j], fontsize=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=12, fancybox=True, shadow=True)
        path1 = "./Cit_Results/Federated_%s_bar_Act.png" % (Terms[Graph_Term[j]])
        plt.xlabel('Activation Function', fontsize=12)
        plt.savefig(path1)
        plt.show()
if __name__ == '__main__':
    Plot_Activation()
    Plot_Fitness()
    Confusion_matrix()
    Plot_Kfold()
    Plot_federated_learning()
