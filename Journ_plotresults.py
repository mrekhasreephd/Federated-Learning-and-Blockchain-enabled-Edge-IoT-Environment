import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def Plot_Activation():
    Eval = np.load('Eval_Act.npy', allow_pickle=True)
    Terms = np.asarray(['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC'])
    Algorithm = ['TERMS', 'DMO-O-DNN', 'GSOA-O-DNNA', 'LOA-O-DNN', 'NRO-O-DNN',
                 'E-NRO-O-DNN']
    Classifier = ['TERMS', 'DRL', 'DTWN', 'CHAINFL', 'DNN', 'E-NRO-O-DNN']

    value = Eval[:, :, 4:14]
    for n in range(10):
        Graph_Term = np.array([n]).astype(int)
        Graph_Term_Array = np.array(Graph_Term)
        variation = ['Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid']
        # value = Eval[:, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], variation[0:])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term])
        print('--------------------------------------------------- - Algorithm Comparison -',
              Terms[Graph_Term],
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], variation[0:])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, j+5, Graph_Term])
        print('--------------------------------------------------- - Classifier Comparison -',
              Terms[Graph_Term],
              '--------------------------------------------------')
        print(Table)


def Plot_federated_learning():
    Eval = np.load('Journ_Evaluate_all.npy', allow_pickle=True)[0]
    Terms = ['Transaction latency', 'System throughput', 'Block processing time', 'Consensus efficacy', 'Data integrity', 'security', 'Fault tolerance']

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
        plt.xlabel('Node Iteration', fontsize=12)
        plt.ylabel(Terms[j], fontsize=12)
        plt.xticks(X + 1, ('10', '20', '30', '40', '50'), fontsize=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.48, 1.15),
                   ncol=3, fancybox=True, shadow=True, fontsize='12')
        path1 = "./Journ_Res/Federated_iter_%s.png" % (Terms[Graph_Term[j]])
        plt.savefig(path1)
        plt.show()

    eval = np.load('Journ_Evaluate_all.npy', allow_pickle=True)[0]
    Terms = ['Transaction latency', 'System throughput', 'Block processing time', 'Consensus efficacy', 'Data integrity', 'security', 'Fault tolerance']
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
            '10', '20', '30', '40', '50'), rotation=7, fontsize=12)
        plt.ylabel(Terms[j], fontsize=12)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=12, fancybox=True, shadow=True)
        path1 = "./Journ_Res/Federated_%s_bar_iter.png" % (Terms[Graph_Term[j]])
        plt.xlabel('Node Iteration', fontsize=12)
        plt.savefig(path1)
        plt.show()

Plot_federated_learning()
# Plot_Activation()