import pandas as pd
import numpy as np
from numpy import matlib
import random as rn

from BlockChain import Blockchain
from DMO import DMO
from GSO import GSO
from Global_Vars import Global_Vars
from LOA import LOA
from Model_CHAINFL import Model_CHAINFL
from Model_DNN import Model_DNN
from Model_DRL import Model_DRL
from Model_DTWN import Model_DTWN
from NRO import NRO
from Objective_Function import Objfun_Cls
from PLOT_RESULTS import Plot_Activation, Plot_Fitness, Confusion_matrix, Plot_Kfold
from PROPOSED import PROPOSED

#Read dataset
an = 0
if an == 1:
    nodes = [50, 100, 150, 200, 250]
    node_names = [f'{nodes[i]}' for i in range(len(nodes))]
    node_coordinates = [(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in nodes]
    topology = np.random.randint(0, 2, size=(len(nodes), len(nodes)))
    distance_matrix = np.random.rand(len(nodes), len(nodes)) * 100
    initial_energy = np.random.rand(len(nodes)) * 100
    trust = np.random.rand(len(nodes))
    hop_count = np.random.randint(1, 10, size=len(nodes))
    residual_energy = initial_energy - np.random.rand(len(nodes)) * 50
    cooperation_history = np.random.randint(0, 10, size=len(nodes))
    message_passing_behavior = np.random.choice(['Message Dropping', 'Message Delay'], size=len(nodes))
    resource_utilization = np.random.choice(['Bandwidth Usage', 'Computation Power'], size=len(nodes))
    participation_level = np.random.choice(['Lack of Contribution', 'Inconsistent Activity'], size=len(nodes))
    transaction_handling = np.random.choice(['Unfair Transaction Prioritization', 'Transaction Blacklisting'], size=len(nodes))
    network_topology_behavior = np.random.choice(['Isolation Tendencies', 'Failure to Relay Information'], size=len(nodes))
    resource_sharing_and_cooperation = np.random.choice(['Unfair Resource Consumption', 'Refusal to Cooperate'], size=len(nodes))
    consistency_and_predictability = np.random.choice(['Behavioral Patterns', 'Predictable Actions'], size=len(nodes))
    game_theoretic_measures = np.random.choice(['Utility Maximization', 'Defection Probability'], size=len(nodes))
    reputation_and_trust = np.random.choice(['Low Reputation Scores', 'Trustworthiness Analysis'], size=len(nodes))
    economic_incentives = np.random.choice(['Economic Rationality', 'Cost-Benefit Analysis'], size=len(nodes))
    target = np.random.choice(['Selected', 'Not Selected'], size=len(nodes))
    data = {
        'Node Names': node_names,
        'Node Coordinates': node_coordinates,
        'Topology': [list(row) for row in topology],
        'Traversal Between Nodes': [list(row) for row in topology],  # Assuming same as topology for simplicity
        'Distance Matrix': [list(row) for row in distance_matrix],
        'Initial Energy': initial_energy,
        'Trust': trust,
        'Hop Count': hop_count,
        'Residual Energy': residual_energy,
        'Cooperation History': cooperation_history,
        'Message Passing Behavior': message_passing_behavior,
        'Resource Utilization': resource_utilization,
        'Participation Level': participation_level,
        'Transaction Handling': transaction_handling,
        'Network Topology Behavior': network_topology_behavior,
        'Resource Sharing and Cooperation': resource_sharing_and_cooperation,
        'Consistency and Predictability': consistency_and_predictability,
        'Game-Theoretic Measures': game_theoretic_measures,
        'Reputation and Trust': reputation_and_trust,
        'Economic Incentives': economic_incentives,
        'Target': target
    }
    df = pd.DataFrame(data)
    datas = df.values
    dat = datas[:,10:]
    for j in range(dat.shape[1]):
        dats = np.asarray(dat[:,j])
        uni = np.unique(dats)
        tar = np.zeros((dats.shape[0])).astype('int')
        for i in range(len(uni)):
            ind = np.where((dats == uni[i]))
            tar[ind[0]] = i+1
        dat[:,j] = tar
    datas[:, 10:] = dat
    Target = datas[:,-1]
    Targ = np.asarray(Target)
    uniq = np.unique(Targ)
    tars = np.zeros((Targ.shape[0])).astype('int')
    for i in range(len(uniq)):
        ind = np.where((Targ == uniq[i]))
        tars[ind[0]] = i
    np.save('Datas.npy',np.asarray(datas))
    np.save('Target.npy',np.asarray(tars).reshape(-1,1))

# Stored in a Blockchain
an = 0
if an == 1:
    Data = np.load('Datas.npy',allow_pickle=True)
    Blockchain(Data)

# Optimization for Classification
an = 0
if an == 1:
    Bestsol = []
    Fitness = []
    no_of_nodes = 5
    Node = [50,100,150,200,250]
    for i in range(5):
        Feat = np.load('Datas.npy', allow_pickle=True)
        Targets = np.load('Target.npy', allow_pickle=True)
        Global_Vars.Feat = Feat
        Global_Vars.target = Targets
        Npop = 10
        Chlen = 3
        xmin = matlib.repmat([5,5,5], Npop, 1)
        xmax = matlib.repmat([255,50,255], Npop, 1)
        fname = Objfun_Cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.asarray(rn.uniform(xmin[p1, p2], xmax[p1, p2]))
        Max_iter = 50

        print("DMO...")
        [bestfit1, fitness1, bestsol1, time1] = DMO(initsol, fname, xmin, xmax, Max_iter)

        print("GSOA...")
        [bestfit2, fitness2, bestsol2, time2] = GSO(initsol, fname, xmin, xmax, Max_iter)

        print("LOA...")
        [bestfit4, fitness4, bestsol4, time3] = LOA(initsol, fname, xmin, xmax, Max_iter)

        print("NRO...")
        [bestfit3, fitness3, bestsol3, time4] = NRO(initsol, fname, xmin, xmax, Max_iter)

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        Bestsol.append([bestsol1,bestsol2,bestsol3,bestsol4,bestsol5])
        Fitness.append([fitness1.ravel(),fitness2.ravel(),fitness3.ravel(),fitness4.ravel(),fitness5.ravel()])
    np.save('Bestsol.npy', Bestsol)
    np.save('Fitness.npy',Fitness)

# CLASSIFICATION
an = 0
if an == 1:
    Feat = np.load('Datas.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy',allow_pickle = True)
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Eval_all = []
    Activation = ["Linear", "ReLU", "Tanh", "Softmax", "Sigmoid"]
    for epcs in range(len(Activation)):
        Eval = np.zeros((10, 14))
        learnperc = round(Feat.shape[0] * 0.75)
        for j in range(sol.shape[0]):
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval[j,:] = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target, sol[j].astype('int'))
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval[5, :] = Model_DRL(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :] = Model_DTWN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :] = Model_CHAINFL(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :] = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :] = Eval[4, :]
        Eval_all.append(Eval)
    np.save('Eval_Act.npy', np.asarray(Eval_all))

Plot_Activation()
Plot_Fitness()
Confusion_matrix()
Plot_Kfold()

