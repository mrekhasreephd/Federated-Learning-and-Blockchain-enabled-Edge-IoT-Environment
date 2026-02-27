import numpy as np
from Global_Vars import Global_Vars
from Model_DNN import Model_DNN


def Objfun_Cls(Soln):
    images = Global_Vars.Feat
    Targ = Global_Vars.target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        learnper = round(images .shape[0] * 0.75)
        train_data = images [learnper:, :]
        train_target = Targ[learnper:, :]
        test_data = images [:learnper, :]
        test_target = Targ[:learnper, :]
        Eval = Model_DNN(train_data, train_target, test_data, test_target,sol.astype('int'))
        Fitn[i] = (1 / Eval[11]) + Eval[13]
    return Fitn



