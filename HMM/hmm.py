
from enum import IntEnum 
import numpy as np
import sys
import subprocess
from hmmlearn.hmm import CategoricalHMM
import random

class SuspectState(IntEnum):
    Planning = 1,
    Scouting = 2,
    Burglary = 3,
    Migrating = 4,
    Misc = 5

class Daytime(IntEnum):
    Day = 6
    Evening = 7
    Night = 8

class Action(IntEnum):
    Roaming = 9,
    Eating = 10,
    Home = 11,
    Untracked = 12,

class Observation:
    def __init__(self, d:Daytime, a:Action) -> None:
        self.daytime = d
        self.action = a


# This function reads the string file 
# that contains the sequence of observations
# that is required to learn the HMM model.
def ReadDataset() -> list:
    # Converts integer to Daytime enum
    def getDay(p: int) -> Daytime:
        if p == 6:
            return Daytime.Day
        elif p == 7:
            return Daytime.Evening
        elif p == 8:
            return Daytime.Night
        else:
            assert False, 'Unexpected Daytime!'

    # Converts integer to Action enum
    def getAct(p: int) -> Action:
        if p == 9:
            return Action.Roaming
        elif p == 10:
            return Action.Eating
        elif p == 11:
            return Action.Home
        elif p == 12:
            return Action.Untracked
        else:
            assert False, 'Unexpected Action!'
    filepath = 'database.txt' 
    with open(filepath, 'r') as file:
        seq_count = int(file.readline())
        seq_list = []
        for _ in range(seq_count):
            w = file.readline().split(' ')
            len = int(w[0])
            seq_i = []
            for k in range(0, len):
                idx = (k*2) + 1
                day = int(w[idx])
                act = int(w[idx + 1])
                o = Observation(getDay(day), getAct(act))
                seq_i.append(o)
            seq_list.append(seq_i)
    return seq_list


#  --------------Do not change anything above this line---------------
class HMM:
    # Complete the HMM implementation.
    # The three function below must
    # be implemented.
    
    # mat_A = np.array([[0.3, 0.4, 0.0, 0.0, 0.3],
    #                 [0.0, 0.3, 0.3, 0.0, 0.4],
    #                 [0.0, 0.0, 0.0, 1.0, 0.0],
    #                 [1.0, 0.0, 0.0, 0.0, 0.0],
    #                 [0.3, 0.4, 0.0, 0.0, 0.3]])
    
    # mat_B = np.array([[0.00, 0.00, 0.40, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.40, 0.00],
    #                 [0.25, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00],
    #                 [0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.40],
    #                 [0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.40],
    #                 [0.00, 0.25, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00]])
        
    # PI = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    
    def __init__(self):
        
        self.mat_A = np.array([[0.2, 0.3, 0.1, 0.2, 0.2],
                            [0.2, 0.3, 0.3, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.5, 0.1],
                            [0.5, 0.1, 0.1, 0.2, 0.1],
                            [0.2, 0.3, 0.1, 0.1, 0.3]])
    
        # self.mat_B = np.array([[0.00, 0.00, 0.40, 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.40, 0.00],
        #                     [0.25, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00],
        #                     [0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.40],
        #                     [0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.40],
        #                     [0.00, 0.25, 0.00, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00]])

        self.mat_B = np.array([[0.00, 0.00, 0.70, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.15, 0.00],
                            [0.15, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.70, 0.00, 0.00, 0.00],
                            [0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.70],
                            [0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00, 0.70],
                            [0.00, 0.15, 0.00, 0.00, 0.00, 0.70, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00]])
        
        self.PI = np.array([0.0, 0.0, 0.5, 0.5, 0.0])

        self.model = None
        
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hmmlearn', '--quiet'])


         # process output with an API in the subprocess module:
        # # reqs = subprocess.check_output([sys.executable, '-m', 'pip',        'freeze'])
        # installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

        # print(installed_packages)
        
        # self.mat_B = np.array([[0.00, 0.00, 0.40-(1e-3), 0.00, 0.00, 0.00, 0.20, 0.00, 0.00, 0.00, 0.40, 0.00, 1e-3],
        #                     [0.25, 0.00, 0.00, 0.00, 0.25, 0.00, 0.00, 0.00, 0.50-(1e-3), 0.00, 0.00, 0.00, 1e-3],
        #                     [0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.40-(1e-3), 1e-3],
        #                     [0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.40-(1e-3), 1e-3],
        #                     [0.00, 0.25, 0.00, 0.00, 0.00, 0.50-(1e-3), 0.00, 0.00, 0.00, 0.25, 0.00, 0.00, 1e-3]])
        
        # self.PI = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    # def update_pi(self):
    #     # mat_A2 = self.mat_A.copy()
    #     for i in range(1000):
    #         # print(i)
    #         self.PI = np.matmul(self.PI, self.mat_A)

    #     return self.PI
    


    def A(self, a: SuspectState, b: SuspectState) -> float:
        # Compute the probablity of going from one
        # state a to the other state b
        # Hint: The necessary code does not need
        # to be within this function only
        # pass
        
        if a == SuspectState.Planning:
            if b == SuspectState.Planning:
                i, j = 0, 0
            elif b == SuspectState.Scouting:
                i, j = 0, 1
            elif b == SuspectState.Burglary:
                i, j = 0, 2
            elif b == SuspectState.Migrating:
                i, j = 0, 3
            elif b == SuspectState.Misc:
                i, j = 0, 4
        elif a == SuspectState.Scouting:
            if b == SuspectState.Planning:
                i, j = 1, 0
            elif b == SuspectState.Scouting:
                i, j = 1, 1
            elif b == SuspectState.Burglary:
                i, j = 1, 2
            elif b == SuspectState.Migrating:
                i, j = 1, 3
            elif b == SuspectState.Misc:
                i, j = 1, 4
        elif a == SuspectState.Burglary:
            if b == SuspectState.Planning:
                i, j = 2, 0
            elif b == SuspectState.Scouting:
                i, j = 2, 1
            elif b == SuspectState.Burglary:
                i, j = 2, 2
            elif b == SuspectState.Migrating:
                i, j = 2, 3
            elif b == SuspectState.Misc:
                i, j = 2, 4
        elif a == SuspectState.Migrating:
            if b == SuspectState.Planning:
                i, j = 3, 0
            elif b == SuspectState.Scouting:
                i, j = 3, 1
            elif b == SuspectState.Burglary:
                i, j = 3, 2
            elif b == SuspectState.Migrating:
                i, j = 3, 3
            elif b == SuspectState.Misc:
                i, j = 3, 4
        elif a == SuspectState.Misc:
            if b == SuspectState.Planning:
                i, j = 4, 0
            elif b == SuspectState.Scouting:
                i, j = 4, 1
            elif b == SuspectState.Burglary:
                i, j = 4, 2
            elif b == SuspectState.Migrating:
                i, j = 4, 3
            elif b == SuspectState.Misc:
                i, j = 4, 4
        # mat_A = np.array(mat_A)

        return self.mat_A[i][j]

        # return mat_A

    def B(self, a: SuspectState, b: Observation) -> float:
        # Compute the probablity of obtaining
        # observation b from state a
        # Hint: The necessary code does not need
        # to be within this function only
        # pass
        
        if a == SuspectState.Planning:
            if b.daytime == Daytime.Day and b.action == Action.Roaming:
                i, j = 0, 0
            elif b.daytime == Daytime.Day and b.action == Action.Eating:
                i, j = 0, 1
            elif b.daytime == Daytime.Day and b.action == Action.Home:
                i, j = 0, 2
            elif b.daytime == Daytime.Day and b.action == Action.Untracked:
                i, j = 0, 3
            elif b.daytime == Daytime.Evening and b.action == Action.Roaming:
                i, j = 0, 4
            elif b.daytime == Daytime.Evening and b.action == Action.Eating:
                i, j = 0, 5
            elif b.daytime == Daytime.Evening and b.action == Action.Home:
                i, j = 0, 6
            elif b.daytime == Daytime.Evening and b.action == Action.Untracked:
                i, j = 0, 7
            elif b.daytime == Daytime.Night and b.action == Action.Roaming:
                i, j = 0, 8
            elif b.daytime == Daytime.Night and b.action == Action.Eating:
                i, j = 0, 9
            elif b.daytime == Daytime.Night and b.action == Action.Home:
                i, j = 0, 10
            elif b.daytime == Daytime.Night and b.action == Action.Untracked:
                i, j = 0, 11
        elif a == SuspectState.Scouting:
            if b.daytime == Daytime.Day and b.action == Action.Roaming:
                i, j = 1, 0
            elif b.daytime == Daytime.Day and b.action == Action.Eating:
                i, j = 1, 1
            elif b.daytime == Daytime.Day and b.action == Action.Home:
                i, j = 1, 2
            elif b.daytime == Daytime.Day and b.action == Action.Untracked:
                i, j = 1, 3
            elif b.daytime == Daytime.Evening and b.action == Action.Roaming:
                i, j = 1, 4
            elif b.daytime == Daytime.Evening and b.action == Action.Eating:
                i, j = 1, 5
            elif b.daytime == Daytime.Evening and b.action == Action.Home:
                i, j = 1, 6
            elif b.daytime == Daytime.Evening and b.action == Action.Untracked:
                i, j = 1, 7
            elif b.daytime == Daytime.Night and b.action == Action.Roaming:
                i, j = 1, 8
            elif b.daytime == Daytime.Night and b.action == Action.Eating:
                i, j = 1, 9
            elif b.daytime == Daytime.Night and b.action == Action.Home:
                i, j = 1, 10
            elif b.daytime == Daytime.Night and b.action == Action.Untracked:
                i, j = 1, 11
        elif a == SuspectState.Burglary:
            if b.daytime == Daytime.Day and b.action == Action.Roaming:
                i, j = 2, 0
            elif b.daytime == Daytime.Day and b.action == Action.Eating:
                i, j = 2, 1
            elif b.daytime == Daytime.Day and b.action == Action.Home:
                i, j = 2, 2
            elif b.daytime == Daytime.Day and b.action == Action.Untracked:
                i, j = 2, 3
            elif b.daytime == Daytime.Evening and b.action == Action.Roaming:
                i, j = 2, 4
            elif b.daytime == Daytime.Evening and b.action == Action.Eating:
                i, j = 2, 5
            elif b.daytime == Daytime.Evening and b.action == Action.Home:
                i, j = 2, 6
            elif b.daytime == Daytime.Evening and b.action == Action.Untracked:
                i, j = 2, 7
            elif b.daytime == Daytime.Night and b.action == Action.Roaming:
                i, j = 2, 8
            elif b.daytime == Daytime.Night and b.action == Action.Eating:
                i, j = 2, 9
            elif b.daytime == Daytime.Night and b.action == Action.Home:
                i, j = 2, 10
            elif b.daytime == Daytime.Night and b.action == Action.Untracked:
                i, j = 2, 11
        elif a == SuspectState.Migrating:
            if b.daytime == Daytime.Day and b.action == Action.Roaming:
                i, j = 3, 0
            elif b.daytime == Daytime.Day and b.action == Action.Eating:
                i, j = 3, 1
            elif b.daytime == Daytime.Day and b.action == Action.Home:
                i, j = 3, 2
            elif b.daytime == Daytime.Day and b.action == Action.Untracked:
                i, j = 3, 3
            elif b.daytime == Daytime.Evening and b.action == Action.Roaming:
                i, j = 3, 4
            elif b.daytime == Daytime.Evening and b.action == Action.Eating:
                i, j = 3, 5
            elif b.daytime == Daytime.Evening and b.action == Action.Home:
                i, j = 3, 6
            elif b.daytime == Daytime.Evening and b.action == Action.Untracked:
                i, j = 3, 7
            elif b.daytime == Daytime.Night and b.action == Action.Roaming:
                i, j = 3, 8
            elif b.daytime == Daytime.Night and b.action == Action.Eating:
                i, j = 3, 9
            elif b.daytime == Daytime.Night and b.action == Action.Home:
                i, j = 3, 10
            elif b.daytime == Daytime.Night and b.action == Action.Untracked:
                i, j = 3, 11
        elif a == SuspectState.Misc:
            if b.daytime == Daytime.Day and b.action == Action.Roaming:
                i, j = 4, 0
            elif b.daytime == Daytime.Day and b.action == Action.Eating:
                i, j = 4, 1
            elif b.daytime == Daytime.Day and b.action == Action.Home:
                i, j = 4, 2
            elif b.daytime == Daytime.Day and b.action == Action.Untracked:
                i, j = 4, 3
            elif b.daytime == Daytime.Evening and b.action == Action.Roaming:
                i, j = 4, 4
            elif b.daytime == Daytime.Evening and b.action == Action.Eating:
                i, j = 4, 5
            elif b.daytime == Daytime.Evening and b.action == Action.Home:
                i, j = 4, 6
            elif b.daytime == Daytime.Evening and b.action == Action.Untracked:
                i, j = 4, 7
            elif b.daytime == Daytime.Night and b.action == Action.Roaming:
                i, j = 4, 8
            elif b.daytime == Daytime.Night and b.action == Action.Eating:
                i, j = 4, 9
            elif b.daytime == Daytime.Night and b.action == Action.Home:
                i, j = 4, 10
            elif b.daytime == Daytime.Night and b.action == Action.Untracked:
                i, j = 4, 11

        # mat_B = np.array(mat_B)

        return self.mat_B[i][j]


    def Pi(self, a: SuspectState) -> float:
        # Compute the initial probablity of
        # being at state a
        # Hint: The necessary code does not need
        # to be within this function only
        # pass
        
        if a == SuspectState.Planning:
            i = 0
        elif a == SuspectState.Scouting:
            i = 1
        elif a == SuspectState.Burglary:
            i = 2
        elif a == SuspectState.Migrating:
            i = 3
        elif a == SuspectState.Misc:
            i = 4

        return self.PI[i]

        
def DatasetToState(seq_list: list) -> list:
    
    # print(seq_list)
    new_data = []


    for i in seq_list:
        data = []
        count = 0
        # while count < len(i):
        # print(i[0].daytime, i[0].action)
        # break
        for j in i:
            # print(j)
            if j.daytime == Daytime.Day:
                if j.action == Action.Roaming:
                    data.append(0)
                elif j.action == Action.Eating:
                    data.append(1)
                elif j.action == Action.Home:
                    data.append(2)
                elif j.action == Action.Untracked:
                    data.append(3)
            
            elif j.daytime == Daytime.Evening:
                if j.action == Action.Roaming:
                    data.append(4)
                elif j.action == Action.Eating:
                    data.append(5)
                elif j.action == Action.Home:
                    data.append(6)
                elif j.action == Action.Untracked:
                    data.append(7)

            elif j.daytime == Daytime.Night:
                if j.action == Action.Roaming:
                    data.append(8)
                elif j.action == Action.Eating:
                    data.append(9)
                elif j.action == Action.Home:
                    data.append(10)
                elif j.action == Action.Untracked:
                    data.append(11)

        new_data.append(data)
            # count = count + 2

    # print(new_data)
    l = []
    for i in new_data:
        l.append(len(i))

    max_len = max(l)
    # print(max(l), len(l))
    for k in new_data:
        # print(type(k))
        # break
        if len(k) < max_len:
            k.extend([random.randint(0,11)]*(max_len-len(k)))
            # k.extend([12]*(max_len-len(k)))
    # print(new_data,sep="\n")
    return new_data


# Part I
# ---------

# Reads the dataset of array of sequence of observation
# and initializes a HMM model from it.
# returns Initialized HMM.
# Here, the parameter `dataset` is
# a list of list of `Observation` class.
# Each (inner) list represents the sequence of observation
# from start to end as mentioned in the question


def LearnModel(dataset: list) -> HMM:

    hmm_obj = HMM()

    data = DatasetToState(dataset)
    state_prob = hmm_obj.PI 
    transmat_mat = hmm_obj.mat_A
    emissionprob_mat = hmm_obj.mat_B

    model = CategoricalHMM(n_components = 5, startprob_prior = state_prob, transmat_prior = transmat_mat,
                           emissionprob_prior = emissionprob_mat, params = "te",random_state = 64)

    data = np.asarray(data)

    model.fit(data)

    # print(model.transmat_)
    # print(model.startprob_)
    # print(model.emissionprob_)

    hmm_obj.PI = model.startprob_
    hmm_obj.mat_A = model.transmat_
    hmm_obj.mat_B = model.emissionprob_#model.get_stationary_distribution()
    hmm_obj.model = model

    # print(hmm_obj.PI)
   
    return hmm_obj

# Part II
# ---------

# Given an initialized HMM model,
# and some set of observations, this function evaluates
# the liklihood that this set of observation was indeed
# generated from the given model.
# Here, the obs_list is a list containing
# instances of the `Observation` class.
# The output returned has to be floatint point between
# 0 and 1



    

def Liklihood(model: HMM, obs_list: list) -> float:
    data = []
    data.append(obs_list)
    dataset = DatasetToState(data)
    # print(np.asarray(dataset))
    dt = np.asarray(dataset)
    # pass
    # print(model.model.score(dt))
    return np.exp(model.model.score(dt))
    # pass


# // Part III
# //---------

# Given an initialized model, and a sequence of observation,
# returns  a list of the same size, which contains
# the most likely # states the model
# was in to produce the given observations.
# returns An array/sequence of states that the model was in to
# produce the corresponding sequence of observations

def GetHiddenStates(model: HMM, obs_list: list) -> list:
    State_List = [SuspectState.Planning, SuspectState.Scouting,
              SuspectState.Burglary, SuspectState.Migrating, SuspectState.Misc]
    data = []
    data.append(obs_list)
    dataset = DatasetToState(data)
    # print(np.asarray(dataset))
    dt = np.asarray(dataset)

    lt = model.model.decode(dt)
    # print(type(lt[1]))
    st = []
    # for i in list(lt[1]):
    #     if i == 0:
    #         st.append(SuspectState.Planning)

    for i in lt[1]:
        st.append(State_List[i])

    return st



# hmm_obj = HMM()

# data = DatasetToState(seq_list = ReadDataset())
# print(type(data))
# dataset = np.asarray(data)
# print(type(data))
# LearnModel(dataset)
# print(hmm.mat_A)
# print(hmm.mat_B)
# print(hmm.update_pi())
# hmm.A(SuspectState.Planning, SuspectState.Scouting)

# ls = [[
#         Observation(Daytime.Day, Action.Home),
#         Observation(Daytime.Evening, Action.Eating),
#         Observation(Daytime.Night, Action.Home),
#         Observation(Daytime.Day, Action.Home),
#         Observation(Daytime.Evening, Action.Eating),
#         Observation(Daytime.Night, Action.Roaming),
#     ]]
# Liklihood(hmm_obj, ls)