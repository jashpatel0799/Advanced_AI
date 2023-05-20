from hmm import *

# Part IV
# ---------


class UpdatedSuspectState(IntEnum):
    # Add custom states here
    pass

class Updated_HMM:
    # Complete this implementation
    # for part IV of the assignment
    pass

# Since python does not support function 
# overloading (unlike C/C++), the names of 
# the following functions are made to be different

# Reads the dataset of array of sequence of observation
# and initializes a HMM model from it.
# returns Initialized HMM.
# Here, the parameter `dataset` is
# a list of list of `Observation` class.
# Each (inner) list represents the sequence of observation
# from start to end as mentioned in the question


def LearnUpdatedModel(dataset: list) -> Updated_HMM:
    pass


# Given an initialized HMM model,
# and some set of observations, this function evaluates
# the liklihood that this set of observation was indeed
# generated from the given model.
# Here, the obs_list is a list containing
# instances of the `Observation` class.
# The output returned has to be floatint point between
# 0 and 1


def LiklihoodUpdated(model: Updated_HMM, obs_list: list) -> float:
    pass

# Given an initialized model, and a sequence of observation,
# returns  a list of the same size, which contains
# the most likely # states the model
# was in to produce the given observations.
# returns An array/sequence of states that the model was in to
# produce the corresponding sequence of observations

def GetUpdatedHiddenStates(model: Updated_HMM, obs_list: list) -> list:
    pass

if __name__ == "__main__":
    database = ReadDataset()
    old_model = LearnModel(database)
    new_model = LearnUpdatedModel(database)


    # obs_list = [ ] # Add your list of observations
    # p = Liklihood(old_model, obs_list)
    # q = Liklihood(new_model, obs_list)

    # old_states = GetHiddenStates(old_model, obs_list)
    # new_states = GetUpdatedHiddenStates(new_model, obs_list)

    # Add code to showcase and compare the obtained
    # results between the two models
