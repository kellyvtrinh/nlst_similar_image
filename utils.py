import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'

from itertools import combinations
import torch

class TripletSelector:
    '''
    Implement should return a list of triplets' indices. 
    '''

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

def pdist(vectors):
    '''
    Given a matrix n x d, find the pairwise distance of all the rows in the matrix.
    Logic explained by https://jaykmody.com/posts/distance-matrices-with-numpy/#no-loops.
    Given input matrix X, returns an n x n matrix Y, where Y[i,j] = the distance between X[i] and X[j]. 

    Parameters: 
      vectors: a pytorch tensor, a matrix n x d, where n is the number of examples, 
        and d is the dimension of each of the examples. 
    '''
    return -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)


def pick_hardest(self, loss_values):
    return [np.argmax(loss_values)]

def pick_randomly(self, loss_values):
    idx = np.random.choice(np.arange(len(loss_values)), 1)
    return [idx]

def pick_all(self, loss_values):
    return list[np.range(len(loss_values))]


# Hard negatives: all, random, hardest
# Semi-hard negatives: all, random, hardest
# All negatives

class NegativesSelectors(TripletSelector):

    def __init__(self, margin, negative_selection_fn, selection_criteria):
        ''' 
        Parameters: 
            Margin: used in the loss function to control how dissimilar the embeddings of the negative example with that of the anchor.
            Negative_selection_fn: a function which defines how the negative examples are selected.
            Selection_criteria: out of all the negative examples that qualify (ex: hard, semi-hard, easy negative examples), 
                how do we choose among all these options?
        '''
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.selection_criteria = selection_criteria

    @staticmethod
    def return_all_negatives(self, embeddings, labels):

        ''' 
        '''
        pass

    @staticmethod
    def hard_negatives(self, curr_positive_dist, all_negative_dist, selection_criteria):
        ''' 
        A hard negative is where d(a, p) > d(a, n).
        For one single pair of anchor and its positive (a, p), 
        select one or multiple hard negatives to complete the triplet. 

        Parameters: 
            curr_positive_dist = d(a, p) = distance between anchor and positive
            all_negative_dist = distances of all possible pairs of anchor and negatives
            selection_criteria: For pairs (a, p), there will be multiple possible hard negative examples to choose from.
                                Can choose from functions pick_hardest, pick_randomly, pick_all. 
                                If selection criteria is pick_hardest or pick_randomly, return a list of one item. 
        
        '''
        # loss = d(a, p) - d(a, n) + margin
        # hard negatives meant that d(a, n) < d(a, p) and d(a, p) - d(a, n) = loss - margin > 0
        
        possible_hard_negatives = np.where(curr_positive_dist - all_negative_dist > 0)[0]
        return selection_criteria(possible_hard_negatives)

    @staticmethod
    def semi_hard_negatives(self, curr_positive_dist, all_negative_dist, selection_criteria):
        ''' 
        A semi-hard negative is where d(a, p) < d(a, n) < d(a, p) + margin.
        For one single pair of anchor and its positive (a, p), 
        select one or multiple semi-hard negatives to complete the triplet. 

        Parameters: 
            curr_positive_dist = d(a, p) = distance between anchor and positive
            all_negative_dist = distances of all possible pairs of anchor and negatives
            selection_criteria: For pairs (a, p), there will be multiple possible hard negative examples to choose from.
                                Can choose from functions pick_hardest, pick_randomly, pick_all. 
                                If selection criteria is pick_hardest or pick_randomly, return a list of one item. 
        '''

        condition = (curr_positive_dist > all_negative_dist & curr_positive_dist + self.margin > all_negative_dist)
        possible_semi_hard_negatives = np.where(condition)
        return selection_criteria(possible_semi_hard_negatives)

    
    def get_triplets(self, embeddings, labels):

        # find pairwise distances
        distance_matrix = pdist(embeddings).numpy()
        
        # all patient ids 
        all_patient_ids = set(labels)
        n_patients = len(all_patient_ids)

        # positive_idx[i] = a list of indices of scans belonging to the ith patient
        positive_idx = np.array([np.where(labels == id)[0] for id in all_patient_ids])
        # negative_idx[i] = a list of indices of scans that do not belong to the ith patient
        negative_idx = np.array([np.where(np.logical_not(self.patient_ids == id))[0] for id in all_patient_ids])

        # for example, positive_idx = [[1, 2, 3], [4, 5, 6]]
        # positive_pairings = [   [(1, 2), (1, 3), (2, 3)]  ,  [(4, 5), (4, 6), (5, 6)]    ]
        positive_pairings = [list(combinations(positive_idx[i], 2)) for i in range(n_patients)]
        # positive_pairings = pos_pairs.reshape(pos_pairs.shape[0] * pos_pairs.shape[1], pos_pairs.shape[2])

        # ap_distances[j] = distance between the jth (a, p) pairing, with a = anchor, p = positive pairing
        ap_distances = distance_matrix[positive_pairings[:, :, 0], positive_idx[:, :, 1]]

        triplets = []

        scan_no = 0
        for patient in range(n_patients):
            for pairings in positive_pairings[patient]:
                
                # d(a, p)
                curr_positive_dist = ap_distances[scan_no] 

                # a list of different d(a, n) for all n (negative example) for the current a (anchor)
                all_negative_dist = distance_matrix[pairings[0], negative_idx[patient]]
                # indices of negative pairs that suit whichever method of negative pair selection
                valid_negatives = self.negative_selection_fn(curr_positive_dist, all_negative_dist, self.selection_criteria)
            
                if valid_negatives is not None:

                    selected_negative_idx = negative_idx[valid_negatives]
                    triplets.extend([[positive_pairings[0], positive_pairings[1], idx] for idx in selected_negative_idx])
        
        if len(triplets) == 0:
            # TODO: figure out what to do if there are no negatives that meet the criteria
            triplets.append(positive_idx[0], positive_idx[1], negative_idx)

        return triplets

    
# testing out triplet selectors 

