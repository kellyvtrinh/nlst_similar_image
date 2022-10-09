import numpy as np
import pandas as pd 

from losses import OnlineTripletLoss
from utils import NegativesSelectors as selectors
from utils import pick_hardest, pick_randomly, pick_all
from model import EmbeddingNet
from dataset import load_scans


# loss_fn_hard = OnlineTripletLoss(margin=margin, triplet_selector=hard_selector)
# loss_fn_semi_hard = OnlineTripletLoss(margin=margin, triplet_selector=semi_hard_selector)

# loss_fn(*loss_inputs)
def training_loop():
    pass

def validation():
    pass

def testing_loop():
    pass 

def debug_triplet_selector():
    margin = 1.
    hard_negative = selectors.hard_negatives 
    semi_hard_negative = selectors.semi_hard_negatives

    hard_selector = selectors(margin=margin, negative_selection_fn=pick_hardest, selection_criteria=hard_negative)
    semi_hard_selector = selectors(margin=margin, negative_selection_fn=pick_all, selection_criteria=semi_hard_negative)

    x, y = load_scans(data_loc="Data/NLST_selected_image_folders_metadata_rerun.csv", 
        test_ids=[100002, 100004, 111098])

    encoder = EmbeddingNet()
    embeddings = encoder(x)

    triplets = hard_selector.get_triplets(embeddings=embeddings, labels=y)
    return triplets

if __name__ == "__main__":
    print("Here are the triplet outputs", debug_triplet_selector())