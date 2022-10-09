import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import os 


from load_data import load_dicom
from torch.utils.data import Dataset, DataLoader 
from torch.utils.data.sampler import BatchSampler



class NLST_BatchSampler(BatchSampler):

    # Batch Sampler, Returns n_patients patients, each with n_scans scans per patient.
    # This is customized to the NLST dataset, where each patient has 3 scans.

    def __init__(self, labels, n_patients):
        '''
        Parameters:
            labels: patient ids of all the scans
            n_patients: number of patients in a batch

        '''
        # Each patient in the NLST dataset has 3 scans 
        self.N_SCANS = 3

        self.n_patients = n_patients
        self.labels = labels
        self.batch_size = n_patients * self.N_SCANS
        self.dataset_size = len(labels)

        # a set of all the patients ids
        self.unique_patient_ids = list(set(labels))
        # dictionary of patient id to index of ct scans
        self.patient_scans_idx = {patient_id: np.where(self.labels == patient_id)[0] for patient_id in self.unique_patient_ids}

        # list to keep track of which patients have already been included in a batch 
        self.used_patient_ids = []
        self.unused_patient_ids = self.unique_patient_ids
        self.count = 0
    
    def __iter__(self):
        self.count = 0

        # each batch should have a consistent size
        print("self.dataset_size", self.dataset_size)
        print("self.count", self.count + self.batch_size)
        while self.count + self.batch_size <= self.dataset_size:

            # index of ct scans to be used in upcoming batch
            indices = []
            
            # choose n_patients from patients not used in any prior batches yet
            selected_patient_ids = np.random.choice(self.unused_patient_ids, self.n_patients, replace=False)
            scans_in_batch_idx = [self.patient_scans_idx[ids] for ids in selected_patient_ids]
            indices = np.ndarray.flatten(np.array(scans_in_batch_idx))
            print("Checking what ids look like", scans_in_batch_idx)

            # record used patient ids and updated unused patient ids
            self.used_patient_ids.extend(selected_patient_ids)
            print("Checking used patient ids", selected_patient_ids)
            self.unused_patient_ids = [id for id in self.unused_patient_ids if id not in selected_patient_ids]

            # update count of number of patients used 
            self.count += self.batch_size

            print("Checking what indices look like", indices)

            yield indices

    def __len__(self):
        '''Return number of batches.'''
        return self.dataset_size // self.batch_size
    


class CT_Volume(Dataset):
    def __init__(self, image_data, img_dim, num_slice):
      '''
      :params
        image_data: pandas table with image data (patient id, year, and location of file)
        img_dim: an integer for the dimension of the image (will produce square image)
        num_slice: an integer for the number of slice for the ct volume 
      '''
      self.img_labels = image_data["pid"]
      self.img_data = image_data
      self.img_dim = img_dim
      self.num_slice = num_slice

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
      '''
      :params
        idx: index of the image to retrieve
        
      :returns 
        Returns the pixel array of the image, the label, and the pandas row with the associate id information 
      '''
      print("Checking out idx", idx)
      image_data = self.img_data.iloc[idx, :]
      os.path.join("Data", image_data["shortened_path"])
      image = load_dicom(os.path.join("Data", image_data["shortened_path"]), self.num_slice, self.img_dim)
      label = image_data["pid"]
      return image, label



# functions for loading in data
def shorten_paths(path):
    '''Reformat paths from meta_paths.csv'''
    reformated_paths = "/".join(path.split("/")[9:])
    return reformated_paths

def load_scans(data_loc, test_ids, display=False):
    '''
    Parameters:
        data_loc: The location of the csv file containing locations of a datafile. 
        test_ids: List of patient test ids to use for creating a Dataloader. 
    '''

    # get the file paths of the test ct scans to display
    all_ct_scans = pd.read_csv(data_loc)
    test_ct_scans = all_ct_scans[all_ct_scans["pid"].isin(test_ids)]
    test_ct_scans["shortened_path"] = test_ct_scans["best image folder address"].apply(shorten_paths)

    # # create Pytorch Dataloader
    img_dim = 218
    num_slice = 100
    dataset = CT_Volume(image_data=test_ct_scans, img_dim=img_dim, num_slice=num_slice)
    batch_sampler = NLST_BatchSampler(labels=test_ct_scans["pid"].values, n_patients=2)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    # # display a slice from the Pytorch Dataloader
    _, axes = plt.subplots(nrows=2, ncols=3)

    ax = axes.ravel()

    SLICE_NUMBER = 50


    x, y = next(iter(dataloader))

    if display:
        for i in range(len(y)):
            ax[i].imshow(x[i][SLICE_NUMBER], cmap="gray")
            ax[i].set_title(f"Patient id {y[i]}, slice index {i}")
            ax[i].set_xlim(0, 220)
            ax[i].set_ylim(0, 220)

        plt.tight_layout()
        plt.show()

    return x, y

if __name__ == "__main__":
    load_scans(data_loc="Data/NLST_selected_image_folders_metadata_rerun.csv", 
    test_ids=[100002, 100004, 111098], display=True)