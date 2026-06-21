import torch
import numpy as np
import os
import Read_Data as RD

def generate_and_save_folds():
    # Configuration
    dataset_loc = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets"
    dataset_loc = r"D:\Datasets"
    # dataset_loc = "./Datasets"
    raw_data_folder = os.path.join(dataset_loc,'zenodo_eeg/')
    output_dir = os.path.join(dataset_loc,'CV_Folds')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted([f for f in os.listdir(raw_data_folder) if f.endswith('.edf')])
    np.random.seed(1)
    np.random.shuffle(files)
    
    for r in range(10):
        print(f"Processing Fold {r+1}/10...")
        
        # 1. Define the fold directory
        fold_dir = os.path.join(output_dir, f'fold_{r}')
        
        # 2. Create the fold directory if it doesn't exist
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        
        x_train, y_train, x_test, y_test,ymask_train,ymask_test = RD.read_data(
            raw_data_folder, files, r*4+1, (r+1)*4
        )
        
        # 3. Save to the verified path
        np.save(os.path.join(fold_dir, 'testdata.npy'), x_test)
        np.save(os.path.join(fold_dir, 'testlabels.npy'), y_test)
        np.save(os.path.join(fold_dir, 'testmasks.npy'), ymask_test)
        np.save(os.path.join(fold_dir, 'traindata.npy'), x_train)
        np.save(os.path.join(fold_dir, 'trainlabels.npy'), y_train)
        np.save(os.path.join(fold_dir, 'trainmasks.npy'), ymask_train)

        print(f"Saved Fold {r} to {fold_dir}")

if __name__ == "__main__":
    generate_and_save_folds()