import torch
import torch.nn as nn
import numpy as np
import os
import glob
import json
import argparse
import random
from torch.utils.data import TensorDataset
from torch_geometric.loader import DataLoader 
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score, 
    cohen_kappa_score, 
    recall_score, 
    precision_score, 
    f1_score
)
import re
# import Read_Data as RD
import MyUtils_senn_test as MyUtils
# Import local project modules
import Models_senn as Model
#import Read_Data as RD


def set_random_seed(config_params):
    #random state initialization of the code - values - 8, 24, 30
    torch.manual_seed(config_params['randseedother']) 
    torch.cuda.manual_seed(config_params['randseedother'])
    torch.cuda.manual_seed_all(config_params['randseedother'])
    np.random.seed(config_params['randseeddata'])
    random.seed(config_params['randseeddata'])
    g = torch.Generator()
    g.manual_seed(config_params['randseedother'])
    torch.backends.cudnn.deterministic = True
    return g

def get_patient_id(filepath):
    """Extracts the integer ID from 'patient_12_processed.npz'"""
    match = re.search(r'patient_(\d+)_', os.path.basename(filepath))
    return int(match.group(1)) if match else None

def load_patient_group(file_list):
    """Loads and concatenates multiple .npz files."""
    all_x, all_y = [], []
    for f in file_list:
        data = np.load(f)
        all_x.append(data['x'])
        all_y.append(data['y'])
    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

def cross_validate(model_type,lr=0.002,wd=1e-4,rob_loss=0.0):
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 2. Path Configuration
    #raw_data_folder = 'Datasets/zenodo_eeg/'
    history_dir = "History"
    model_dir = "Saved_models"
    # PROCESSED_DIR = "./Datasets/Processed_data"
    # raw_data_folder = "./Datasets/zenodo_eeg/"
    
    
    # files=os.listdir(raw_data_folder)
    for path in [history_dir, model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # all_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "patient_*_processed.npz")), 
    #                    key=get_patient_id)
    
    # np.random.shuffle(all_files)

    # folds = np.array_split(all_files, 10)

    # Deterministic sorting ensures fold splits are reproducible
    # files = sorted([f for f in os.listdir(raw_data_folder) if f.endswith('.edf')])
    init_adj = Model.adj
    
    # Hyperparameters
    batch_size = 512
    # batch_size = 32#local testing
    epochs = 250
    warmup_epochs = 5
    patience_earlyStop = 40
    learning_rate = lr #starting value, we use LR scheduler
    weight_decay = wd#0.0

    # learning_rate = 0.001 #Original values seem to work better
    # weight_decay = 0.001 #0.0

    
    data_folder = "./Datasets/CV_Folds/"
    # data_folder =  r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\Datasets\CV_Folds/"
    # 3. 10-Fold Cross Validation Loop
    for r in range(10): #first only try first fold
        print(f"\n{'='*20} Starting Fold {r+1}/10 {'='*20}")
        print(f"learning rate = {learning_rate}")
        print(f"weight decay = {weight_decay}")
        
        
        # Load data for this fold

        fold_dir = os.path.join(data_folder, f'fold_{r}')
        print("loading data...")
        x_train = np.load(os.path.join(fold_dir,'traindata.npy'),mmap_mode='r')
        y_train = np.load(os.path.join(fold_dir,'trainlabels.npy'),mmap_mode='r')
        x_test  = np.load(os.path.join(fold_dir,'testdata.npy'),mmap_mode='r')
        y_test  = np.load(os.path.join(fold_dir,'testlabels.npy'),mmap_mode='r')

        x_train = np.nan_to_num(x_train,nan=0.0, posinf=0.0, neginf=0.0)
        x_test = np.nan_to_num(x_test,nan=0.0, posinf=0.0, neginf=0.0)
        # x_train, y_train, x_test, y_test = RD.read_data(
        #     raw_data_folder, files, r*4 + 1, (r+1)*4
        # )

        # np.random.seed(44)
        
        # np.random.shuffle(files)
        # x_train, y_train, x_test, y_test = RD.read_data(
        #     raw_data_folder, files, 1, 8
        # )
        

        # test_files = folds[r]
        # train_files = [f for i, f_list in enumerate(folds) if i != r for f in f_list]
        
        # x_train, y_train = load_patient_group(train_files)
        # x_test, y_test = load_patient_group(test_files)
        
        # Normalization
        global_dataset = False
        #set base values for this to let code work
        mean = np.nan
        std = np.nan
        if global_dataset:
            print("Global normalizing...")
            mean = x_train.mean()
            std = x_train.std()
            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

        global_min = float(x_train.min())
        global_min = float(np.percentile(x_train, 0.01))# Note original gives min of -373-> possible artifact, so use percentiles
        print("Global minimum (train set):", global_min)
        # global_max = float(np.percentile(x_train, 99.99))# Note original gives min of -373-> possible artifact, so use percentiles
        # global_max = float(x_train.max())
        # print("Global max (train set):", global_max)
        shift = -global_min + 1
        neg_rate = np.mean(x_train + shift < 0) #we find 8e-5 neg rate-> acceptable, only a few samples are artefacted by this
        print("neg_rate:", neg_rate)

        # Prepare Tensors (Add channel dimension for GAT)
        
        torch.cuda.empty_cache() # Clear any residual CUDA errors
        #Code below was for original native torch approach
        # x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1).to(device)
        # y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
        # x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1).to(device)
        # y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
        # y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        # x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)
        # y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        #Code below is for PyG
        print("Preparing data...")
        traindata = MyUtils.prepare_graphs_labels(x_train,y_train,Model.adj)
        testdata = MyUtils.prepare_graphs_labels(x_test,y_test,Model.adj)
        del x_train, y_train, x_test, y_test
        
        # print(traindata[1].x.shape) = 12,384
        # debug
        # print(x_train.shape)
        # print(y_train.shape)
        # print(x_test.shape)
        # print(y_test.shape)
        

        # train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
        num_workers = 1 #set to 1 if running on cluster
        pin_mem=True
        train_loader = DataLoader(
            traindata, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=pin_mem,  # Keeps data in "page-locked" memory for faster GPU transfer
            num_workers=num_workers, # This tells Python to use your extra cores
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        test_loader =DataLoader(
            testdata, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=pin_mem,  # Keeps data in "page-locked" memory for faster GPU transfer
            num_workers=num_workers, # This tells Python to use your extra cores
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
    )
        # Initialize Model, Loss, and Optimizer
        # model = Model.EEG_GAT_Model(init_adj).to(device)
        # model = Model.EEG_GAT_Model().to(device)
        # model_key = str(model_type).strip().lower()
        if model_type == "base":
            model = Model.EEG_GAT_Model().to(device)
            criterion_train = Model.BinaryFocalLoss(gamma=2.0, alpha=0.4, from_logits=True)
            criterion_val = Model.BinaryFocalLoss(gamma=2.0, alpha=0.4, from_logits=True)

        elif model_type=="SENNrawx":
            model = Model.SENN_raw(global_min=global_min).to(device)
            l1 = rob_loss
            print(f"l1 reg: {l1}")
            criterion_train = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)
            criterion_val = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)

        elif model_type=="SENNfixed":
            model = Model.SENN_fixedconcepts().to(device)
            l1 = rob_loss
            print(f"l1 reg: {l1}")
            criterion_train = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)
            criterion_val = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)

        elif model_type=="SENNtrivialfixed":
            model = Model.SENN_trivialfixedconcepts().to(device)
            l1 = rob_loss
            print(f"l1 reg: {l1}")
            criterion_train = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)
            criterion_val = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)

        elif model_type=="SENNfixed_concepttheta":
            model = Model.SENN_fixedconcepts_concepttheta().to(device)
            model = Model.SENN_fixedconcepts_concepttheta().to(device)
            l1 = rob_loss
            if l1 == 0.0:
              criterion_train = Model.BinaryFocalLoss(gamma=2.0, alpha=0.4, from_logits=True)
              criterion_val = Model.BinaryFocalLoss(gamma=2.0, alpha=0.4, from_logits=True)
              print("Using focal loss without robustness regularization.")
            else:
              criterion_train = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)
              criterion_val = Model.SENNLOSS(gamma=2.0, alpha=0.4, lambda1=l1, lambda2=0.0, from_logits=True, model=model)
            

        elif model_type== "LogisticConcepts":
            model = Model.ConceptLogisticDual().to(device)
            criterion_train = Model.BinaryFocalLoss(gamma=2.0, alpha=0.4, from_logits=True)
            criterion_val = Model.BinaryFocalLoss(gamma=2.0, alpha=0.4, from_logits=True)
            print("Using focal loss without robustness regularization.")

        else:
            raise ValueError(f"Model type is not supported: {model_type}")

        train_requires_input_grad = rob_loss != 0.0
        val_requires_input_grad = rob_loss != 0.0
        print(f"Training model: {model_type}")


        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # Note change to adamW due to weigth decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( #We say scheduling makes training more stable but plateau too early we therefore go to cosine annealing
        #     optimizer,
        #     mode='max',
        #     factor=0.2,
        #     patience=5,
        #     min_lr=1e-6
        # )

        
        cosine_epochs = epochs - warmup_epochs

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,   # start at 1% of base lr
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        # scheduler =  torch.optim.lr_scheduler.LinearLR(
        #     optimizer,
        #     start_factor=0.01,   # start at 1% of base lr
        #     end_factor=1.0,
        #     total_iters=warmup_epochs)
        
        early_stopper = MyUtils.EarlyStopping(patience=patience_earlyStop, min_delta=0.0005) #we set patience to 50 since that was original epoch length, but we allow up to 100
        # early_stopper = MyUtils.EarlyStoppingValLoss(patience=patience_earlyStop, min_delta=0.00005)

        fold_best_auroc = 0.0
        fold_best_val_loss = 1e6
        fold_history = {
            'loss': [], 'val_loss': [], 'val_auroc': [],'val_auprc': []
        }
        
        fold_save_path = os.path.join(model_dir, f"GAT_CV_10_{r}")
        os.makedirs(fold_save_path, exist_ok=True)
        print("Start training...")
        for epoch in range(epochs):
            # Training Phase
            model.train()
            running_train_loss = 0.0
            # for xb, yb in train_loader:
            for batch in train_loader:
                
                batch = batch.to(device, non_blocking=True)
                # print(xb.shape)
                # print(yb.shape)
                # xb = xb.to(device)
                # yb = yb.to(device)
                
                optimizer.zero_grad()
                batch.x = batch.x.requires_grad_(train_requires_input_grad)
                # print(batch.x.shape) => 6144,384
                # print("Calculating forward pass...")
                out = model(batch.x,batch.edge_index,batch.batch)
                
                if not torch.isfinite(out["logit"]).all():
                    print("NaN/Inf in logit")
                    print("x finite:", torch.isfinite(batch.x).all().item())
                    for k, v in out.items():
                        if torch.is_tensor(v):
                            print(k, torch.isfinite(v).all().item(), v.shape, v.min().item(), v.max().item())
                    raise ValueError("Non-finite model output")
                #for general backwards compatibility to base model use key_word arguments and caputre them in **kwargs for the base loss
                #out.get() will give None since output base is dict
                # print("Calculating loss...")
                loss = criterion_train(out["logit"], batch.y.unsqueeze(1), x=batch.x,h_x=out.get("h_x"),theta_x=out.get("theta_x"),h_x_edge=out.get("h_x_edge"),theta_x_edge=out.get("theta_x_edge")) #+ l2_lambda*l2
                # print("Updating weights...")
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                # print("Going to validation...")
            # ... inside your epoch loop ...
            
            # Validation Phase
            model.eval()
            running_val_loss = 0.0
            all_probs, all_labels = [], []
            # with torch.no_grad():#no grad only works if we do not include robustnessloss in validaiton fase
            # for xb, yb in test_loader:
            for batch in test_loader:
                batch = batch.to(device, non_blocking=True)
                # xb = xb.to(device)
                # yb = yb.to(device)
                batch.x = batch.x.requires_grad_(val_requires_input_grad)
                out = model(batch.x,batch.edge_index,batch.batch)

             

                if not torch.isfinite(out["logit"]).all():
                    print("NaN/Inf in logit")
                    print("x finite:", torch.isfinite(batch.x).all().item())
                    for k, v in out.items():
                        if torch.is_tensor(v):
                            print(k, torch.isfinite(v).all().item(), v.shape, v.min().item(), v.max().item())
                    raise ValueError("Non-finite model output")
                
                # Fix: Use 'probs' from current forward pass, not 'preds' from training
                v_loss = criterion_val(out["logit"], batch.y.unsqueeze(1), x=batch.x,h_x=out.get("h_x"),theta_x=out.get("theta_x"),h_x_edge=out.get("h_x_edge"),theta_x_edge=out.get("theta_x_edge"))
                running_val_loss += v_loss.item()
                all_probs.append(out["prob"].detach().cpu())
                all_labels.append(batch.y.cpu())
                
                

            # Convert to numpy for threshold search
            probs_flat = torch.cat(all_probs).numpy().ravel()
            labels_flat = torch.cat(all_labels).numpy().ravel()

            # 1. SEARCH FOR BEST KAPPA THRESHOLD
            best_k = -1.0
            best_threshold = 0.5
            thresholds = np.arange(0.1, 0.99, 0.01)
            
            for t in thresholds:
                temp_preds = (probs_flat >= t).astype(int)
                temp_k = cohen_kappa_score(labels_flat, temp_preds)
                if temp_k > best_k:
                    best_k = temp_k
                    best_threshold = t

            # 2. CALCULATE REMAINING METRICS AT THE BEST THRESHOLD
            final_preds = (probs_flat >= best_threshold).astype(int)
            metrics = {
                'auprc' : average_precision_score(labels_flat,probs_flat),
                'auroc': roc_auc_score(labels_flat, probs_flat),
                'kappa': best_k,
                'threshold': best_threshold, # Save this!
                'recall': recall_score(labels_flat, final_preds),
                'precision': precision_score(labels_flat, final_preds),
                'f1': f1_score(labels_flat, final_preds)
            }

            avg_train_loss = running_train_loss / len(train_loader)
            avg_val_loss = running_val_loss / len(test_loader)
            
            # ... history logging code ...

            # 3. UPDATED CHECKPOINT: Save metrics and the calculated threshold
            # We still monitor AUROC for the "best" model, but save the Kappa-optimized threshold
            # scheduler.step(metrics['auprc'])
            scheduler.step()

            if metrics['auprc'] >= fold_best_auroc: # this was used for AUPRC training
                fold_best_auroc = metrics['auprc']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,  # Now includes the best threshold found
                    'normalization': {'mean': mean, 'std': std},
                    'model_type': model_type
                }
                torch.save(checkpoint, os.path.join(fold_save_path, "best_auprc.pt"))
                torch.save(model.state_dict(), os.path.join(fold_save_path, "best_model_auprc.pt"))

            if avg_val_loss <= fold_best_val_loss: # this was used for AUPRC training
                fold_best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,  # Now includes the best threshold found
                    'normalization': {'mean': mean, 'std': std},
                    'model_type': model_type
                }
                torch.save(checkpoint, os.path.join(fold_save_path, "best.pt"))
                torch.save(model.state_dict(), os.path.join(fold_save_path, "best_model.pt"))

            

            print(f"Ep {epoch:02d} | Val Loss: {avg_val_loss:.4f} | AUROC: {metrics['auroc']:.4f} | AUPRC: {metrics['auprc']:.4f} | Best Kappa: {metrics['kappa']:.4f} at {metrics['threshold']:.2f}")
            fold_history['loss'].append(avg_train_loss)
            fold_history['val_loss'].append(avg_val_loss)
            fold_history['val_auroc'].append(metrics['auroc'])
            fold_history['val_auprc'].append(metrics['auprc'])

            # Early Stopping check
            early_stopper(metrics['auprc'])
            # early_stopper(avg_val_loss)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        # Save the full history for this fold
        with open(os.path.join(history_dir, f"history_cv_10_{r}.json"), 'w') as f:
            json.dump(fold_history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["base", "SENNrawx","SENNfixed","SENNtrivialfixed","SENNfixed_concepttheta","LogisticConcepts"], required=True)
    parser.add_argument("--LR", type=float, default=2e-3,required=True)
    parser.add_argument("--WD", type=float, default=1e-4,required=True)
    parser.add_argument("--RobLoss", type=float, default=0.0)

    args = parser.parse_args()
    
    config_params = {'randseeddata': 44, 'randseedother': 44}#, 'batch_size': 1, 'nchannels': 19, 'input_sec': 2, 'sfreq': 256}
    # config_params['input_timepoints'] = config_params['input_sec'] * config_params['sfreq']
    set_random_seed(config_params)
    # cross_validate(model_type="base")
    # cross_validate(model_type="SENNrawx")
    # cross_validate(model_type="SENNfixed")
    # cross_validate(model_type = "SENNtrivialfixed")
    cross_validate(model_type = args.model_type,lr = args.LR, wd = args.WD,rob_loss=args.RobLoss)
    # cross_validate(model_type = "LogisticConcepts")
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# import numpy as np 
# import scipy
# import pandas as pd
# import os
# import tensorflow_addons as tfa
# from keras import regularizers
# import Read_Data as RD

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

# channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]
# indices =[[r,i] for r,c1 in enumerate(channel_names) for i,c2 in enumerate(channel_names) if (c1.split("-")[0]==c2.split("-")[1] or c1.split("-")[1]==c2.split("-")[1] 
#           or c1.split("-")[0]==c2.split("-")[0] or c1.split("-")[1]==c2.split("-")[0])]
# adj=np.zeros((12,12))
# for i in indices:
#     adj[i[0]][i[1]]=1
# adj=tf.constant(adj,dtype=tf.float32)

# class GATLayer(layers.Layer):

#     def __init__(self,output_dim):
#         super(GATLayer, self).__init__()
#         self.output_dim = output_dim
#         self.Leakyrelu = layers.LeakyReLU(alpha=0.2)
    
#     def build(self, input_shape):
#         self.W = self.add_weight(name='W',shape=(input_shape[-1], self.output_dim), initializer='random_normal',trainable=True)
#         self.a = self.add_weight(name='a',shape=(2*self.output_dim, 1), initializer='random_normal',trainable=True)
    
#     def call(self,input,adj):
#         H= tf.matmul(input, self.W)
#         h1=tf.tile(tf.expand_dims(H, axis=1), [1,12,1,1])
#         h2=tf.tile(tf.expand_dims(H, axis=2), [1,1,12,1])
#         result =tf.concat([h1 , h2], axis=-1)
#         e=self.Leakyrelu(tf.squeeze(tf.matmul(result, self.a),axis=-1))
#         zero_mat= -1e9*tf.ones_like(e)
#         msked_e=tf.where(adj==1.0,e,zero_mat)
#         alpha=tf.nn.softmax(msked_e,axis=-1)
#         HPrime=tf.matmul(alpha,H)
#         return tf.nn.elu(HPrime)

# def create_model():
#     Input= keras.Input(shape=(12,384,1))
#     regularizer_dense=regularizers.l2(0.0001)

#     x= layers.Conv2D(32,(1,5),activation='relu',padding='same')(Input)
#     y= layers.Conv2D(32,(1,7),activation='relu',padding='same')(Input)
#     x= layers.add([x,y])
#     x= layers.AveragePooling2D((1,2))(x)
#     x= layers.BatchNormalization()(x)
#     x= layers.SpatialDropout2D(0.2)(x)

#     x= layers.Conv2D(64,(1,5),activation='relu',padding='same')(x)
#     y= layers.Conv2D(64,(1,7),activation='relu',padding='same')(x)
#     x= layers.add([x,y])
#     x= layers.AveragePooling2D((1,2))(x)
#     x= layers.BatchNormalization()(x)
#     x= layers.SpatialDropout2D(0.2)(x)

#     x= layers.Conv2D(8,(1,5),activation='relu',padding='same')(x)
#     y= layers.Conv2D(8,(1,7),activation='relu',padding='same')(x)
#     x= layers.add([x,y])
#     x= layers.AveragePooling2D((1,2))(x)
#     x= layers.BatchNormalization()(x)
#     x= layers.SpatialDropout2D(0.2)(x)

#     x= layers.Conv2D(1,(1,5),activation='relu',padding='same')(x)
#     y= layers.Conv2D(1,(1,7),activation='relu',padding='same')(x)
#     x= layers.add([x,y])
#     x= layers.AveragePooling2D((1,2))(x)
#     x= layers.Reshape((12,24))(x)

#     x= GATLayer(37)(x,adj)
#     x= layers.Dropout(0.2)(x)
#     x= GATLayer(32)(x,adj)
#     x= layers.Dropout(0.2)(x)
#     x= GATLayer(16)(x,adj)

#     x= layers.GlobalAveragePooling1D()(x)
#     x= layers.Dropout(0.2)(x)
#     x= layers.Dense(32,activation='relu',kernel_regularizer=regularizer_dense)(x)
#     x= layers.Dropout(0.2)(x)
#     x= layers.Dense(16,activation='relu',kernel_regularizer=regularizer_dense)(x)
#     x= layers.Dropout(0.2)(x)
#     x= layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizer_dense)(x)

#     model = keras.Model(inputs=Input, outputs=x)

#     optimizer=keras.optimizers.Adam(learning_rate=0.002)
#     loss=keras.losses.BinaryFocalCrossentropy(from_logits=False,gamma=2,alpha=0.4,apply_class_balancing=True)
#     kappa=tfa.metrics.CohenKappa(num_classes=2)
#     fp=keras.metrics.FalsePositives()
#     tn=keras.metrics.TrueNegatives()
#     precision = keras.metrics.Precision()
#     recall = keras.metrics.Recall()
#     AUROC = keras.metrics.AUC(curve='ROC', name = 'AUROC')
#     AUPRC = keras.metrics.AUC(curve='PR', name = 'AUPRC')
#     model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy', AUROC, AUPRC,fp,tn, precision, recall,kappa])   
#     return model

# folder="...Path to the folder containing the dataset..."
# files=os.listdir(folder)

# for r in range(10):
#     x_train,y_train,x_test,y_test=RD.read_data(folder,files,r*4+1,(r+1)*4)
#     mean=x_train.mean()
#     std=x_train.std()
#     x_train=(x_train-mean)/std
#     x_test=(x_test-mean)/std

#     x_train=np.expand_dims(x_train,axis=-1)
#     x_test=np.expand_dims(x_test,axis=-1)

#     np.random.seed(42)
#     train_indices = np.arange(x_train.shape[0])
#     np.random.shuffle(train_indices)
#     x_train = x_train[train_indices]
#     y_train = y_train[train_indices]
    
#     model=create_model()
#     checkpoint_path = f"Saved_models/GAT_CV_10_{r}"+"/cp_{epoch:03d}.ckpt"
#     checkpoint_dir = os.path.dirname(checkpoint_path) 
#     cp_callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=False,verbose=0,save_best_only=True,monitor='val_AUROC',mode='max')
#     history=model.fit(x_train,y_train,epochs=50,batch_size=512,verbose=1,validation_data=(x_test,y_test),callbacks=[cp_callback]) 
#     with open(f"History/history_cv_10_{r}.jason", 'w') as f:
#         pd.DataFrame(history.history).to_json(f)

    