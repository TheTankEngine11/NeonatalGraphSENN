# ===========================
# FULL TF-EQUIVALENT PYTORCH IMPLEMENTATION
# ===========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, GATv2Conv
from torch_geometric.utils import dense_to_sparse
import numpy as np
from typing import Union


# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Adjacency matrix 
# ---------------------------
channel_names = [
    "Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2",
    "Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"
]

indices = [
    [r, i]
    for r, c1 in enumerate(channel_names)
    for i, c2 in enumerate(channel_names)
    if (
        c1.split("-")[0] == c2.split("-")[1]
        or c1.split("-")[1] == c2.split("-")[1]
        or c1.split("-")[0] == c2.split("-")[0]
        or c1.split("-")[1] == c2.split("-")[0]
    )
]

adj = np.zeros((12, 12), dtype=np.float32)
for i, j in indices:
    adj[i, j] = 1.0


# # adj = torch.tensor(adj, device=device)
# edge_index, _ = dense_to_sparse(torch.tensor(adj)).to(device)


#############
# BASE MODEL
#############
class CNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1: Standard Parallel
        self.conv1a = nn.Conv1d(1, 32, kernel_size=5, padding='same')
        self.conv1b = nn.Conv1d(1, 32, kernel_size=7, padding='same')
        self.bn1 = nn.BatchNorm1d(32, momentum=0.01)

        # Block 2: Serial/Parallel Residual
        self.conv2a = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.conv2b = nn.Conv1d(64, 64, kernel_size=7, padding='same')
        self.bn2 = nn.BatchNorm1d(64, momentum=0.01)

        # Block 3: Serial/Parallel Residual
        self.conv3a = nn.Conv1d(64, 8, kernel_size=5, padding='same')
        self.conv3b = nn.Conv1d(8, 8, kernel_size=7, padding='same')
        self.bn3 = nn.BatchNorm1d(8, momentum=0.01)

        # Block 4: Final Feature Mapping
        self.conv4a = nn.Conv1d(8, 1, kernel_size=5, padding='same')
        self.conv4b = nn.Conv1d(1, 1, kernel_size=7, padding='same')

        self.pool = nn.AvgPool1d(2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (TotalNodes, Time) -> (TotalNodes, 1, Time)
        x = x.unsqueeze(1)

        #block 1
        xprime = F.relu(self.conv1a(x))
        y = F.relu(self.conv1b(x))
        x = self.pool(xprime + y)
        x = self.bn1(x)
        x = self.dropout(x)
        #block 2
        x = F.relu(self.conv2a(x))
        y = F.relu(self.conv2b(x))
        x = self.pool(x + y)
        x = self.bn2(x)
        x = self.dropout(x)
        #block 3
        x = F.relu(self.conv3a(x))
        y = F.relu(self.conv3b(x))
        x = self.pool(x + y)
        x = self.bn3(x)
        x = self.dropout(x)
        #block 4
        x = F.relu(self.conv4a(x))
        y = F.relu(self.conv4b(x))
        x = self.pool(x + y)

        

        return x.squeeze(1) # (TotalNodes, 24)
    
class GNN_Head(nn.Module):
    def __init__(self, in_channels=24):
        super().__init__()
        # self.gat1 = GATConv(in_channels, 37, heads=1, add_self_loops=False)
        # self.gat2 = GATConv(37, 32, heads=1, add_self_loops=False)
        # self.gat3 = GATConv(32, 16, heads=1, add_self_loops=False)
        self.gat1 = GATv2Conv(in_channels, 37, heads=1, add_self_loops=False)
        self.gat2 = GATv2Conv(37, 32, heads=1, add_self_loops=False)
        self.gat3 = GATv2Conv(32, 16, heads=1, add_self_loops=False)


        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        # GNN Layers
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat3(x, edge_index))

        # Graph Readout
        x = global_mean_pool(x, batch)

        # Final MLP
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        
        return self.fc3(x)
       

class EEG_GAT_Model(nn.Module):
    def __init__(self,logit_out=True):
        super().__init__()
        self.cnn = CNN_Encoder()
        self.gnn = GNN_Head(in_channels=24)
        self.logit_out = logit_out

    def forward(self, raw_features,edges,batch):
        # Pass raw node signals to CNN
        node_features = self.cnn(raw_features)
        logit = self.gnn(node_features, edges, batch)
        prob = torch.sigmoid(logit)
        # Pass features and graph structure to GNN
        
        out = {
            "logit": logit,
            "prob":prob
        }

        return out
    # def forward(self, data):
    #     # Pass raw node signals to CNN
    #     node_features = self.cnn(data.x)
        
    #     # Pass features and graph structure to GNN
    #     return self.gnn(node_features, data.edge_index, data.batch)
    
#############
# SENN MODEL
#############
class GNN_CORE(nn.Module):
    def __init__(self, in_channels=24):
        super().__init__()
        # self.gat1 = GATConv(in_channels, 37, heads=1, add_self_loops=False)
        # self.gat2 = GATConv(37, 32, heads=1, add_self_loops=False)
        # self.gat3 = GATConv(32, 16, heads=1, add_self_loops=False)
        self.gat1 = GATv2Conv(in_channels, 37, heads=1, add_self_loops=False)
        self.gat2 = GATv2Conv(37, 32, heads=1, add_self_loops=False)
        self.gat3 = GATv2Conv(32, 16, heads=1, add_self_loops=False)


        # self.fc1 = nn.Linear(16, 32)
        # self.fc2 = nn.Linear(32, 16)
        # self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch): #The GNNcore does not need batch for pooling anymore, but for consistency we still pass it
        # GNN Layers
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat3(x, edge_index))
        return x
       
class IdentityConceptizer(nn.Module):
    """
    Note we need to shift raw feautures upward such that we do not have zero crossings which would give problems in hammard product in aggregrator
    We have the option to do shifting per graph or per sample.
    Suppose shift of c -> prediction f = (x_z+c) * theta =  x_z*theta+c*Theta ; so we introduce bias c*theta
    If we do per sample, this would cloud the model since it could also learn somethingh based on the shift. This introduces a "hidden" concept
    And thus violates the notion that f(h(x),theta(x)) should be consiting of explainable and meaningfull concepts. 
    Global shift would introduce general c*theta(x_z); but this concepts c is consistent for all samples and thus can be considerd model bias, around which
    theta and h(x) can learn around. We argue that global bias (based on train set) is less detrimental to the model that having the multiplication problems around the zero crossings
    Note that to prevent data leakage we need to use global minimum from trainset; and test set might thus have zero crossings -> this thus influences generalizability
    Note that in the aggregrator we can compensate for the bias term by centering
    This problem is not present for defined base concepts which are not raw signal, which is a nice discussion point about model interpretability and faithfullness to senn.

    we could also use x**2 as ceoncpetizer, but then we are less faithfull to original model (evalauted with IG) where the input was really x: h(x)=x 
    And then our "concept" would not be the raw signal but the signal enegergy, which is a valid approach but not what we want to test, might produce better results though
    """
    def __init__(self,global_min):
        super().__init__()
        if global_min is not None:
            shift = -global_min+1 
        else: #if loading for the test set make sure it gets initialised without needing to track shift
            shift = 0.0
        print(f"shift = {shift}")
        self.register_buffer("shift",torch.tensor(shift))
    def forward(self,raw_features):
        h_x = raw_features + self.shift
        
        return torch.clamp(h_x, min=1e-12) #remove negative outliers (calculated 0.008% removed->all hihly likely artefacts)(make sure concepts do not cross zero for interpretabilit)

class FixedConceptizer(nn.Module):
    """
    This conceptizer returns a set of fixed base concepts
    These concepts should be based on domain knowledge 
    Note that for the robustness regularization, we need jacobian of J_x^h 
    The calculation of the concepts should thus be end-to-end differentiable
    In the original SENN paper h(x) is defined as (V)AE and thus automatically satisfies
    We however are thus constraints by (torch) differentiable functions to calculate concepts
    """
    def __init__(self,fs=32,N=384,nleo_smooth_len=7):
        super().__init__()
        self.fs = fs
        self.N = N
        self.nleo_smooth_len = nleo_smooth_len
    
    def forward(self,raw_features,edges):
        eps = 1e-8
        #raw_features.shape = (TotelNodes, Timesamples)
        #Raw features must be the raw time sample (may be filtered/prerpocessed of course)
        window = torch.hann_window(raw_features.shape[-1],device=raw_features.device)
        x = window*raw_features
        x_fft = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(x.size(-1), d=1.0 / self.fs).to(x.device)
        

        psd = (x_fft.abs().square()) # / (self.N* self.fs * window.square().sum()) #Possible correction

        mask = (freqs >= 1) & (freqs<= 16) #we only want to look in 1 to 16Hz range (range of our BP filter so outisde will be little skewed)

        #normalized psd:
        psd_norm = psd[:,mask] / psd[:,mask].sum(dim=-1, keepdim=True).clamp_min(eps)
        #total power in signal:
        power_total = torch.trapezoid(psd[:,mask],freqs[mask],dim=-1).clamp_min(eps)

        ####### NODE FEATURES #######
        # # Frequency band relative powers
        # delta 0.5-4Hz (1-4Hz due to preprocessnig)
        mask = (freqs >= 1) & (freqs<= 4)
        h_rbp_delta = torch.trapezoid(psd[:,mask],freqs[mask],dim=-1)/ power_total
        # Theta 4-8 Hz
        mask = (freqs >= 4) & (freqs<= 8)
        h_rbp_theta= torch.trapezoid(psd[:,mask],freqs[mask],dim=-1)/ power_total
        # Alpha 8-12Hz
        mask = (freqs >= 8) & (freqs<= 12)
        h_rbp_alpha= torch.trapezoid(psd[:,mask],freqs[mask],dim=-1)/ power_total
        # Beta 12-32Hz (capped to 16Hz due to preprocessing)
        mask = (freqs >= 12) & (freqs<= 16)
        h_rbp_beta= torch.trapezoid(psd[:,mask],freqs[mask],dim=-1)/ power_total

        # # Rythmicity (1-shannon entropy)
        spec_entropy = -1.* (psd_norm * psd_norm.log()).sum(dim=-1)
        #normalize entropy:
        K = psd_norm.shape[-1]
        spec_entropy = spec_entropy / (torch.log(torch.tensor(K,device=psd_norm.device,dtype=psd_norm.dtype)))
        h_ryth = 1-spec_entropy

        # # Signal power through SNLEO
        x0 = raw_features[:, 3:]    # S(n)
        x1 = raw_features[:, 2:-1]  # S(n-1)
        x2 = raw_features[:, 1:-2]  # S(n-2)
        x3 = raw_features[:, :-3]   # S(n-3)

        nleo = x1*x2 - x0*x3 
        snleo = F.avg_pool1d(nleo.unsqueeze(1),self.nleo_smooth_len,stride=1,padding=3).squeeze(1)

        h_snleo = snleo.abs().mean(dim=-1)
        # print(
        #     h_rbp_delta.shape,
        #     h_rbp_theta.shape,
        #     h_rbp_alpha.shape,
        #     h_rbp_beta.shape,
        #     h_snleo.shape,
        #     h_ryth.shape,
        #     h_snleo.shape,
        # )
        # safeguard
        h_rbp_delta = torch.nan_to_num(h_rbp_delta, nan=0.0, posinf=0.0, neginf=0.0)
        h_rbp_theta = torch.nan_to_num(h_rbp_theta, nan=0.0, posinf=0.0, neginf=0.0)
        h_rbp_alpha = torch.nan_to_num(h_rbp_alpha, nan=0.0, posinf=0.0, neginf=0.0)
        h_rbp_beta  = torch.nan_to_num(h_rbp_beta,  nan=0.0, posinf=0.0, neginf=0.0)
        h_ryth      = torch.nan_to_num(h_ryth,      nan=0.0, posinf=0.0, neginf=0.0)
        h_snleo     = torch.nan_to_num(h_snleo,     nan=0.0, posinf=0.0, neginf=0.0)
        # node_feats = torch.cat((h_rbp_delta,h_rbp_theta,h_rbp_alpha,h_rbp_beta,h_ryth,h_snleo),dim=-1)
        node_feats = torch.stack((h_rbp_delta,h_rbp_theta,h_rbp_alpha,h_rbp_beta,h_ryth,h_snleo),dim=-1)
        # print(node_feats.shape)
        ####### EDGE FEATURES #######
        # # Imaginary coherence (abs)
        src, dst = edges
        # assert torch.all(batch[src] == batch[dst]) #Check we have no leakage between samples
        S_xy = x_fft[src] * x_fft[dst].conj()
        S_xx = x_fft[src] * x_fft[src].conj()
        S_yy = x_fft[dst] * x_fft[dst].conj()
        iCoh = (S_xy.imag.abs()) / (S_xx*S_yy).real.clamp_min(eps).sqrt()
        h_iCoh = iCoh.mean(dim=-1,keepdim=True)

        
        h_iCoh_real = h_iCoh.real # we take real values to get rid of complex tensor dtype (values were strictly speaking already real)
        # safeguard
        edge_feats   = torch.nan_to_num(h_iCoh_real,     nan=0.0, posinf=0.0, neginf=0.0)
        


        return node_feats, edge_feats

    
class TrivialConceptizer(nn.Module):
    """
    Returns trivial constant concepts:
    - node concepts: all ones, shape [N_total, 6]
    - edge concepts: all ones, shape [E_total, 1]

    This is useful to test whether the fixed-concept SENN is mainly using h(x)
    as a calibration/gating factor while the relevance network carries the real signal.
    """
    def __init__(self, n_node_concepts=6, n_edge_concepts=1):
        super().__init__()
        self.n_node_concepts = n_node_concepts
        self.n_edge_concepts = n_edge_concepts

    def forward(self, raw_features, edges):
        # raw_features: [N_total, T]
        # edges: [2, E_total]
        n_nodes = raw_features.shape[0]
        n_edges = edges.shape[1]

        node_feats = torch.ones(
            (n_nodes, self.n_node_concepts),
            dtype=raw_features.dtype,
            device=raw_features.device,
        )

        edge_feats = torch.ones(
            (n_edges, self.n_edge_concepts),
            dtype=raw_features.dtype,
            device=raw_features.device,
        )

        return node_feats, edge_feats

class theta_STGAT(nn.Module):
    def __init__(self,k_concepts=24, n_fc_layers=3, hidden_dim=32):#24 is base sicne it represents time encoding from cnn part
        super().__init__()
        self.cnn = CNN_Encoder()
        self.gnn = GNN_CORE(in_channels=24) #out is 16

        layers = []
        in_dim = 16

        for _ in range(n_fc_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        # final layer : note no elu or dropout since we want negative attributions
        layers.append(nn.Linear(in_dim, k_concepts))
        self.to_T_low = nn.Sequential(*layers)

    def forward(self, raw_features,edges,batch):
        # Pass raw node signals to CNN
        node_features = self.cnn(raw_features)
        #pass through GAT core
        node_latent = self.gnn(node_features, edges, batch)
        #transform to concept space
        T_low = self.to_T_low(node_latent)
        # Pass features and graph structure to GNN
        return T_low
    
class theta_STGAT_dual(nn.Module):
    def __init__(self,k_node_concepts=6,k_edge_concepts=1, n_fc_layers=3, hidden_dim=32,edge_hidden_dim=32):
        super().__init__()
        self.cnn = CNN_Encoder()
        self.gnn = GNN_CORE(in_channels=24) #out is 16

        layers = []
        in_dim = 16
        ## Node Head
        for _ in range(n_fc_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        # final layer : note no elu or dropout since we want negative attributions
        layers.append(nn.Linear(in_dim, k_node_concepts))
        self.node_head = nn.Sequential(*layers)

        ## Edge head
        layers = []
        in_dim = 16 * 2 # we use recirpocal nodes: [z_src, z_dst]
        ## Node Head
        for _ in range(n_fc_layers - 1):
            layers.append(nn.Linear(in_dim, edge_hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        # final layer : note no elu or dropout since we want negative attributions
        layers.append(nn.Linear(in_dim, k_edge_concepts))
        self.edge_head = nn.Sequential(*layers)


    def forward(self, raw_features,edges,batch):
        # Pass raw node signals to CNN
        node_features = self.cnn(raw_features)
        #pass through GAT core
        node_latent = self.gnn(node_features, edges, batch)
        #transform to concept space
        theta_node = self.node_head(node_latent)
        src, dst = edges
        z_src = node_latent[src]
        z_dst = node_latent[dst]
        # edge_repr = torch.cat(
        #     [z_src, z_dst, z_src - z_dst, z_src * z_dst],
        #     dim=-1
        # ) 
        edge_repr = torch.cat(
            [z_src, z_dst],
            dim=-1
        )                                                       
        theta_edge = self.edge_head(edge_repr)                # [E_total, K_edge]
        return theta_node,theta_edge


class theta_GATConceptDual(nn.Module):
    """
    Concept-driven dual relevance network.

    Uses fixed node concepts as the graph node features for the relevance pathway,
    keeping a shared GNN backbone and separate node/edge heads.
    """
    def __init__(
        self,
        k_node_concepts=6,
        k_edge_concepts=1,
        concept_in_dim=6,
        backbone_in_dim=24,
        n_fc_layers=3,
        hidden_dim=32,
        edge_hidden_dim=32,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(concept_in_dim, backbone_in_dim),
            nn.ELU(),
            nn.Dropout(0.2),
        )
        self.gnn = GNN_CORE(in_channels=backbone_in_dim)  # out is 16

        layers = []
        in_dim = 16
        for _ in range(n_fc_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, k_node_concepts))
        self.node_head = nn.Sequential(*layers)

        layers = []
        in_dim = 16 * 2 + k_edge_concepts  # [z_src, z_dst, h_edge]
        for _ in range(n_fc_layers - 1):
            layers.append(nn.Linear(in_dim, edge_hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(0.2))
            in_dim = edge_hidden_dim
        layers.append(nn.Linear(in_dim, k_edge_concepts))
        self.edge_head = nn.Sequential(*layers)

    def forward(self, h_node, h_edge, edges, batch):
        node_features = self.node_encoder(h_node)
        z_node = self.gnn(node_features, edges, batch)
        theta_node = self.node_head(z_node)

        src, dst = edges
        z_src = z_node[src]
        z_dst = z_node[dst]
        edge_repr = torch.cat([z_src, z_dst, h_edge], dim=-1)
        theta_edge = self.edge_head(edge_repr)
        return theta_node, theta_edge, z_node
    
class UpsamplerInterpol(nn.Module):
    """
    Fixed (parameter-free) upsampling to avoid adding modeling capacity.
    """

    def __init__(self, t_out: int = 384, mode: str = "linear", align_corners: bool = False):
        super().__init__()
        self.t_out = t_out
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, t_low: torch.Tensor) -> torch.Tensor:
        """
        t_low:  (num_nodes, T_low)
        returns (num_nodes, T_out)
        """
        # interpolate expects (N, C, W) but PyG has (N,W) so squeeze and unsqueeze
        t = F.interpolate(
            t_low.unsqueeze(1),
            size=self.t_out,
            mode=self.mode,
            align_corners=self.align_corners if self.mode in ("linear", "bilinear", "bicubic", "trilinear") else None,
        )
        return t.squeeze(1)

class sennAggregrator(nn.Module):
    #In the aggregrator we need aggregrate nodes and generate explanatuions and probs/ logits 
    # Since we work in pytorch the data.batch tensor is our biggest friend
    def __init__(self, reduction,reduction_graph, center_theta=False,return_node_scores=True,return_fmap=True):
        super().__init__()
        self.center_theta = center_theta
        self.reduction = reduction
        self.reduction_graph = reduction_graph
        self.return_node_scores = return_node_scores
        self.return_fmap = return_fmap

    def forward(self, h_x, theta_x, batch, ):

        if self.center_theta:
            theta_x = theta_x - theta_x.mean(dim=1, keepdim=True)

        F_map = h_x * theta_x

        # time / concept reduction
        if self.reduction == "mean":
            node_score = F_map.mean(dim=1, keepdim=True)  # (TotalNodes, 1)
        else:
            node_score = F_map.sum(dim=1, keepdim=True) 
        

        # graph pooling (we choose mean instead of sum, since it is more conventional in graph networks)
        if self.reduction_graph == "mean":
            logit =  global_mean_pool(node_score, batch)  # (B, 1)
        else:
            logit =  global_add_pool(node_score, batch)
    
        prob = torch.sigmoid(logit)

        out = {
            "prob": prob,
            "logit": logit,
        }
        if self.return_fmap:
            out["F_map"]= F_map
        if self.return_node_scores:
            out["node_score"] = node_score

        return out


class sennAggregatorDual(nn.Module):
    """
    Dual-branch aggregator:
      node concepts -> node scores -> graph node logit
      edge concepts -> edge scores -> graph edge logit
      final logit   = node logit + edge logit
    """
    def __init__(
        self,
        concept_reduce="sum",   # "sum" or "mean" within node/edge concept axis
        graph_reduce="mean",    # "sum" or "mean" over nodes/edges
        center_theta_node=False,
        center_theta_edge=False,
        return_node_scores=True,
        return_edge_scores=True,
        return_fmap=True,
    ):
        super().__init__()
        self.concept_reduce = concept_reduce
        self.graph_reduce = graph_reduce
        self.center_theta_node = center_theta_node
        self.center_theta_edge = center_theta_edge
        self.return_node_scores = return_node_scores
        self.return_edge_scores = return_edge_scores
        self.return_fmap = return_fmap

    def _reduce_concepts(self, fmap):
        if self.concept_reduce == "sum":
            return fmap.sum(dim=-1, keepdim=True)
        elif self.concept_reduce == "mean":
            return fmap.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown concept_reduce: {self.concept_reduce}")

    def _pool_graph(self, scores, batch_idx):
        if self.graph_reduce == "mean":
            return global_mean_pool(scores, batch_idx)
        elif self.graph_reduce == "sum":
            return global_add_pool(scores, batch_idx)
        else:
            raise ValueError(f"Unknown graph_reduce: {self.graph_reduce}")

    def forward(
        self,
        h_node,
        theta_node,
        batch_node,
        h_edge,
        theta_edge,
        batch_edge,
    ):
        if self.center_theta_node:
            theta_node = theta_node - theta_node.mean(dim=-1, keepdim=True)

        if self.center_theta_edge:
            theta_edge = theta_edge - theta_edge.mean(dim=-1, keepdim=True)

        F_node = h_node * theta_node              # [N_total, 6]
        F_edge = h_edge * theta_edge              # [E_total, 1]

        node_score = self._reduce_concepts(F_node)   # [N_total, 1]
        edge_score = self._reduce_concepts(F_edge)   # [E_total, 1]

        logit_node = self._pool_graph(node_score, batch_node)   # [B, 1]
        logit_edge = self._pool_graph(edge_score, batch_edge)   # [B, 1]
        logit = logit_node + logit_edge

        out = {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "logit_node": logit_node,
            "logit_edge": logit_edge,
        }

        if self.return_fmap:
            out["F_node"] = F_node
            out["F_edge"] = F_edge
        if self.return_node_scores:
            out["node_score"] = node_score
        if self.return_edge_scores:
            out["edge_score"] = edge_score

        return out

class SENN_raw(nn.Module):
    def __init__(self,global_min=None,return_node_scores=False,return_fmap=False):
        super().__init__()
        if global_min is not None:
            shift = -global_min+1 
        else: #if loading for the test set make sure it gets initialised without needing to track shift
            shift = 0.0
        self.return_node_scores = return_node_scores
        self.return_fmap = return_fmap
        self.conceptizer = IdentityConceptizer(global_min=global_min)
        self.relevance = theta_STGAT(k_concepts=24,n_fc_layers=3,hidden_dim=32) #hidden dimensions for fc head
        self.upscaler = UpsamplerInterpol(t_out=384,mode="linear",align_corners=False)
        self.aggregrator = sennAggregrator(reduction = "sum",reduction_graph="mean",center_theta=True,return_node_scores=self.return_node_scores,return_fmap=self.return_fmap)

    def forward(self, raw_features,edges,batch):
        # Pass raw node signals to CNN
        h_x = self.conceptizer(raw_features)

        theta_x = self.relevance(raw_features,edges,batch)
        theta_x = self.upscaler(theta_x)#since we use raw input we need upscaler

        aggr_out= self.aggregrator(h_x=h_x,theta_x=theta_x,batch=batch) 


        out = {
            "prob": aggr_out["prob"],
            "logit": aggr_out["logit"],
            "h_x": h_x,
            "theta_x": theta_x,
        }
        if self.return_fmap:
            out["explanation"]=aggr_out["F_map"]
        if self.return_node_scores:
            out["node_score"] = aggr_out["node_score"] 
        
        
        return out

class SENN_fixedconcepts(nn.Module):
    def __init__(self,return_node_scores=False,return_edge_scores=False,return_fmap=False):
        super().__init__()
        self.return_node_scores = return_node_scores
        self.return_edge_scores = return_edge_scores
        self.return_fmap = return_fmap

        self.conceptizer = FixedConceptizer(fs=32,N=384,nleo_smooth_len=7)
        #Note we need to freeze concept encoder if present / learnable
        self.relevance = theta_STGAT_dual(k_node_concepts=6, k_edge_concepts=1, n_fc_layers=3, hidden_dim=32,edge_hidden_dim=32)#hidden dimensions for fc head
        
        self.aggregrator =  sennAggregatorDual(
            concept_reduce="sum",
            graph_reduce="mean",
            center_theta_node=False,
            center_theta_edge=False,
            return_node_scores=return_node_scores,
            return_edge_scores=return_edge_scores,
            return_fmap=return_fmap,
        )

    def forward(self, raw_features, edges, batch):
        h_x_node, h_x_edge = self.conceptizer(raw_features, edges)
        theta_x_node, theta_x_edge = self.relevance(raw_features, edges, batch)

        # PyG edge -> graph mapping
        edge_batch = batch[edges[0]]
        assert torch.equal(batch[edges[0]], batch[edges[1]]), "Cross-graph edges detected in batched edge_index"

        aggr_out = self.aggregrator(
            h_node=h_x_node,
            theta_node=theta_x_node,
            batch_node=batch,
            h_edge=h_x_edge,
            theta_edge=theta_x_edge,
            batch_edge=edge_batch,
        )

        out = {
            "prob": aggr_out["prob"],
            "logit": aggr_out["logit"],
            "h_x": h_x_node,
            "theta_x": theta_x_node,
            "h_x_edge": h_x_edge,
            "theta_x_edge": theta_x_edge,
            "logit_node": aggr_out["logit_node"],
            "logit_edge": aggr_out["logit_edge"],
        }

        if self.return_fmap:
            out["explanation"] = aggr_out["F_node"]
            out["explanation_edge"] = aggr_out["F_edge"]
        if self.return_node_scores:
            out["node_score"] = aggr_out["node_score"]
        if self.return_edge_scores:
            out["edge_score"] = aggr_out["edge_score"]

        
        
        # Pass features and graph structure to GNN
        return out
    
class SENN_trivialfixedconcepts(nn.Module):
    def __init__(self,return_node_scores=False,return_edge_scores=False,return_fmap=False):
        super().__init__()
        self.return_node_scores = return_node_scores
        self.return_edge_scores = return_edge_scores
        self.return_fmap = return_fmap

        self.conceptizer = TrivialConceptizer(n_node_concepts=6,n_edge_concepts=1)
        #Note we need to freeze concept encoder if present / learnable
        self.relevance = theta_STGAT_dual(k_node_concepts=6, k_edge_concepts=1, n_fc_layers=3, hidden_dim=32,edge_hidden_dim=32)#hidden dimensions for fc head
        
        self.aggregrator =  sennAggregatorDual(
            concept_reduce="sum",
            graph_reduce="mean",
            center_theta_node=False,
            center_theta_edge=False,
            return_node_scores=return_node_scores,
            return_edge_scores=return_edge_scores,
            return_fmap=return_fmap,
        )

    def forward(self, raw_features, edges, batch):
        h_x_node, h_x_edge = self.conceptizer(raw_features, edges)
        theta_x_node, theta_x_edge = self.relevance(raw_features, edges, batch)

        # PyG edge -> graph mapping
        edge_batch = batch[edges[0]]
        assert torch.equal(batch[edges[0]], batch[edges[1]]), "Cross-graph edges detected in batched edge_index"

        aggr_out = self.aggregrator(
            h_node=h_x_node,
            theta_node=theta_x_node,
            batch_node=batch,
            h_edge=h_x_edge,
            theta_edge=theta_x_edge,
            batch_edge=edge_batch,
        )

        out = {
            "prob": aggr_out["prob"],
            "logit": aggr_out["logit"],
            "h_x": h_x_node,
            "theta_x": theta_x_node,
            "h_x_edge": h_x_edge,
            "theta_x_edge": theta_x_edge,
            "logit_node": aggr_out["logit_node"],
            "logit_edge": aggr_out["logit_edge"],
        }

        if self.return_fmap:
            out["explanation"] = aggr_out["F_node"]
            out["explanation_edge"] = aggr_out["F_edge"]
        if self.return_node_scores:
            out["node_score"] = aggr_out["node_score"]
        if self.return_edge_scores:
            out["edge_score"] = aggr_out["edge_score"]

        
        
        
        return out


class SENN_fixedconcepts_concepttheta(nn.Module):
    """
    Fixed-concept SENN where the relevance pathway is driven by the fixed concepts
    instead of the raw EEG input.
    """
    def __init__(self, return_node_scores=False, return_edge_scores=False, return_fmap=False):
        super().__init__()
        self.return_node_scores = return_node_scores
        self.return_edge_scores = return_edge_scores
        self.return_fmap = return_fmap

        self.conceptizer = FixedConceptizer(fs=32, N=384, nleo_smooth_len=7)
        self.relevance = theta_GATConceptDual(
            k_node_concepts=6,
            k_edge_concepts=1,
            concept_in_dim=6,
            backbone_in_dim=24,
            n_fc_layers=3,
            hidden_dim=32,
            edge_hidden_dim=32,
        )
        self.aggregrator = sennAggregatorDual(
            concept_reduce="sum",
            graph_reduce="mean",
            center_theta_node=False,
            center_theta_edge=False,
            return_node_scores=return_node_scores,
            return_edge_scores=return_edge_scores,
            return_fmap=return_fmap,
        )

    def forward(self, raw_features, edges, batch):
        h_x_node, h_x_edge = self.conceptizer(raw_features, edges)
        theta_x_node, theta_x_edge, z_node = self.relevance(h_x_node, h_x_edge, edges, batch)

        edge_batch = batch[edges[0]]
        assert torch.equal(batch[edges[0]], batch[edges[1]]), "Cross-graph edges detected in batched edge_index"

        aggr_out = self.aggregrator(
            h_node=h_x_node,
            theta_node=theta_x_node,
            batch_node=batch,
            h_edge=h_x_edge,
            theta_edge=theta_x_edge,
            batch_edge=edge_batch,
        )

        out = {
            "prob": aggr_out["prob"],
            "logit": aggr_out["logit"],
            "h_x": h_x_node,
            "theta_x": theta_x_node,
            "h_x_edge": h_x_edge,
            "theta_x_edge": theta_x_edge,
            "z_node": z_node,
            "logit_node": aggr_out["logit_node"],
            "logit_edge": aggr_out["logit_edge"],
        }

        if self.return_fmap:
            out["explanation"] = aggr_out["F_node"]
            out["explanation_edge"] = aggr_out["F_edge"]
        if self.return_node_scores:
            out["node_score"] = aggr_out["node_score"]
        if self.return_edge_scores:
            out["edge_score"] = aggr_out["edge_score"]

        return out


class ConceptLogisticDual(nn.Module):
    """
    Maximally interpretable dual-branch logistic model on fixed concepts.

    Node concepts and edge concepts are pooled per graph and scored by a linear
    branch each. For evaluation compatibility, per-node and per-edge contribution
    tensors are also returned.
    """
    def __init__(self, return_node_scores=False, return_edge_scores=False, return_fmap=False):
        super().__init__()
        self.return_node_scores = return_node_scores
        self.return_edge_scores = return_edge_scores
        self.return_fmap = return_fmap

        self.conceptizer = FixedConceptizer(fs=32, N=384, nleo_smooth_len=7)
        self.node_linear = nn.Linear(6, 1, bias=False)
        self.edge_linear = nn.Linear(1, 1, bias=False)


        self.bias = nn.Parameter(torch.zeros(1))
    #Compatability functions for XAI methods created for senn-fc-theta(h); originally not used for training due to hardcoding in self.forward
    def relevance(self, h_node, h_edge, edges=None, batch=None):
        theta_node = self.node_linear.weight.expand(h_node.shape[0], -1)
        theta_edge = self.edge_linear.weight.expand(h_edge.shape[0], -1)
        return theta_node, theta_edge, None
    
    def aggregrator(self,h_node,theta_node,batch_node,h_edge,theta_edge,batch_edge,):
        F_node = h_node * theta_node
        F_edge = h_edge * theta_edge

        node_score = F_node.sum(dim=-1, keepdim=True)
        edge_score = F_edge.sum(dim=-1, keepdim=True)

        logit_node = global_mean_pool(node_score, batch_node)
        logit_edge = global_mean_pool(edge_score, batch_edge)
        logit = logit_node + logit_edge + self.bias

        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "logit_node": logit_node,
            "logit_edge": logit_edge,
            "F_node": F_node,
            "F_edge": F_edge,
            "node_score": node_score,
            "edge_score": edge_score,
        }

    def forward(self, raw_features, edges, batch):
        h_x_node, h_x_edge = self.conceptizer(raw_features, edges)

        edge_batch = batch[edges[0]]
        assert torch.equal(batch[edges[0]], batch[edges[1]]), "Cross-graph edges detected in batched edge_index"

        node_coeff = self.node_linear.weight
        edge_coeff = self.edge_linear.weight

        theta_x_node = node_coeff.expand(h_x_node.shape[0], -1)
        theta_x_edge = edge_coeff.expand(h_x_edge.shape[0], -1)

        F_node = h_x_node * theta_x_node
        F_edge = h_x_edge * theta_x_edge
        node_score = F_node.sum(dim=-1, keepdim=True)
        edge_score = F_edge.sum(dim=-1, keepdim=True)

        logit_node = global_mean_pool(node_score, batch)
        logit_edge = global_mean_pool(edge_score, edge_batch)
        logit = logit_node + logit_edge + self.bias

        #Note that below we do mathematically the same thing, but this way we have both local focus map 
        # and acces to the pooled contributions that actually make up the final prediction
        pooled_node_concepts = global_mean_pool(h_x_node, batch)
        pooled_edge_concepts = global_mean_pool(h_x_edge, edge_batch)
        pooled_node_contributions = pooled_node_concepts * node_coeff.expand(pooled_node_concepts.shape[0], -1)
        pooled_edge_contributions = pooled_edge_concepts * edge_coeff.expand(pooled_edge_concepts.shape[0], -1)

        out = {
            "prob": torch.sigmoid(logit),
            "logit": logit,
            "h_x": h_x_node,
            "theta_x": theta_x_node,
            "h_x_edge": h_x_edge,
            "theta_x_edge": theta_x_edge,
            "logit_node": logit_node,
            "logit_edge": logit_edge,
            "pooled_node_concepts": pooled_node_concepts,
            "pooled_edge_concepts": pooled_edge_concepts,
            "pooled_node_contributions": pooled_node_contributions,
            "pooled_edge_contributions": pooled_edge_contributions,
            "node_coefficients": node_coeff,
            "edge_coefficients": edge_coeff,
        }

        if self.return_fmap:
            out["explanation"] = F_node
            out["explanation_edge"] = F_edge
        if self.return_node_scores:
            out["node_score"] = node_score
        if self.return_edge_scores:
            out["edge_score"] = edge_score

        return out
# ---------------------------
# LOSS FUNCTIONS
# ---------------------------
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.4,from_logits=False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def forward(self, inputs, targets,  **kwargs):
        # inputs: (batch, 1) - probabilities from sigmoid
        # targets: (batch, 1) - float labels
        eps = 1e-7
        
        targets = targets.float()
        # print(f"input size {inputs.shape}")
        # print(f"Target size {targets.shape}")
        # return False #debug

        # 1. Calculate Standard BCE
        # Keras: - (y_true * log(p) + (1-y_true) * log(1-p))
        # bce = -(targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(inputs,targets,reduction='none')
            probs = torch.sigmoid(inputs)

        else:
            probs = torch.clamp(inputs, eps, 1.0 - eps)
            bce = F.binary_cross_entropy(probs,targets,reduction='none')

        # 2. Calculate Modulating Factor (1 - pt)^gamma
        # pt is the probability of the true class
        pt = (probs*targets) + ((1 - probs)*(1 - targets))
        modulating_factor = (1.0 - pt) ** self.gamma

       
        # In BinaryFocalCrossentropy, alpha weights the positive class 
        # while (1-alpha) weights the negative class. Alpha thus introduces extra class balancing
        alpha_weight = (targets * self.alpha) + ((1.0 - targets) * (1.0 - self.alpha))
        
        # Combine: alpha * balancing * modulating * bce
        loss = alpha_weight * modulating_factor * bce
        
        return loss.mean()
    

# class RobustnessLoss(nn.Module):
#     """
#     This fucntion aims to implement robustness loss:L = || grad_x(f(x)) -  theta(x)^T J_x^h(x) ||_p
#     with f(x) is the prediction of sample x, theta are the relevance scores
#     and h(x) are the cocnepts

#     We use logits such that model local sensitivity is faithfully refelcted, otherwise 
#     we get induced zero due to sigmoid at zero and one
#     """
#     def __init__(self,reduction_type):
#         super().__init__()
#         self.reduction = reduction_type
#     def forward(self, x, concepts, thetas, logits):
#         """
#         x:        input tensor, must have requires_grad=True
#         concepts: h(x), shape (B, K, ...) or (N, K) depending on setup
#         thetas:   theta(x), same shape as concepts
#         logits:   model output logits, shape (B, 1) or (B,)
#         """
#         if not x.requires_grad:
#             raise ValueError("x must have requires_grad=True for robustness loss")

#         if logits.dim() == 2 and logits.size(-1) == 1:
#             logits = logits.squeeze(-1)

#         # df/dx
#         grad_f = torch.autograd.grad(
#             outputs=logits.sum(),
#             inputs=x,
#             create_graph=True,
#             retain_graph=True,
#         )[0]

        

#         #  theta(x)^T J_h(x) via VJP , is more efficient than for looping
#         jht_theta = torch.autograd.grad(
#             outputs=concepts,
#             inputs=x,
#             grad_outputs=thetas,
#             create_graph=True,
#             retain_graph=True,
#         )[0]
#         # print(x.shape)
#         # print(concepts.shape)
#         # print(thetas.shape)
#         # print(logits.shape)
#         # print(grad_f.shape)
#         # print(jht_theta.shape)
#         # raise ValueError('A very specific bad thing happened.')
#         diff = grad_f - jht_theta

#         # print(diff.shape)
        
#         per_sample = torch.norm(diff.flatten(start_dim=1), p=2, dim=1)

#         if self.reduction == "mean":
#             return per_sample.mean()
#         return per_sample.sum()
    
class RobustnessLoss(nn.Module):

    """
    Robustness loss for two-branch fixed-concept SENN:
      || d f / d x  -  (theta_node^T J_x h_node + theta_edge^T J_x h_edge) ||_2
      Works also on identity conceptizer but extends to the extra edge concepts as well, since G is additively speerable for both we can calculate Jac seperatly
    """
    def __init__(self, reduction_type="mean"):
        super().__init__()
        self.reduction = reduction_type

    def forward(
        self,
        x,
        concepts, # node
        thetas,   # node
        logits,
        h_x_edge=None,
        theta_x_edge=None,
    ):
        if not x.requires_grad:
            raise ValueError("x must have requires_grad=True for robustness loss")

        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        grad_f = torch.autograd.grad(
            outputs=logits.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]

        jht_node = torch.autograd.grad(
            outputs=concepts,
            inputs=x,
            grad_outputs=thetas,
            create_graph=True,
            retain_graph=True,
        )[0]

        jht_total = jht_node

        if h_x_edge is not None and theta_x_edge is not None:
            jht_edge = torch.autograd.grad(
                outputs=h_x_edge,
                inputs=x,
                grad_outputs=theta_x_edge,
                create_graph=True,
                retain_graph=True,
            )[0]
            jht_total = jht_total + jht_edge

        diff = grad_f - jht_total
        per_sample = torch.norm(diff.flatten(start_dim=1), p=2, dim=1)

        if self.reduction == "mean":
            return per_sample.mean()
        return per_sample.sum()

class ReconLoss(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self):
        # For now not implemented since we use fixed or identity h(x)
        return 0.0
    
class SENNLOSS(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.4,lambda1 = 0.0,lambda2=0.0,from_logits=True,model = None):
        super().__init__()
        self.model = model
        self.focalLoss = BinaryFocalLoss(gamma=gamma,alpha=alpha,from_logits=from_logits)

        self.robustnessloss = RobustnessLoss(reduction_type="mean")
        self.lambda1 = lambda1 

        self.recon_loss = ReconLoss()
        self.lambda2 = lambda2

    def forward(self, inputs, targets, x=None, h_x=None, theta_x=None, h_x_edge=None, theta_x_edge=None):
        # inputs: (batch, 1) - probabilities/logits
        # targets: (batch, 1) - float labels

        #Calculate prediction focal loss
        pred_loss = self.focalLoss(inputs,targets)

        #Calculate robustness regularization loss
        rob_loss = inputs.new_zeros(())
        if self.lambda1 != 0.0:
   
            rob_loss = self.robustnessloss(x = x, concepts=h_x, thetas = theta_x, logits = inputs,h_x_edge=h_x_edge,theta_x_edge=theta_x_edge)

        #If h(x) is simultaneous learned AE we need reconstruction loss
        #For us we do not use this and therefore not implement this (returns 0)
        recon_loss = inputs.new_zeros(())
        if self.lambda2 != 0.0:
            recon_loss = self.recon_loss()


        debug = True #here we can check the gradients and losses from robustness loss and its ratio to pred loss.
        #Note that in our case 1e-4 gives rob_loss sensible values, but the gradients associated are very high and therefore make training not leaning anything
        if debug:
            params = [p for p in self.model.parameters() if p.requires_grad]

            def grad_norm(grads):
                total = 0.0
                for g in grads:
                    if g is not None:
                        total += g.pow(2).sum()
                return total.sqrt()

            # compute gradients separately
            pred_grads = torch.autograd.grad(pred_loss, params, retain_graph=True, allow_unused=True)
            rob_grads  = torch.autograd.grad(rob_loss,  params, retain_graph=True, allow_unused=True)

            pred_norm = grad_norm(pred_grads)
            rob_norm  = grad_norm(rob_grads)

            scaled_rob_norm = self.lambda1 * rob_norm
            ratio = scaled_rob_norm / (pred_norm + 1e-12)

            print(f"GradNorms | pred: {pred_norm:.3e} | rob: {rob_norm:.3e} | "
                f"λ*rob: {scaled_rob_norm:.3e} | ratio: {ratio:.3f}")
    # ---------------------------------------------------

            print(f"pred loss: {pred_loss.item()} | rob loss:{rob_loss.item()}| {self.lambda1* rob_loss.item()}")
        return pred_loss + self.lambda1 * rob_loss + self.lambda2 * recon_loss
    
# ---------------------------
# GAT Layer Custom Depreciated due to use of GATv2conv from PyG
# ---------------------------
class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim,init_adj=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(input_dim, output_dim))
        self.a = nn.Parameter(torch.empty(2 * output_dim, 1))

        nn.init.normal_(self.W, mean=0.0, std=0.05)
        nn.init.normal_(self.a, mean=0.0, std=0.05)

        self.leakyrelu = nn.LeakyReLU(0.2)

        if init_adj is None:
            # Placeholder – will be overwritten on load
            self.register_buffer("adj", torch.zeros(12, 12))
        else:
            self.register_buffer("adj", init_adj)
        


    def forward(self, x):
        H = torch.matmul(x, self.W)   # (B, N, D)
        B, N, D = H.shape
        h1 = H.unsqueeze(1).expand(-1, N, -1, -1)  # H_j
        h2 = H.unsqueeze(2).expand(-1, -1, N, -1)  # H_i
        concat = torch.cat([h1, h2], dim=-1)

        e = self.leakyrelu(torch.matmul(concat, self.a).squeeze(-1))
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(self.adj > 0.5, e, zero_vec)

        alpha = F.softmax(attention, dim=-1)
        return F.elu(torch.matmul(alpha, H))

