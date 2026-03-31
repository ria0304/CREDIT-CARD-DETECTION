import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, ConfusionMatrixDisplay, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.stats import wilcoxon
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import to_undirected, add_self_loops
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — pip install xgboost to enable.")

try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False

# ========================
# CONFIGURATION
# ========================
SEEDS         = [42, 123, 2024,1 7, 99]                     # list of integer seeds used for reproducible runs
HIDDEN_DIM    = 128                                             #Hidden dimension size for the GNN layers
NUM_LAYERS    = 3                                               #Number of graph convolution layers 
LEARNING_RATE = 0.001                                     # Initial learning rate for the AdamW optimiser.
EPOCHS        = 200                                                  #Maximum number of training epochs.
PATIENCE      = 20                                                  #Early stopping patience – if validation AUC does not improve for 20 epochs, training stops.
MC_SAMPLES    = 30                                             #Number of forward passes with dropout enabled to estimate uncertainty (Monte Carlo dropout).
DROPOUT       = 0.3                                                
DATASET       = 'elliptic'

PROJECT_DATA_PATH = r"C:\\Users\\Ria S\\OneDrive\\Attachments\\Desktop\\projects\\CREDIT CARD DETECTION\\data"
ELLIPTIC_PATH = r"C:\\Users\\Ria S\\.cache\\kagglehub\\datasets\\ellipticco\\elliptic-data-set\\versions\\1"

# ========================
# DATASET DISCOVERY
# ========================
def ensure_dataset(dataset_name):
    if not HAS_KAGGLEHUB:
        return None
    try:
        if dataset_name == 'elliptic':
            return kagglehub.dataset_download("ellipticco/elliptic-data-set")
    except Exception as e:
        print(f"Auto-download failed: {e}")
    return None

def find_dataset_path(dataset_name):
    project_path = os.path.join(PROJECT_DATA_PATH, dataset_name)
    if os.path.exists(project_path):
        return project_path
    if dataset_name == 'elliptic':
        for path in [ELLIPTIC_PATH,
                     r"C:\\Users\\Ria S\\.cache\\kagglehub\\datasets\\ellipticco\\elliptic-data-set",
                     os.path.join(PROJECT_DATA_PATH, 'elliptic')]:
            if not os.path.exists(path):
                continue
            if os.path.exists(os.path.join(path, 'elliptic_txs_features.csv')):
                return path
            for sub in (os.listdir(path) if os.path.isdir(path) else []):
                full = os.path.join(path, sub)
                if os.path.isdir(full) and os.path.exists(
                        os.path.join(full, 'elliptic_txs_features.csv')):
                    return full
    print(f"Dataset {dataset_name} not found locally. Attempting download...")
    return ensure_dataset(dataset_name)

# ========================
# ELLIPTIC DATASET LOADER
# ========================
def load_elliptic(path):
    feats_path   = os.path.join(path, 'elliptic_txs_features.csv')
    classes_path = os.path.join(path, 'elliptic_txs_classes.csv')
    edges_path   = os.path.join(path, 'elliptic_txs_edgelist.csv')

    if not os.path.exists(feats_path):
        raise FileNotFoundError(f"elliptic_txs_features.csv not found at {feats_path}")

    feats   = pd.read_csv(feats_path, header=None)
    classes = pd.read_csv(classes_path)

    feat_cols = ['txId', 'timestep'] + [f'f{i}' for i in range(feats.shape[1] - 2)]           #column names to the raw features DataFrame
    feats.columns = feat_cols
    classes['class'] = classes['class'].map({'1': 1, '2': 0, 'unknown': np.nan})

    # Merges the features DataFrame with the classes DataFrame on txId using a left join.
    #This ensures all transactions (even those without a label) are kept.
    all_nodes = feats.merge(classes, on='txId', how='left')
    all_nodes = all_nodes.rename(columns={'class': 'isFraud'})

    # Build a txId → contiguous index map over ALL nodes
    txid_to_idx = {tid: i for i, tid in enumerate(all_nodes['txId'])}

    # Build edge_index over ALL nodes (for full message passing)
    full_edge_index = None
    if os.path.exists(edges_path):
        el = pd.read_csv(edges_path, header=None)
        el.columns = ['txId1', 'txId2']
        # Keep only edges where both endpoints are in our node set
        mask = el['txId1'].isin(txid_to_idx) & el['txId2'].isin(txid_to_idx)
        el   = el[mask]
        src  = torch.tensor([txid_to_idx[t] for t in el['txId1']], dtype=torch.long)
        dst  = torch.tensor([txid_to_idx[t] for t in el['txId2']], dtype=torch.long)
        full_edge_index = torch.stack([src, dst], dim=0)
        full_edge_index = to_undirected(full_edge_index)   # undirected for GCN/GAT
        print(f"  Loaded edgelist: {el.shape[0]} directed → "
              f"{full_edge_index.shape[1]} undirected edges")

    # Raw feature matrix (all nodes)
    raw_feat_cols = [c for c in feat_cols if c.startswith('f')]
    X_all = all_nodes[raw_feat_cols].values.astype(np.float32)

    # Labelled-only df (for supervised training)
    labelled = all_nodes.dropna(subset=['isFraud']).copy()
    labelled['isFraud'] = labelled['isFraud'].astype(int)

    # Synthesise Card_ID / Merchant_City / Timestamp for velocity features
    labelled['Card_ID']       = labelled['timestep'].astype(str)
    labelled['Merchant_City'] = (labelled['txId'] % 500).astype(str)
    base = pd.Timestamp('2019-01-01')
    labelled['Timestamp'] = base + pd.to_timedelta(labelled['timestep'] * 2, unit='W')
    labelled['Amount']    = labelled['f0']
    labelled = labelled.sort_values('Timestamp').reset_index(drop=True)

    print(f"[Elliptic] {len(all_nodes)} total nodes | "
          f"{len(labelled)} labelled | Fraud ratio: {labelled['isFraud'].mean():.4f}")
    return labelled, X_all, full_edge_index, txid_to_idx, all_nodes

# ========================
# LOAD DATASET
# ========================
print(f"Loading dataset: {DATASET.upper()}...")
dataset_path = find_dataset_path(DATASET)
if dataset_path is None:
    raise FileNotFoundError(f"Dataset {DATASET} not found")
print(f"Using dataset path: {dataset_path}")

# Globals set by Elliptic loader
ELLIPTIC_X_ALL       = None                    # will hold the raw feature matrix for all nodes (labelled + unknown).
ELLIPTIC_EDGE_INDEX  = None            # will hold the undirected edge index for the full graph.
ELLIPTIC_TXID_TO_IDX = None            # dictionary mapping transaction IDs to contiguous node indices.
ELLIPTIC_ALL_NODES   = None             # full node DataFrame
ELLIPTIC_FEAT_COLS   = []

# Load Elliptic
df, ELLIPTIC_X_ALL, ELLIPTIC_EDGE_INDEX, ELLIPTIC_TXID_TO_IDX, ELLIPTIC_ALL_NODES = load_elliptic(dataset_path)
ELLIPTIC_FEAT_COLS = [c for c in df.columns if c.startswith('f')]

# ========================
# VELOCITY FEATURES :-features that capture the recent transaction behaviour of a card
# ========================
print("Computing velocity features...")
df = df.sort_values(['Card_ID','Timestamp'])

def rolling_count_1h(g):                                                                                                        #For a group of transactions belonging to one card, count how many transactions occurred in the previous hour
    ts     = g.set_index('Timestamp')
    result = ts.index.to_series().rolling('1h', closed='left').count().fillna(0)
    result.index = g.index
    return result

def rolling_sum_1h(g):                                                                                                             #it sums the Amount of transactions that occurred in the previous hour.
    result = g.set_index('Timestamp')['Amount'].rolling('1h', closed='left').sum().fillna(0)
    result.index = g.index
    return result

df['tx_count_1h']   = df.groupby('Card_ID', group_keys=False).apply(rolling_count_1h)
df['amount_sum_1h'] = df.groupby('Card_ID', group_keys=False).apply(rolling_sum_1h)
df['Amount_log']    = np.log1p(df['Amount'])
df['Hour_sin']      = np.sin(2 * np.pi * df['Timestamp'].dt.hour / 24)
df['Hour_cos']      = np.cos(2 * np.pi * df['Timestamp'].dt.hour / 24)

# ========================
# CHRONOLOGICAL SPLIT:- sets up all necessary data structures for the tabular baselines (MLP and XGBoost) and also prepares the label tensors and masks that will be used later in the graph building for the GNNs
# ========================
train_size = int(len(df) * 0.70)
val_size   = int(len(df) * 0.15)
train_df   = df.iloc[:train_size]
val_df     = df.iloc[train_size : train_size + val_size]
test_df    = df.iloc[train_size + val_size:]

velocity_cols = ['Amount_log', 'tx_count_1h', 'amount_sum_1h', 'Hour_sin', 'Hour_cos']

# Tabular features: velocity + all Elliptic raw features
tabular_cols = velocity_cols + ELLIPTIC_FEAT_COLS
print(f"  Using {len(tabular_cols)} features for tabular models")

y          = torch.tensor(df['isFraud'].values, dtype=torch.float)
train_mask = torch.zeros(len(df), dtype=torch.bool); train_mask[:train_size] = True
val_mask   = torch.zeros(len(df), dtype=torch.bool); val_mask[train_size:train_size+val_size] = True
test_mask  = torch.zeros(len(df), dtype=torch.bool); test_mask[train_size+val_size:] = True

X_train    = df.iloc[:train_size][tabular_cols].values
X_val      = df.iloc[train_size:train_size+val_size][tabular_cols].values
X_test     = df.iloc[train_size+val_size:][tabular_cols].values
y_train_np = df.iloc[:train_size]['isFraud'].values.astype(np.float32)
y_val_np   = df.iloc[train_size:train_size+val_size]['isFraud'].values.astype(np.float32)
y_test_np  = df.iloc[train_size+val_size:]['isFraud'].values.astype(np.float32)

scaler_tab = StandardScaler().fit(X_train)
X_train_s  = scaler_tab.transform(X_train)
X_val_s    = scaler_tab.transform(X_val)
X_test_s   = scaler_tab.transform(X_test)

# ========================
# GRAPH BUILDER — NODE CLASSIFICATION
# ========================
def build_node_graph(df, train_size, val_size):
    """
   Maps labelled transactions to global node indices.

   Scales raw features (166) using train‑only stats; adds zero‑padded velocity features (5) for all nodes → 171 features per node.

   Uses full undirected edge list.

   Labels: 0/1 for labelled, -1 for unknown.

   Creates train/val/test masks based on chronological split of labelled nodes.

    Returns PyG Data object for GNN training.
    """
    # Map labelled txIds to their global node indices
    labelled_global_idx = torch.tensor(
        [ELLIPTIC_TXID_TO_IDX[tid] for tid in df['txId']],
        dtype=torch.long
    )

    # Scale the full node feature matrix (fit on train global indices only)
    train_global = labelled_global_idx[:train_size].numpy()
    scaler_node  = StandardScaler().fit(ELLIPTIC_X_ALL[train_global])
    X_all_scaled = scaler_node.transform(ELLIPTIC_X_ALL).astype(np.float32)

    # Build velocity feature matrix for labelled nodes (zero-pad for unknown nodes)
    N_all = ELLIPTIC_X_ALL.shape[0]
    vel   = np.zeros((N_all, len(velocity_cols)), dtype=np.float32)
    vel_vals = StandardScaler().fit(
        df.iloc[:train_size][velocity_cols].values
    ).transform(df[velocity_cols].values)
    for i, gidx in enumerate(labelled_global_idx.numpy()):
        vel[gidx] = vel_vals[i]

    # Concatenate: 166 raw + 5 velocity = 171 features per node
    node_feats = np.concatenate([X_all_scaled, vel], axis=1)
    x = torch.tensor(node_feats, dtype=torch.float)

    # Subgraph: keep all edges (message passing over entire graph)
    edge_index = ELLIPTIC_EDGE_INDEX  # already undirected

    # Labels: -1 for unknown, 0/1 for labelled
    labels_full = torch.full((N_all,), -1, dtype=torch.float)
    for i, gidx in enumerate(labelled_global_idx.numpy()):
        labels_full[gidx] = float(df.iloc[i]['isFraud'])

    # Masks pointing into the global node index space
    tr_mask = torch.zeros(N_all, dtype=torch.bool)
    vl_mask = torch.zeros(N_all, dtype=torch.bool)
    te_mask = torch.zeros(N_all, dtype=torch.bool)
    tr_mask[labelled_global_idx[:train_size]] = True
    vl_mask[labelled_global_idx[train_size:train_size+val_size]] = True
    te_mask[labelled_global_idx[train_size+val_size:]] = True

    data = Data(x=x, edge_index=edge_index, y=labels_full,
                train_mask=tr_mask, val_mask=vl_mask, test_mask=te_mask)
    return data

# ========================
# METRICS
# ========================
def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        f = f1_score(y_true, (probs > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

def compute_metrics(y_true, probs, threshold=None):
    if threshold is None:
        threshold = find_best_threshold(y_true, probs)
    y_pred = (probs > threshold).astype(int)
    return {
        'Accuracy':     (y_pred == y_true).mean(),
        'Balanced Acc': balanced_accuracy_score(y_true, y_pred),
        'MCC':          matthews_corrcoef(y_true, y_pred),
        'Precision':    precision_score(y_true, y_pred, zero_division=0),
        'Recall':       recall_score(y_true, y_pred, zero_division=0),
        'F1':           f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC':      roc_auc_score(y_true, probs),
        'threshold':    threshold,
        'probs':        probs,
        'true':         y_true
    }

# ========================
# LOSS:-Handles class imbalance by down‑weighting easy examples and focusing on hard, misclassified ones.
# ========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce    = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt     = torch.exp(-F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'))
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()

# ========================
# MODEL A: EllipticGNN (proposed)
# ========================
class EllipticGNN(nn.Module):
    """
    3-layer Graph Attention Network for node-level fraud classification.
    Uses multi-head attention, residual connections, and layer norm.
    Operates on the full Elliptic transaction graph.
    """
    def __init__(self, in_dim, hidden_dim, num_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs  = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.skips  = nn.ModuleList()

        for i in range(num_layers):
            in_ch  = hidden_dim
            out_ch = hidden_dim // heads
            self.convs.append(
                GATConv(in_ch, out_ch, heads=heads, dropout=dropout,
                        add_self_loops=True, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.skips.append(
                nn.Linear(hidden_dim, hidden_dim, bias=False)
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def encode(self, x, edge_index):
        h = F.elu(self.input_proj(x))
        for conv, norm, skip in zip(self.convs, self.norms, self.skips):
            h2 = F.elu(conv(h, edge_index))
            h  = norm(h2 + skip(h))   # residual
            h  = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, data):
        h = self.encode(data.x, data.edge_index)
        return self.classifier(h).squeeze(-1)

    def mc_dropout_forward(self, data, n):
        self.train()
        with torch.no_grad():
            return torch.stack([self.forward(data) for _ in range(n)])

# ========================
# MODEL B: HomoGNN (ablation)
# ========================
class HomoGNN(nn.Module):
    """
    2-layer GraphSAGE ablation — same graph, simpler aggregation.
    """
    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.proj  = nn.Linear(in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head  = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x  = F.elu(self.proj(data.x))
        ei = data.edge_index
        h  = self.norm1(F.elu(self.conv1(x,  ei)) + x)
        h  = F.dropout(h, p=self.dropout, training=self.training)
        h  = self.norm2(F.elu(self.conv2(h,  ei)) + h)
        return self.head(h).squeeze(-1)

# ========================
# TRAINING HELPERS
# ========================
def train_node_gnn(model, data, pos_weight):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)

    best_auc, no_improve, best_state = 0.0, 0, None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data)
            val_probs  = torch.sigmoid(val_logits[data.val_mask]).cpu().numpy()
            val_y      = data.y[data.val_mask].cpu().numpy()

        if len(np.unique(val_y)) < 2:
            continue
        val_auc = roc_auc_score(val_y, val_probs)

        if val_auc > best_auc:
            best_auc   = val_auc
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

def eval_node_gnn(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs  = torch.sigmoid(logits[data.test_mask]).cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
    return compute_metrics(y_true, probs)

# ========================
# MULTI-SEED LOOP
# ========================
results = {m: [] for m in ['HeteroGNN', 'HomoGNN', 'MLP', 'XGBoost']}
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nRunning {len(SEEDS)} seeds on {device}...\n")

last_gnn_model = None
last_data      = None

# Build graph once
print("Building node graph...")
graph_data = build_node_graph(df, train_size, val_size)
graph_data = graph_data.to(device)
print(f"  Nodes: {graph_data.num_nodes} | Edges: {graph_data.num_edges} | "
      f"Features: {graph_data.num_node_features}")

in_dim = graph_data.num_node_features
n_pos  = float(y_train_np.sum())
n_neg  = float(len(y_train_np) - n_pos)

for seed in SEEDS:
    print(f"\n{'='*55}")
    print(f"  SEED {seed}")
    print(f"{'='*55}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    # ── EllipticGNN (proposed) ──
    print("  [EllipticGNN — 3-layer GAT]")
    model_a = EllipticGNN(in_dim, HIDDEN_DIM, num_layers=NUM_LAYERS,
                          heads=4, dropout=DROPOUT).to(device)
    train_node_gnn(model_a, graph_data, pw)
    ma = eval_node_gnn(model_a, graph_data)
    results['HeteroGNN'].append(ma)
    print(f"  → AUC={ma['ROC-AUC']:.4f}  F1={ma['F1']:.4f}  MCC={ma['MCC']:.4f}")

    # ── GraphSAGE ablation ──
    print("  [HomoGNN — GraphSAGE ablation]")
    torch.manual_seed(seed)
    model_b = HomoGNN(in_dim, HIDDEN_DIM, dropout=DROPOUT).to(device)
    train_node_gnn(model_b, graph_data, pw)
    mb = eval_node_gnn(model_b, graph_data)
    results['HomoGNN'].append(mb)
    print(f"  → AUC={mb['ROC-AUC']:.4f}  F1={mb['F1']:.4f}  MCC={mb['MCC']:.4f}")

    # ── MLP baseline ──
    print("  [MLP baseline]")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300,
                        random_state=seed, early_stopping=True,
                        validation_fraction=0.1)
    mlp.fit(X_train_s, y_train_np)
    mm = compute_metrics(y_test_np, mlp.predict_proba(X_test_s)[:, 1])
    results['MLP'].append(mm)
    print(f"  → AUC={mm['ROC-AUC']:.4f}  F1={mm['F1']:.4f}  MCC={mm['MCC']:.4f}")

    # ── XGBoost baseline (if available) ──
    if HAS_XGB:
        print("  [XGBoost baseline]")
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1),
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='auc',
            random_state=seed, verbosity=0
        )
        xgb.fit(X_train_s, y_train_np,
                eval_set=[(X_val_s, y_val_np)],
                verbose=False)
        xm = compute_metrics(y_test_np, xgb.predict_proba(X_test_s)[:, 1])
        results['XGBoost'].append(xm)
        print(f"  → AUC={xm['ROC-AUC']:.4f}  F1={xm['F1']:.4f}  MCC={xm['MCC']:.4f}")

    last_gnn_model = model_a
    last_data      = graph_data

# ========================
# AGGREGATE + WILCOXON
# ========================
def agg(res_list, name):
    keys = ['ROC-AUC', 'F1', 'MCC', 'Balanced Acc', 'Precision', 'Recall']
    out  = {}
    print(f"\n{'─'*55}")
    print(f"  {name}  (n={len(res_list)} seeds)")
    print(f"{'─'*55}")
    for k in keys:
        v = [r[k] for r in res_list]
        out[k] = (np.mean(v), np.std(v), v)
        print(f"  {k:<16}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    return out

print("\n\n" + "="*55)
print("FINAL AGGREGATED RESULTS")
print("="*55)
aggs = {m: agg(results[m], m) for m in results if results[m]}

# Wilcoxon: HeteroGNN vs best competitor
competitors = [m for m in ['XGBoost', 'MLP', 'HomoGNN'] if results[m]]
if competitors:
    best_comp = max(competitors, key=lambda m: aggs[m]['ROC-AUC'][0])
    gnn_aucs  = aggs['HeteroGNN']['ROC-AUC'][2]
    cmp_aucs  = aggs[best_comp]['ROC-AUC'][2]
    diffs = np.array(gnn_aucs) - np.array(cmp_aucs)
    if len(set(diffs)) > 1:
        stat, p = wilcoxon(gnn_aucs, cmp_aucs)
        sig = "✅ Significant (p<0.05)" if p < 0.05 else "❌ Not significant"
        print(f"\nWilcoxon HeteroGNN vs {best_comp}: p={p:.4f}  {sig}")

# ========================
# ABLATION TABLE
# ========================
print("\n\n" + "="*55)
print("TABLE 1 — Ablation Study (Mean ± Std, n=5 seeds)")
print("="*55)
print(f"{'Model':<22}{'AUC':>14}{'F1':>14}{'MCC':>14}")
print("─"*64)
order = ['MLP', 'XGBoost', 'HomoGNN', 'HeteroGNN']
for m in order:
    if m not in aggs:
        continue
    a   = aggs[m]
    auc = f"{a['ROC-AUC'][0]:.4f}±{a['ROC-AUC'][1]:.4f}"
    f1  = f"{a['F1'][0]:.4f}±{a['F1'][1]:.4f}"
    mcc = f"{a['MCC'][0]:.4f}±{a['MCC'][1]:.4f}"
    tag = "  ours" if m == 'HeteroGNN' else ""
    print(f"{m:<22}{auc:>14}{f1:>14}{mcc:>14}{tag}")

# ========================
# MC DROPOUT + CALIBRATION
# ========================
print("\nComputing MC Dropout uncertainty...")
logits_mc  = last_gnn_model.mc_dropout_forward(last_data, MC_SAMPLES)
std_mc     = torch.sigmoid(logits_mc).std(dim=0)
test_unc   = std_mc[last_data.test_mask].cpu().numpy()
print(f"Mean uncertainty: {test_unc.mean():.4f}")

last_gnn_res = results['HeteroGNN'][-1]
prob_true_cal, prob_pred_cal = calibration_curve(
    last_gnn_res['true'], last_gnn_res['probs'], n_bins=10)
ece = np.mean(np.abs(prob_true_cal - prob_pred_cal))
print(f"ECE: {ece:.4f}")

# ========================
# FIGURES
# ========================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'Fraud Detection Results — {DATASET.upper()} Dataset',
             fontsize=14, fontweight='bold')

# ROC comparison
ax = axes[0, 0]
for mname, color, ls in [('HeteroGNN', '#e74c3c', '-'), ('HomoGNN', '#3498db', '--'),
                          ('MLP', '#2ecc71', ':'), ('XGBoost', '#f39c12', '-.')]:
    if not results[mname]:
        continue
    r = results[mname][-1]
    fpr, tpr, _ = roc_curve(r['true'], r['probs'])
    ax.plot(fpr, tpr, color=color, ls=ls, label=f"{mname} AUC={r['ROC-AUC']:.3f}")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_title('ROC Curve Comparison'); ax.legend(fontsize=8)
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')

# Ablation bar
ax = axes[0, 1]
metrics_bar = ['ROC-AUC', 'F1', 'MCC']
x = np.arange(len(metrics_bar))
w = 0.8 / len(aggs)
colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']
for i, (mname, color) in enumerate(zip(order, colors)):
    if mname not in aggs:
        continue
    means = [aggs[mname][k][0] for k in metrics_bar]
    stds  = [aggs[mname][k][1] for k in metrics_bar]
    ax.bar(x + i * w, means, w, label=mname, color=color, yerr=stds, capsize=3)
ax.set_xticks(x + w * 1.5); ax.set_xticklabels(metrics_bar)
ax.set_title('Ablation Study'); ax.legend(fontsize=8); ax.set_ylim(0, 1.1)

# Calibration
ax = axes[0, 2]
ax.plot(prob_pred_cal, prob_true_cal, 'o-', label=f'EllipticGNN (ECE={ece:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
ax.set_title('Calibration Curve'); ax.legend()
ax.set_xlabel('Mean predicted prob'); ax.set_ylabel('Fraction positives')

# Uncertainty histogram
ax = axes[1, 0]
ax.hist(test_unc, bins=30, alpha=0.8, edgecolor='black', color='#3498db')
ax.set_title('MC Dropout Uncertainty (Test Set)')
ax.set_xlabel('Std Dev'); ax.set_ylabel('Count')

# Confusion matrix
ax = axes[1, 1]
best_thresh = last_gnn_res.get('threshold', 0.5)
ConfusionMatrixDisplay.from_predictions(
    last_gnn_res['true'],
    (last_gnn_res['probs'] > best_thresh).astype(int), ax=ax
)
ax.set_title(f'EllipticGNN Confusion Matrix (t={best_thresh:.2f})')

# AUC box plot
ax = axes[1, 2]
plot_data  = [aggs[m]['ROC-AUC'][2] for m in order if m in aggs]
plot_names = [m for m in order if m in aggs]
ax.boxplot(plot_data, patch_artist=True,
           boxprops=dict(facecolor='#3498db', alpha=0.6))
ax.set_xticklabels(plot_names, rotation=15)
ax.set_title('AUC Distribution (5 seeds)')
ax.set_ylabel('ROC-AUC')

plt.tight_layout()
plt.savefig('research_results.png', dpi=150)
print("\nSaved research_results.png")

torch.save(last_gnn_model.state_dict(), 'best_model.pt')
print("Saved best_model.pt")
