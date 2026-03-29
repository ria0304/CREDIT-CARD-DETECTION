Credit Card Fraud Detection using Graph Neural Networks

Overview  
This project implements an advanced credit card fraud detection system using Graph Neural Networks (GNNs) and machine learning models. It leverages both transaction behavior and graph relationships to accurately detect fraudulent activities.

Key Features  
- Graph-based fraud detection using Graph Neural Networks (GNN)  
- Velocity features (transaction frequency and spending patterns)  
- Multiple model comparison:  
  - Graph Attention Network (GAT) – Proposed Model  
  - GraphSAGE (Baseline GNN)  
  - MLP Classifier  
  - XGBoost (optional)  
- Evaluation using: ROC-AUC, F1-score, Precision, Recall, MCC  
- Multi-seed training for robust results  
- Uncertainty estimation using Monte Carlo Dropout  

Models Used  

EllipticGNN (Proposed)  
- 3-layer Graph Attention Network (GAT)  
- Multi-head attention  
- Residual connections  
- Designed for node-level fraud classification  

HomoGNN (Baseline)  
- GraphSAGE model  
- Simpler aggregation without attention  

MLP Classifier  
- Fully connected neural network on tabular features  

XGBoost  
- Gradient boosting model for comparison  

Dataset  
This project uses the Elliptic Bitcoin Dataset for graph-based fraud detection.  

Download dataset from Kaggle:  
https://www.kaggle.com/datasets/ellipticco/elliptic-data-set  

Note: Dataset is not included in this repository due to size constraints.  

Tech Stack  
- Python  
- PyTorch  
- PyTorch Geometric  
- Scikit-learn  
- NumPy, Pandas  
- Matplotlib  

How to Run  

1. Clone the repository  
git clone https://github.com/ria0304/CREDIT-CARD-DETECTION.git  
cd CREDIT-CARD-DETECTION  

2. Install dependencies  
pip install -r requirements.txt  

3. Run the project  
python main.py  

Results  
- Achieves strong performance using graph-based learning  
- Outperforms traditional ML models in fraud detection  
- Evaluated across multiple random seeds for stability  

Goal  
To build a robust and scalable fraud detection system by combining transaction-level features and relationship-based graph features.  

Future Improvements  
- Real-time fraud detection system  
- API deployment (Flask / FastAPI)  
- Integration with live transaction streams  
- Model optimization for large-scale data  


If you like this project  
Give it a star on GitHub
