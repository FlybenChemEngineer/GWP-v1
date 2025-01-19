import torch
import json
import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from dgllife.utils import (AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer)
from GNN_Model.Get_graphs import batch_smiles_to_graph
from GNN_Model.Initialize_model import MPNN_initialize_model

# 其他代码...
atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='atom_feat')
bond_featurizer = AttentiveFPBondFeaturizer(bond_data_field='bond_feat', self_loop=True)
atom_feat_size, bond_feat_size = atom_featurizer.feat_size(), bond_featurizer.feat_size()

# Check if CUDA (GPU support) is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))  # Assumes only一台GPU时，必要时可以修改

# 获取当前脚本目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义模型参数文件夹路径
model_parameters_dir = os.path.join(current_dir, 'model_parameters')

# 加载最优超参数的 JSON 文件路径
json_file_path = os.path.join(model_parameters_dir, 'gwp20_best_MPNN_hyperparameters_trial.json')

# 加载预训练模型权重文件路径
model_weights_path = os.path.join(model_parameters_dir, 'gwp20_best_MPNN_model_trial.pth')

# 加载最佳超参数的 JSON 文件
try:
    with open(json_file_path, 'r') as f:
        best_hyperparameters = json.load(f)
    print("Best hyperparameters loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {json_file_path} was not found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file {json_file_path}.")

best_model = MPNN_initialize_model(atom_feat_size=atom_feat_size,
                              bond_feat_size=bond_feat_size,
                              node_out_feats=best_hyperparameters['node_out_feats'],
                              edge_hidden_feats=best_hyperparameters['edge_hidden_feats'],
                              num_step_message_passing=best_hyperparameters['num_step_message_passing'],
                              num_step_set2set=best_hyperparameters['num_step_set2set'],
                              num_layer_set2set=best_hyperparameters['num_layer_set2set']
                              )

# Load the pre-trained model weights
best_model.load_state_dict(torch.load(model_weights_path))
best_model.to(device)
best_model.eval()

def predict_smiles(smiles, atom_bond_type='AttentiveFP'):
    """
    Predict the value for a given SMILES string.
    
    Args:
        smiles (str): The SMILES string to predict.
        atom_bond_type (str): The type of atom and bond features to use.
    
    Returns:
        float: The predicted value.
    """
    # Convert the SMILES string to graph
    bg_list, atom_feat_size, bond_feat_size = batch_smiles_to_graph([smiles], atom_bond_type)
    
    # 如果 bg_list 是一个列表，取第一个图
    bg = bg_list[0].to(device)
    
    # Extract node features
    features_node = bg.ndata['atom_feat'].to(device).float()
    
    # Make prediction based on the presence of edge features
    if bond_feat_size is not None:
        features_edge = bg.edata['bond_feat'].to(device).float()
        # Make prediction with edge features
        prediction = best_model(bg, features_node, features_edge).squeeze().float()
    else:
        # Make prediction without edge features
        prediction = best_model(bg, features_node).squeeze().float()

    predicted_value_log10 = 10 ** prediction.item()
    # Convert prediction to a scalar value and return
    return predicted_value_log10

#'''
# Example usage: Predict a SMILES string
smiles_input = "O=S(=O)(F)F"  # Replace with your SMILES input
predicted_value = predict_smiles(smiles_input)
#original_value = 10 ** predicted_value

print(f"Predicted value for SMILES (GWP20)'{smiles_input}': {predicted_value}")
#print(f"Original value (restored from log10): {original_value}")
'''
# 读取CSV文件
input_file = 'S03_cases.csv'  # 替换为你的文件路径
df = pd.read_csv(input_file)

# 确保有一列是 'smiles'
if 'smiles' not in df.columns:
    raise ValueError("输入的CSV文件缺少'smiles'列")

# 批量计算预测值
df['predicted_gwp20'] = df['smiles'].apply(predict_smiles)

# 保存到新的CSV文件
output_file = 'gwp20_smiles_predictions.csv'
df.to_csv(output_file, index=False)

print(f"预测结果已保存到 {output_file}")
'''