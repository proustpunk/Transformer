from .selfattention import layer_norm_output
import numpy as np

expand_ffn_w = np.random.randn(300, 1200)   
contract_ffn_w = np.random.randn(1200, 300) 


layer_norm_output_expanded = layer_norm_output @ expand_ffn_w  


def relu(x):
    return np.maximum(0, x) 

apply_relu = relu(layer_norm_output_expanded)  

layer_norm_output_contracted = apply_relu @ contract_ffn_w  

final_ffn = layer_norm_output_contracted + layer_norm_output 


mean = np.mean(final_ffn, axis=1, keepdims=True)
std = np.std(final_ffn, axis=1, keepdims=True)
final_ffn_normalized = (final_ffn - mean) / (std + 1e-5) 