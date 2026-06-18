from word2vec import final_embedding, d,vectors
import math
import numpy as np
X = final_embedding
mask = []
OUTPUTS = []


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1,keepdims=True)) 
    return exp_x / np.sum(exp_x,axis=-1,keepdims=True)
for i in range(len(vectors)):
    row = []                    
    for j in range(len(vectors)):
        if i >= j:
            row.append(0)       
        else:
            row.append(-math.inf)
    mask.append(row)        


for i in range(3):

    w_q = np.random.randn(d, 64)
    w_k = np.random.randn(d, 64)
    w_v = np.random.randn(d, 64)


    Q = X @ w_q
    K = X @ w_k
    V = X @ w_v

    SCORE = Q @ np.transpose(K)


    SCALED_SCORES = np.array(SCORE / math.sqrt(64))

    ACUTAL_SCALED_SCORES = SCALED_SCORES + np.array(mask)
    attention_weight = softmax(ACUTAL_SCALED_SCORES)
    OUTPUT = attention_weight @ V

    w_proj = np.random.randn(64, 100)  
    OUTPUT_projected = OUTPUT @ w_proj  

    OUTPUTS.append(X + OUTPUT_projected)  

for output in OUTPUTS:
    concatenated_attention = np.concatenate(output, axis=1)



