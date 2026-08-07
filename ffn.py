import gensim.downloader as api
import numpy as np
import math



SENTENCE = "My cat is on the mat"

model = api.load("glove-wiki-gigaword-100")

words = SENTENCE.lower().split()


vectors = []


for word in words:
    if word in model:
        vectors.append(model[word])

d = len(vectors[0])
pe = np.zeros((len(vectors),d))
final_embedding = []
for i in range(len(vectors)):
    for j in range(d):
        pe[i][j] = math.sin(i / math.pow(10000, j/d))

final_embedding = []
for i in range(len(vectors)):
    final_embedding.append(vectors[i] + pe[i])


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

w_q_list_init = [np.random.randn(d, 64) for _ in range(3)]
w_k_list_init = [np.random.randn(d, 64) for _ in range(3)]
w_v_list_init = [np.random.randn(d, 64) for _ in range(3)]
w_proj_list_init = [np.random.randn(64, 100) for _ in range(3)]


expand_ffn_w = np.random.randn(300, 1200)   
contract_ffn_w = np.random.randn(1200, 300)

projected_matrix = np.random.randn(300, 6)



def feedforward():     

    w_q_list = []
    w_k_list = []
    w_v_list = []
    w_proj_list = []

    for i in range(3):

        w_q = w_q_list_init[i]
        w_k = w_k_list_init[i]
        w_v = w_v_list_init[i]
        w_proj = w_proj_list_init[i]

        


        Q = X @ w_q
        K = X @ w_k
        V = X @ w_v

        SCORE = Q @ np.transpose(K)


        SCALED_SCORES = np.array(SCORE / math.sqrt(64))

        ACUTAL_SCALED_SCORES = SCALED_SCORES + np.array(mask)
        attention_weight = softmax(ACUTAL_SCALED_SCORES)
        OUTPUT = attention_weight @ V

        OUTPUT_projected = OUTPUT @ w_proj  

        OUTPUTS.append(X + OUTPUT_projected)  

        w_q_list.append(w_q)
        w_k_list.append(w_k)
        w_v_list.append(w_v)
        w_proj_list.append(w_proj)

    concatenated_attention = np.concatenate(OUTPUTS, axis=1)  # Shape: (6, 300)


    mean1 = np.mean(concatenated_attention, axis=1, keepdims=True)  
    std1 = np.std(concatenated_attention, axis=1, keepdims=True)    

    layer_norm_output = (concatenated_attention - mean1) / (std1 + 1e-5)  

    layer_norm_output_expanded = layer_norm_output @ expand_ffn_w  


    def relu(x):
        return np.maximum(0, x) 

    apply_relu = relu(layer_norm_output_expanded)  

    layer_norm_output_contracted = apply_relu @ contract_ffn_w  #contracted = 1200*300...to make 6 * 300...expanded is 300*1200

    final_ffn = layer_norm_output_contracted + layer_norm_output #


    mean2 = np.mean(final_ffn, axis=1, keepdims=True)
    std2 = np.std(final_ffn, axis=1, keepdims=True)
    final_ffn_normalized = (final_ffn - mean2) / (std2 + 1e-5) 


    vocab_words = ["my", "cat", "is", "on", "the", "mat"]

    index_of_words = {}

    for i in range (len(vocab_words)):
        index_of_words[vocab_words[i]] = i
    vocab_size = len(vocab_words)


    project_ffn = final_ffn_normalized @ projected_matrix ##logits,outputneuron

    logit_to_probab = softmax(project_ffn)
    last_position_probs = logit_to_probab[-1, :]


    target_word = "mat"
    target_idx = index_of_words[target_word]

    # Compute loss
    loss = -np.log(last_position_probs[target_idx])



    correct = np.zeros(vocab_size) 
    correct[target_idx] = 1        

    # Error signal
    error = last_position_probs - correct  ##dL/d(project_ffn)

    return (w_q_list, w_k_list, w_v_list, w_proj_list,
            expand_ffn_w, contract_ffn_w, projected_matrix,
            concatenated_attention, mean1, std1, layer_norm_output,
            layer_norm_output_expanded, apply_relu, final_ffn, mean2, std2,
            final_ffn_normalized, project_ffn, logit_to_probab,
            target_idx, correct, error, loss) 




def backprop(w_q_list, w_k_list,w_vl_list,w_proj_list,expand_ffn_w,contract_ffn_w,projected_matrix,concatenated_attention,mean1,std1,layer_norm_output,layer_norm_output_expanded,
             apply_relu, final_ffn, mean2, std2,
            final_ffn_normalized, project_ffn, logit_to_probab,
            target_idx, correct, error, loss):
    
    correct_error_dimension = np.zeros_like(project_ffn)
    correct_error_dimension[-1,:] = error
    d_projected_matrix = final_ffn_normalized.T @ correct_error_dimension

    d_final_ffn_normalized = correct_error_dimension @ projected_matrix.T


    n = final_ffn.shape[0]      # number of rows (6)
    N = final_ffn.shape[1]      # number of columns per row (300)

    d_final_ffn = np.zeros_like(final_ffn)

    for i in range(n):
        mean_i = mean2[i][0]
        std_i = std2[i][0]
        eps = 1e-5

        for j in range(N):
            total = 0
            for k in range(N):
                y = final_ffn[i][k]
                z = mean_i
                w = std_i
                a = eps

                dy_dt = 1 if k == j else 0   
                dz_dt = 1 / N                
                dw_dt = (final_ffn[i][j] - mean_i) / (N * std_i)  

                # quotient rule: x = (y-z)/(w+a)
                dx_dt = ((w + a) * (dy_dt - dz_dt) - (y - z) * dw_dt) / (w + a)**2

                g_k = d_final_ffn_normalized[i][k]   
                total += g_k * dx_dt

            d_final_ffn[i][j] = total


    d_contract_ffn = apply_relu.T @ d_final_ffn

    d_apply_relu = d_final_ffn @ contract_ffn_w.T

    d_layer_norm_output_expanded = d_apply_relu * (layer_norm_output_expanded > 0)

    d_expand_ffn_w = layer_norm_output.T @ d_layer_norm_output_expanded

