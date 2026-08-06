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

d = len(vectors[0])  # d = 100
pe = np.zeros((len(vectors), d))  # (6, 100)
final_embedding = []
for i in range(len(vectors)):
    for j in range(d):
        pe[i][j] = math.sin(i / math.pow(10000, j/d))

final_embedding = []
for i in range(len(vectors)):
    final_embedding.append(vectors[i] + pe[i])

X = final_embedding  # (6, 100)
mask = []
OUTPUTS = []

def softmax(x):  # x: (6, 6) or (6, 6) or (6, 6)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # (6, 6)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # (6, 6)

for i in range(len(vectors)):
    row = []
    for j in range(len(vectors)):
        if i >= j:
            row.append(0)
        else:
            row.append(-math.inf)
    mask.append(row)  # (6, 6)

w_q_list_init = [np.random.randn(d, 64) for _ in range(3)]  # (100, 64) each
w_k_list_init = [np.random.randn(d, 64) for _ in range(3)]  # (100, 64) each
w_v_list_init = [np.random.randn(d, 64) for _ in range(3)]  # (100, 64) each
w_proj_list_init = [np.random.randn(64, 100) for _ in range(3)]  # (64, 100) each

expand_ffn_w = np.random.randn(300, 1200)  # (300, 1200)
contract_ffn_w = np.random.randn(1200, 300)  # (1200, 300)
projected_matrix = np.random.randn(300, 6)  # (300, 6)

def feedforward():
    w_q_list = []
    w_k_list = []
    w_v_list = []
    w_proj_list = []

    Q_list = []
    K_list = []
    V_list = []
    attention_weight_list = []
    OUTPUT_list = []

    for i in range(3):
        w_q = w_q_list_init[i]  # (100, 64)
        w_k = w_k_list_init[i]  # (100, 64)
        w_v = w_v_list_init[i]  # (100, 64)
        w_proj = w_proj_list_init[i]  # (64, 100)

        Q = X @ w_q  # (6, 64)
        K = X @ w_k  # (6, 64)
        V = X @ w_v  # (6, 64)

        SCORE = Q @ np.transpose(K)  # (6, 6)
        SCALED_SCORES = np.array(SCORE / math.sqrt(64))  # (6, 6)
        ACUTAL_SCALED_SCORES = SCALED_SCORES + np.array(mask)  # (6, 6)
        attention_weight = softmax(ACUTAL_SCALED_SCORES)  # (6, 6)
        OUTPUT = attention_weight @ V  # (6, 64)
        OUTPUT_projected = OUTPUT @ w_proj  # (6, 100)
        OUTPUTS.append(X + OUTPUT_projected)  # (6, 100)

        w_q_list.append(w_q)
        w_k_list.append(w_k)
        w_v_list.append(w_v)
        w_proj_list.append(w_proj)

        Q_list.append(Q)
        K_list.append(K)
        V_list.append(V)
        attention_weight_list.append(attention_weight)
        OUTPUT_list.append(OUTPUT)

    concatenated_attention = np.concatenate(OUTPUTS, axis=1)  # (6, 300)
    mean1 = np.mean(concatenated_attention, axis=1, keepdims=True)  # (6, 1)
    std1 = np.std(concatenated_attention, axis=1, keepdims=True)  # (6, 1)
    layer_norm_output = (concatenated_attention - mean1) / (std1 + 1e-5)  # (6, 300)
    layer_norm_output_expanded = layer_norm_output @ expand_ffn_w  # (6, 1200)

    def relu(x):  # x: (6, 1200)
        return np.maximum(0, x)  # (6, 1200)

    apply_relu = relu(layer_norm_output_expanded)  # (6, 1200)
    layer_norm_output_contracted = apply_relu @ contract_ffn_w  # (6, 300)
    final_ffn = layer_norm_output_contracted + layer_norm_output  # (6, 300)
    mean2 = np.mean(final_ffn, axis=1, keepdims=True)  # (6, 1)
    std2 = np.std(final_ffn, axis=1, keepdims=True)  # (6, 1)
    final_ffn_normalized = (final_ffn - mean2) / (std2 + 1e-5)  # (6, 300)

    vocab_words = ["my", "cat", "is", "on", "the", "mat"]
    index_of_words = {}
    for i in range(len(vocab_words)):
        index_of_words[vocab_words[i]] = i
    vocab_size = len(vocab_words)

    project_ffn = final_ffn_normalized @ projected_matrix  # (6, 6)
    logit_to_probab = softmax(project_ffn)  # (6, 6)
    last_position_probs = logit_to_probab[-1, :]  # (6,)

    target_word = "mat"
    target_idx = index_of_words[target_word]
    loss = -np.log(last_position_probs[target_idx])
    correct = np.zeros(vocab_size)  # (6,)
    correct[target_idx] = 1
    error = last_position_probs - correct  # (6,)

    return (w_q_list, w_k_list, w_v_list, w_proj_list,
            expand_ffn_w, contract_ffn_w, projected_matrix,
            concatenated_attention, mean1, std1, layer_norm_output,
            layer_norm_output_expanded, apply_relu, final_ffn, mean2, std2,
            final_ffn_normalized, project_ffn, logit_to_probab,
            target_idx, correct, error, loss,
            Q_list, K_list, V_list, attention_weight_list, OUTPUT_list)

def backprop(w_q_list, w_k_list, w_v_list, w_proj_list, expand_ffn_w, contract_ffn_w, projected_matrix,
             concatenated_attention, mean1, std1, layer_norm_output, layer_norm_output_expanded,
             apply_relu, final_ffn, mean2, std2, final_ffn_normalized, project_ffn, logit_to_probab,
             target_idx, correct, error, loss, Q_list, K_list, V_list, attention_weight_list, OUTPUT_list, lr=0.01):
    
    correct_error_dimension = np.zeros_like(project_ffn)  # (6, 6)
    correct_error_dimension[-1, :] = error  # (6,)
    d_projected_matrix = final_ffn_normalized.T @ correct_error_dimension  # (300, 6)
    d_final_ffn_normalized = correct_error_dimension @ projected_matrix.T  # (6, 300)

    n = final_ffn.shape[0]  # 6
    N = final_ffn.shape[1]  # 300
    d_final_ffn = np.zeros_like(final_ffn)  # (6, 300)

    for i in range(n):
        mean_i = mean2[i][0]  # scalar
        std_i = std2[i][0]  # scalar
        eps = 1e-5

        for j in range(N):
            total = 0
            for k in range(N):
                y = final_ffn[i][k]  # scalar
                z = mean_i  # scalar
                w = std_i  # scalar
                a = eps  # scalar

                dy_dt = 1 if k == j else 0  # scalar
                dz_dt = 1 / N  # scalar
                dw_dt = (final_ffn[i][j] - mean_i) / (N * std_i)  # scalar

                dx_dt = ((w + a) * (dy_dt - dz_dt) - (y - z) * dw_dt) / (w + a)**2  # scalar
                g_k = d_final_ffn_normalized[i][k]  # scalar
                total += g_k * dx_dt  # scalar

            d_final_ffn[i][j] = total  # (6, 300)

    d_layer_norm_output_contracted = d_final_ffn  # (6, 300)
    d_contract_ffn_w = apply_relu.T @ d_layer_norm_output_contracted  # (1200, 300)
    d_apply_relu = d_layer_norm_output_contracted @ contract_ffn_w.T  # (6, 1200)
    d_layer_norm_output_expanded = d_apply_relu * (layer_norm_output_expanded > 0)  # (6, 1200)
    d_expand_ffn_w = layer_norm_output.T @ d_layer_norm_output_expanded  # (300, 1200)
    d_layer_norm_output_from_expand = d_layer_norm_output_expanded @ expand_ffn_w.T  # (6, 300)
    d_layer_norm_output = d_layer_norm_output_from_expand + d_final_ffn  # (6, 300)

    n1 = concatenated_attention.shape[0]  # 6
    N1 = concatenated_attention.shape[1]  # 300
    d_concatenated_attention = np.zeros_like(concatenated_attention)  # (6, 300)

    for i in range(n1):
        mean_i = mean1[i][0]  # scalar
        std_i = std1[i][0]  # scalar
        eps = 1e-5

        for j in range(N1):
            total = 0
            for k in range(N1):
                y = concatenated_attention[i][k]  # scalar
                z = mean_i  # scalar
                w = std_i  # scalar
                a = eps  # scalar

                dy_dt = 1 if k == j else 0  # scalar
                dz_dt = 1 / N1  # scalar
                dw_dt = (concatenated_attention[i][j] - mean_i) / (N1 * std_i)  # scalar

                dx_dt = ((w + a) * (dy_dt - dz_dt) - (y - z) * dw_dt) / (w + a)**2  # scalar
                g_k = d_layer_norm_output[i][k]  # scalar
                total += g_k * dx_dt  # scalar

            d_concatenated_attention[i][j] = total  # (6, 300)

    d_outputs = np.split(d_concatenated_attention, 3, axis=1)  # 3 x (6, 100)

    d_w_q_list = []
    d_w_k_list = []
    d_w_v_list = []
    d_w_proj_list = []

    dX_total = np.zeros_like(np.array(X))  # (6, 100)

    for h in range(3):
        d_out = d_outputs[h]  # (6, 100)
        dX_total += d_out  # (6, 100)
        d_output_projected = d_out  # (6, 100)

        OUTPUT = OUTPUT_list[h]  # (6, 64)
        attention_weight = attention_weight_list[h]  # (6, 6)
        Q = Q_list[h]  # (6, 64)
        K = K_list[h]  # (6, 64)
        V = V_list[h]  # (6, 64)
        w_q = w_q_list[h]  # (100, 64)
        w_k = w_k_list[h]  # (100, 64)
        w_v = w_v_list[h]  # (100, 64)
        w_proj = w_proj_list[h]  # (64, 100)

        d_w_proj = OUTPUT.T @ d_output_projected  # (64, 100)
        d_output = d_output_projected @ w_proj.T  # (6, 64)

        d_attention_weight = d_output @ V.T  # (6, 6)
        d_v = attention_weight.T @ d_output  # (6, 64)

        d_actual_scaled_scores = np.zeros_like(attention_weight)  # (6, 6)
        for row in range(attention_weight.shape[0]):
            p = attention_weight[row]  # (6,)
            jacobian = np.zeros((6, 6))  # (6, 6)
            for i in range(6):
                for j in range(6):
                    if i == j:
                        jacobian[i][j] = p[i] * (1 - p[i])
                    else:
                        jacobian[i][j] = -p[i] * p[j]
            d_actual_scaled_scores[row] = jacobian @ d_attention_weight[row]  # (6,)

        d_scaled_scores = d_actual_scaled_scores  # (6, 6)
        d_score = d_scaled_scores / math.sqrt(64)  # (6, 6)

        d_q = d_score @ K  # (6, 64)
        d_k = d_score.T @ Q  # (6, 64)

        d_w_q = X.T @ d_q  # (100, 64)
        d_w_k = X.T @ d_k  # (100, 64)
        dX_total += d_q @ w_q.T + d_k @ w_k.T  # (6, 100)

        d_w_v = np.array(X).T @ d_v  # (100, 64)
        dX_total += d_v @ w_v.T  # (6, 100)

        d_w_q_list.append(d_w_q)
        d_w_k_list.append(d_w_k)
        d_w_v_list.append(d_w_v)
        d_w_proj_list.append(d_w_proj)

    projected_matrix -= lr * d_projected_matrix  # (300, 6)
    contract_ffn_w -= lr * d_contract_ffn_w  # (1200, 300)
    expand_ffn_w -= lr * d_expand_ffn_w  # (300, 1200)

    for h in range(3):
        w_proj_list_init[h] -= lr * d_w_proj_list[h]  # (64, 100)
        w_v_list_init[h] -= lr * d_w_v_list[h]  # (100, 64)
        w_q_list_init[h] -= lr * d_w_q_list[h]  # (100, 64)
        w_k_list_init[h] -= lr * d_w_k_list[h]  # (100, 64)

    return loss