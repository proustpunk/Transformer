from Relu import final_ffn_normalized
from selfattention import softmax
import numpy as np

vocab_words = ["My", "cat", "is", "on", "the", "mat"]

index_of_words = {}

for i in range (len(vocab_words)):
    index_of_words[vocab_words[i]] = i
vocab_size = len(vocab_words)

projected_matrix = np.random.randn(300, vocab_size)

project_ffn = final_ffn_normalized @ projected_matrix ##logits

logit_to_probab = softmax(project_ffn)
last_position_probs = logit_to_probab[-1, :]


target_word = "mat"
target_idx = index_of_words[target_word]

# Compute loss
loss = -np.log(last_position_probs[target_idx])

print(loss)

print(loss)





