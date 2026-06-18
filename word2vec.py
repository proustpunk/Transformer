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

print(final_embedding)
print(len(final_embedding))