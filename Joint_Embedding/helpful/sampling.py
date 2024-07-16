import torch
import random

def sample_contrastive_pairs_SL(train_x, train_y, N=4):
    positive_pairs = []
    negative_pairs = []

    indices = list(range(len(train_x)))
    
    for i in range(len(train_x)):
        random.shuffle(indices)
        positive_found = False
        negative_pair = []

        for j in indices:
            if (i != j) and (train_y[i] == train_y[j]):
                positive_pairs.append(train_x[j])
                positive_found = True
                break

        if not positive_found:
            positive_pairs.append(train_x[i])

        counter = 0
        for j in indices:
            if counter == N:
                break
            if train_y[i] != train_y[j]:
                negative_pair.append(train_x[j])
                counter += 1

        if counter < N:
            x = negative_pair[-1]
            negative_pair.extend([x] * (N - counter))

        negative_pair = torch.stack(negative_pair, dim=0)
        negative_pairs.append(negative_pair)

    positive_pair = torch.stack(positive_pairs, dim=0)
    negative_pairs = torch.stack(negative_pairs, dim=0)
    return positive_pair, negative_pairs