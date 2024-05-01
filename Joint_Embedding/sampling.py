def sample_contrastive_pairs(train_x, train_y, N = 4):
    positive_pairs = []
    negative_pairs = []

    for i in range(len(train_x)):
        counter = 0
        negative_pair = []

        for j in range(len(train_x)):
            if (i != j) and (train_y[i] == train_y[j]):
                positive_pairs.append(train_x[j])
                break

        for j in range(len(train_x)):
            if counter == N:
                break
            if train_y[i] > 0:
                if 0 == train_y[j]:
                    negative_pair.append(train_x[j])
                    counter = counter + 1
            else:
                if train_y[i] != train_y[j]:
                    negative_pair.append(train_x[j])
                    counter = counter + 1

        negative_pair = torch.stack(negative_pair, dim = 0)
        negative_pairs.append(negative_pair)

    positive_pair = torch.stack(positive_pairs, dim = 0)
    negative_pairs = torch.stack(negative_pairs, dim = 0)
    return positive_pair, negative_pairs
