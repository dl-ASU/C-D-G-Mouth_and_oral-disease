import torch
import torch.nn.functional as F

def contrastive_loss(act, pos, neg):
    sim_pos = torch.exp(torch.sum(act * pos, dim=1, keepdim=True))
    sim_neg = torch.exp(torch.sum(act.unsqueeze(1) * neg, dim=2))
    loss = -torch.log(sim_pos / (sim_pos + torch.sum(sim_neg, dim=1, keepdim=True)))
    return torch.mean(loss.squeeze())

# The following taken from https://github.com/AnnaManasyan/VICReg
# variance loss
def std_loss(z_a, z_b):
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - std_z_a)) + torch.mean(F.relu(1 - std_z_b))
    return std_loss


#function taken from https://github.com/facebookresearch/barlowtwins/tree/a655214c76c97d0150277b85d16e69328ea52fd9
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# covariance loss
def cov_loss(z_a, z_b):
    N = z_a.shape[0]
    D = z_a.shape[1]
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)
    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D
    return cov_loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Compute the distance between the anchor and positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    # Compute the distance between the anchor and negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    # Compute the triplet loss
    basic_loss = pos_dist - neg_dist + margin
    loss = tf.maximum(basic_loss, 0.0)
    # Return the mean triplet loss over the batch
    return tf.reduce_mean(loss)

def npair_loss(anchor, positive, negatives, margin=0.1):
    # Compute pairwise distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negatives), axis=-1) 
    # Calculate the loss
    loss = tf.reduce_mean(tf.nn.relu(pos_dist - neg_dist + margin))
    return loss
def online_contrastive_loss(anchor, positive, margin=1.0, max_negatives=10):
    # Compute pairwise distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    # Select hard negative examples
    neg_dist = tf.reduce_sum(tf.square(tf.expand_dims(anchor, axis=1) - tf.expand_dims(anchor, axis=0)), axis=-1)
    neg_dist = tf.boolean_mask(neg_dist, neg_dist > pos_dist + margin)
    neg_dist = tf.nn.top_k(neg_dist, k=tf.minimum(max_negatives, tf.shape(neg_dist)[0])).values
    # Calculate the loss
    loss = tf.reduce_mean(tf.nn.relu(pos_dist - neg_dist + margin))
    return loss


