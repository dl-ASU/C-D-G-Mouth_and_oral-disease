import torch
import torch.nn as nn


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.Tensor(mask))

    def forward(self, input):
        return nn.functional.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.fc1 = MaskedLinear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([MaskedLinear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)])
        self.fc_out = MaskedLinear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.m = [torch.arange(input_dim) if i == 0 else torch.randint(input_dim, (hidden_dim,)) for i in range(num_hidden_layers + 2)]

        self.create_masks()

    def create_masks(self):

        masks = []

        for i in range(self.num_hidden_layers + 1):
          mask = (self.m[i][None, :] < self.m[i+1][:, None]).float()
          masks.append(mask)

        self.fc1.set_mask(masks[0])
        for i, hidden_layer in enumerate(self.hidden_layers):
            hidden_layer.set_mask(masks[i + 1])

        self.fc_out.set_mask((self.m[-1][None, :] < self.m[0][:, None]).float())

    def forward(self, x):
        x = self.relu(self.fc1(x))
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        return self.sigmoid(self.fc_out(x))

    def log_likelihood(self, x):
        logits = self.forward(x)
        return torch.sum(x * torch.log(logits + 1e-8) + (1 - x) * torch.log(1 - logits + 1e-8), dim=1)

    def nll(self, x):
      return -self.log_likelihood(x).mean()

    def sample(self, n_samples):
        samples = torch.zeros(n_samples, self.input_dim)
        for i in range(self.input_dim):
          logits = self.forward(samples)
          samples[:, i] = (logits[:, i] > 0.5).float()
        return samples