import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


class Normalization(nn.Module):
    """
    1D batch normalization for [*, C] input
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.norm = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        size = x.size()
        return self.norm(x.view(-1, size[-1])).view(*size)


class Attention(nn.Module):
    """
    Multi-head attention
    Input:
      q: [batch_size, n_node, hidden_dim]
      k, v: q if None
    Output:
      att: [n_node, hidden_dim]
    """
    def __init__(self,
                 q_hidden_dim,
                 k_dim,
                 v_dim,
                 n_head,
                 k_hidden_dim=None,
                 v_hidden_dim=None):
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim if k_hidden_dim else q_hidden_dim
        self.v_hidden_dim = v_hidden_dim if v_hidden_dim else q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head,
                                     self.v_hidden_dim,
                                     bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        if v is None:
            v = q

        bsz, n_node, hidden_dim = q.size()

        qs = torch.stack(torch.chunk(self.proj_q(q), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        ks = torch.stack(torch.chunk(self.proj_k(k), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, k_dim]
        vs = torch.stack(torch.chunk(self.proj_v(v), self.n_head, dim=-1),
                         dim=1)  # [batch_size, n_head, n_node, v_dim]

        normalizer = self.k_dim**0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask, float('-inf'))
        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, n_node,
                                          self.v_dim * self.n_head)
        att = self.proj_output(att)
        return att


class TSPEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 ff_dim,
                 n_layer,
                 k_dim=16,
                 v_dim=16,
                 n_head=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.n_layer = n_layer
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attentions = nn.ModuleList([
            Attention(hidden_dim, k_dim, v_dim, n_head) for _ in range(n_layer)
        ])
        self.ff = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(hidden_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, hidden_dim)
            ]) for _ in range(n_layer)
        ])
        self.bn1 = nn.ModuleList(
            [Normalization(hidden_dim) for _ in range(n_layer)])
        self.bn2 = nn.ModuleList(
            [Normalization(hidden_dim) for _ in range(n_layer)])

    def forward(self, x):
        h = self.embedding(x)
        for i in range(self.n_layer):
            h = self.bn1[i](h + self.attentions[i](h))
            h = self.bn2[i](h + self.ff[i](h))
        return h


class TSPDecoder(nn.Module):
    def __init__(self, hidden_dim, k_dim, v_dim, n_head):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.v_l = nn.Parameter(
            torch.FloatTensor(size=[1, 1, hidden_dim]).uniform_())
        self.v_f = nn.Parameter(
            torch.FloatTensor(size=[1, 1, hidden_dim]).uniform_())

        self.attention = Attention(hidden_dim * 3,
                                   k_dim,
                                   v_dim,
                                   n_head,
                                   k_hidden_dim=hidden_dim,
                                   v_hidden_dim=hidden_dim)

        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def decode_all(self, x, C=10, rollout=False):
        """
        x: [batch_size, n_node, hidden_dim] node embeddings of TSP graph
        """
        bsz, n_node, hidden_dim = x.size()

        h_avg = x.mean(dim=-2, keepdim=True)
        first = self.v_f.repeat(bsz, 1, 1)
        last = self.v_l.repeat(bsz, 1, 1)

        k = self.proj_k(x)

        normalizer = self.hidden_dim**0.5
        visited_idx = []
        mask = torch.zeros([bsz, n_node], device=x.device).bool()
        log_prob = 0

        for i in range(n_node):
            h_c = torch.cat([h_avg, last, first], -1)
            q = self.attention(h_c, x, x, mask=mask)

            u = torch.tanh(q.bmm(k.transpose(-2, -1)) / normalizer) * C
            u = u.masked_fill(mask.unsqueeze(1), float('-inf'))

            if rollout:
                visit_idx = u.max(-1)[1]
            else:
                m = Categorical(logits=u)
                visit_idx = m.sample()
                log_prob += m.log_prob(visit_idx)

            visited_idx += [visit_idx]
            mask = mask.scatter(1, visit_idx, True)

            visit_idx = visit_idx.unsqueeze(-1).repeat(1, 1, hidden_dim)
            last = torch.gather(x, 1, visit_idx)
            if len(visited_idx) == 1:
                first = last

        visited_idx = torch.cat(visited_idx, -1)
        return visited_idx, log_prob

    def initialize(self, x):
        """
        x: [batch_size, n_node, hidden_dim] node embeddings of TSP graph
        """
        bsz, n_node, hidden_dim = x.size()

        self.h_avg = x.mean(dim=-2, keepdim=True)
        self.first = self.v_f.repeat(bsz, 1, 1)
        self.last = self.v_l.repeat(bsz, 1, 1)
        self.k = self.proj_k(x)
        self.normalizer = self.hidden_dim ** 0.5
        self.x = x

    def forward(self, first, last, mask, C=10):
        """
        decoding one step given the state of the problem
        """
        h_c = torch.cat([self.h_avg, last, first], -1)
        q = self.attention(h_c, self.x, self.x, mask=mask)

        u = torch.tanh(q.bmm(self.k.transpose(-2, -1)) / self.normalizer) * C
        u = u.masked_fill(mask.unsqueeze(1), float('-inf'))
        probs = torch.softmax(u, dim=-1).squeeze()
        return probs


class TSPValue(nn.Module):
    def __init__(self, hidden_dim, k_dim, v_dim, n_head):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.v_l = nn.Parameter(
            torch.FloatTensor(size=[1, 1, hidden_dim]).uniform_())
        self.v_f = nn.Parameter(
            torch.FloatTensor(size=[1, 1, hidden_dim]).uniform_())

        self.attention = Attention(hidden_dim * 3,
                                   k_dim,
                                   v_dim,
                                   n_head,
                                   k_hidden_dim=hidden_dim,
                                   v_hidden_dim=hidden_dim)

        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*2, bias=False)
        self.fc2 = nn.Linear(hidden_dim*2, 1, bias=False)

    def initialize(self, x):
        """
        x: [batch_size, n_node, hidden_dim] node embeddings of TSP graph
        """
        bsz, n_node, hidden_dim = x.size()

        self.first = self.v_f.repeat(bsz, 1, 1)
        self.last = self.v_l.repeat(bsz, 1, 1)
        self.k = self.proj_k(x)
        self.normalizer = self.hidden_dim ** 0.5
        self.x = x


    def forward(self, first, last, mask):
        """
        value of a given state
        """
        non_vis = torch.logical_not(mask.unsqueeze(1))
        non_vis = torch.transpose(non_vis, 1, 2)
        non_vis = non_vis.expand(self.x.shape[0], self.x.shape[1], self.x.shape[2])
        remaining = self.x*non_vis
        h_avg = remaining.mean(dim=-2, keepdim=True)
        h_c = torch.cat([h_avg, last, first], -1)
        q = self.attention(h_c, self.x, self.x, mask=mask)
        h = self.fc1(q)
        h = F.relu(h)
        val = self.fc2(h)
        return val.squeeze()


class TSPSolver(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=128,
                 ff_dim=512,
                 n_layer=3,
                 k_dim=16,
                 v_dim=16,
                 n_head=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = TSPEncoder(input_dim, hidden_dim, ff_dim, n_layer,
                                  k_dim, v_dim, n_head)
        self.decoder = TSPDecoder(hidden_dim, k_dim, v_dim, n_head)
        self.value = TSPValue(hidden_dim, k_dim, v_dim, n_head)

    def encode(self, x):

        self.embeddings = self.encoder(x)
        self.bs = self.embeddings.shape[0]
        self.decoder.initialize(self.embeddings)
        self.value.initialize(self.embeddings)

    def decode(self, x, rollout=True):
        x = self.encoder(x)
        return self.decoder.decode_all(x, rollout=rollout)

    def forward(self, first, last, mask):
        first = first.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        first = torch.gather(self.embeddings, 1, first)
        last = last.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        last = torch.gather(self.embeddings, 1, last)
        pi = self.decoder(first, last, mask)
        val = self.value(first, last, mask)
        return pi, val

    def save(self, fn):
        torch.save(self.state_dict(), fn)
        self.name = fn

    def load(self, fn):
        try:
            if fn is None:
                self.name = "initial"
            else:
                self.load_state_dict(torch.load(fn), strict=False)
                self.name = fn
        except FileNotFoundError as e:
            print(f"Param file not found {e}")


x = np.random.random((2, 4, 2))
x = torch.Tensor(x)
solver = TSPSolver()
#print(solver.decode(x, rollout=False))