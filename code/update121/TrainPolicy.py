import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from copy import copy, deepcopy
from scipy import stats
from time import time
from Attention import TSPSolver as Attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TSP2D:

    @classmethod
    def n_attr(cls):
        return 10, 1

    def __init__(self, pos=20):
        # pos : np array of shape (n,2)
        if isinstance(pos, int):
            n_node = pos
            pos = np.random.rand(n_node, 2).astype(np.float32)
        self.n_node = pos.shape[0]
        self.pos = pos
        self.X = torch.from_numpy(pos)
        self.D = squareform(pdist(pos, metric='euclidean')).astype(np.float32)

    def eval_tour(self, tour, closed=True):
        n = len(tour)
        if closed:
            tour_len = self.D[tour[n - 1], tour[0]]
        else:
            tour_len = 0.0
        for i in range(1, n):
            tour_len += self.D[tour[i - 1], tour[i]]
        return tour_len

    def get_pos_tensor(self, v):
        x = torch.tensor(self.pos[v, :])
        XT = x.expand(self.n_node, -1)
        return XT

    def set_tour(self, tour):
        self.tour = tour
        self.tour_len = self.eval_tour(self.tour)



def read_tsp_file(fn, n=0):
    x = np.loadtxt(fn, delimiter=',').astype(np.float32)
    if n <= 0:
        n = x.shape[0] // 3
    tsp_list = []
    for i in range(n):
        pos = x[3 * i:3 * i + 2, :].T
        tsp = TSP2D(pos)
        tsp.set_tour(list(x[3 * i + 2, :].astype(int)))
        tsp_list.append(tsp)
    return tsp_list


def Cost(graphs, permutations):
    """
    Author: wouterkool
    Copied from https://github.com/wouterkool/attention-learn-to-route
    """
    # Check that tours are valid, i.e. contain 0 to n -1
    assert (torch.arange(
        permutations.size(1), out=permutations.data.new()).view(
        1, -1).expand_as(permutations) == permutations.data.sort(1)[0]
            ).all(), "Invalid tour"

    # Gather dataset in order of tour
    d = graphs.gather(1, permutations.unsqueeze(-1).expand_as(graphs))

    # Length is distance (L2-norm of difference) from each next location from
    # its prev and of last from first
    return ((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) +
            (d[:, 0] - d[:, -1]).norm(p=2, dim=1)).view(-1, 1)

def Cost_base(graphs, permutations):
    """
    Author: wouterkool
    Copied from https://github.com/wouterkool/attention-learn-to-route
    """
    # Check that tours are valid, i.e. contain 0 to n -1
    assert (torch.arange(
        permutations.size(1), out=permutations.data.new()).view(
        1, -1).expand_as(permutations) == permutations.data.sort(1)[0]
            ).all(), "Invalid tour"

    # Gather dataset in order of tour
    d = graphs.gather(1, permutations.unsqueeze(-1).expand_as(graphs))

    # Length is distance (L2-norm of difference) from each next location from
    # its prev and of last from first
    costs = ((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) +
            (d[:, 0] - d[:, -1]).norm(p=2, dim=1)).view(8, -1)
    mins = torch.mean(costs, dim=-2)
    return mins.repeat(8).view(-1, 1)

def train_reinforce_step(problem, net, target, opt, epoch):
    opt.zero_grad()
    tour, log_prob = net.decode(problem, rollout=False)
    with torch.no_grad():
        cost = Cost(problem, tour)
        # rollout baseline
        #baseline_tour, _ = target.decode(problem, rollout=True)
        baseline_cost = Cost_base(problem, tour)
        advantage = cost - baseline_cost
    loss = advantage * log_prob
    loss = loss.mean()
    loss.backward()  # Derive gradients.
    opt.step()
    problem = torch.rand(1000,50, 2).to(device)
    with torch.no_grad():
      net_tour, _ = net.decode(problem, rollout=True)
      baseline_tour, _ = target.decode(problem, rollout=True)
      net_greedy = Cost(problem, net_tour)
      baseline_cost = Cost(problem, baseline_tour)
      improve = (baseline_cost - net_greedy).mean() >= 0
      _, p_value = stats.ttest_rel(baseline_cost.flatten().cpu().numpy(), net_greedy.flatten().cpu().numpy())
      if improve and p_value <= 0.05:
        target.load_state_dict(net.state_dict())
        print('target improved')
    return loss.detach()


def train_reinforce(net, target, epoch, batch_size):
    opt = torch.optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-6)
    for i in range(epoch):
        problem = torch.rand((int(50*batch_size/8), 2)).to(device)
        p = torch.zeros_like(problem)
        p[:, 0] = problem[:, 1]
        p[:, 1] = problem[:, 0]
        ref1 = problem.clone()
        ref1[:, 0] = 1 - problem[:, 0]
        ref2 = problem.clone()
        ref2[:, 1] = 1 - problem[:, 1]
        ref3 = 1 - problem
        ref4 = p.clone()
        ref4[:, 0] = 1 - p[:, 0]
        ref5 = p.clone()
        ref5[:, 1] = 1 - p[:, 1]
        ref6 = 1 - p
        problem = torch.cat([problem, p, ref1, ref2, ref3, ref4, ref5, ref6])
        loss = train_reinforce_step(problem.view(batch_size, 50, 2), net, target, opt, i)
        print(f"Loss is {loss} for Epoch = {i} in {epoch} epoch")

def attention_eval(tsp, pi):
    with torch.no_grad():
        tour = pi.decode(tsp.X.unsqueeze(0).to(device), rollout=True)
    return tour[0].squeeze().cpu().numpy()


def compare_policy(att=None, TSPs=30):
    # np.random.seed(1)
    if isinstance(TSPs, int):
        TSPs = [TSP2D(50) for i in range(TSPs)]
    elif isinstance(TSPs, str):
        TSPs = read_tsp_file(TSPs)
    n_tsp = len(TSPs)
    X = np.zeros((n_tsp))
    for i in range(n_tsp):
        tsp = TSPs[i]
        temp = tsp.eval_tour(tsp.tour)
        X[i] = (tsp.eval_tour(attention_eval(tsp, att)) - temp) / temp
    m = X.mean()
    s = X.std()
    print("mean(D) = ", m)
    print("std(D)  = ", s)
    return X

TSPs = read_tsp_file('tsp50.csv')

net = Attention()
net.load("param/AT_Net_AttentionReinfRemPol.pt")
net.to(device)
attention = Attention()
attention.load("param/AT_Net_AttentionReinfRemPol.pt")
attention.to(device)

for i in range(100):
    train_reinforce(net, attention, 1000, 512)
    attention.save("param/AT_Net_AttentionReinfRemPol.pt")
    print('ith episode')
    print(i)

compare_policy(att=attention, TSPs=TSPs)