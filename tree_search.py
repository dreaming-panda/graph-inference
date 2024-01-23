import torch
torch.set_printoptions(profile="full")
p = torch.tensor([0, 7.6527e-01, 9.5442e-02, 4.0663e-02, 2.0413e-02, 1.2295e-02, 9.5130e-03,
        6.3538e-03, 5.2739e-03, 4.1210e-03, 3.2638e-03, 3.0301e-03, 2.5622e-03,
        1.8990e-03, 2.0484e-03, 1.3920e-03, 1.5674e-03, 1.3609e-03, 1.5058e-03,
        1.1261e-03, 8.7023e-04, 6.1686e-04, 5.3829e-04, 8.0728e-04, 7.8044e-04,
        8.3824e-04, 5.6952e-04, 5.0139e-04, 2.8778e-04, 2.3417e-04, 7.0067e-04,
        5.2266e-04, 3.2044e-04])

max_branch = p.shape[0] - 1

max_depth = 10

max_budget = 128

T = torch.zeros((max_budget + 1, max_depth + 1, max_branch + 1)).fill_(-torch.inf)
T_max = torch.zeros((max_budget + 1, max_depth + 1))

for l in range(1, max_depth + 1):
    for b in range(0, max_branch + 1):
        if b == 0:
            T[1][l][b] = 1.0


for m in range(2, max_budget+1):
    for l in range(2, max_depth + 1):
        T[m][l][1] = 1 + p[1] * T[m-1][l-1].max()
        for b in range(2, max_branch + 1):
            max_value = -torch.inf
            for y in range(1, m):
                new_value = T[y][l][b-1] + p[b] * T[m-y][l-1].max()
                max_value = max(max_value, new_value)
            T[m][l][b] = max_value
    

results = T.max(dim=2).values

draft_inference_time = 0.0

target_verify_time = []

valid_budget = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128]

for i, b in enumerate(valid_budget):
    target_time = target_verify_time[i]
    






