import torch
N, D, M= 3, 8, 3
offset = torch.tensor([0,2], device='cuda').long()
def copy_tensor(x: torch.Tensor, y :torch.Tensor, offset :torch.Tensor):
    x.index_copy_(dim=1, index=offset, source=y)
# Placeholders used for capture
static_input = torch.zeros(N, D, M, device='cuda')
static_target = torch.ones(N, 2, M, device='cuda')
static_target[..., 1, :] = 2

# warmup
# Uses static_input and static_target here for convenience,
# but in a real setting, because the warmup includes optimizer.step()
# you must use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    copy_tensor(static_input, static_target, offset)
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool

with torch.cuda.graph(g):
    copy_tensor(static_input, static_target, offset)


real_inputs = [torch.zeros_like(static_input) for _ in range(4)]
real_targets = [torch.ones_like(static_target) for _ in range(4)]
print(static_input)


for i in range(2):
    # Fills the graph's input memory with new data to compute on
    # static_input.copy_(data)
    # static_target.copy_(target)

   
    # replay() includes forward, backward, and step.
    # You don't even need to call optimizer.zero_grad() between iterations
    # because the captured backward refills static .grad tensors in place.
    print(static_input)
    offset.add_(1)
    g.replay()
    
    # Params have been updated. static_y_pred, static_loss, and .grad
    # attributes hold values from computing on this iteration's data.
print(static_input)

