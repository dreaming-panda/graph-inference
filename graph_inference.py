import torch
from Engine import GraphInferenceEngine
from Llama_utils import _make_causal_mask

DTYPE = torch.float32
DEVICE = "cuda:0"
MAX_LENGTH = 256
DEC_LENGTH = 8
MODEL_NAME = "JackFram/llama-68m"

engine = GraphInferenceEngine(max_length=MAX_LENGTH, model_name_or_path=MODEL_NAME, dtype=DTYPE, device=DEVICE)
engine.initialize_cuda_graph([DEC_LENGTH])

input_ids = torch.Tensor(
[
    [    1, 21429, 29899,  6451, 22545,  1078,   505, 1063],
]
).long().cuda()

position_ids = torch.Tensor(
[
    [    1, 2, 2,  3, 3,  3,   4, 5],
]
).long().cuda()

storage_ids = torch.Tensor(
[
    0, 1, 2,  3, 4,  5,  6, 7
]
).long().cuda()
attn_mask = torch.full((input_ids.shape[1], MAX_LENGTH), torch.tensor(torch.finfo(DTYPE).min, device=DEVICE), device=DEVICE)
attn_mask[:,:input_ids.shape[1]] = _make_causal_mask(input_ids_shape=input_ids.shape, dtype=DTYPE, device=DEVICE)
attn_mask[7][6] = torch.finfo(DTYPE).min
attn_mask[5][4] = torch.finfo(DTYPE).min
attn_mask = attn_mask[None, None,:,:]


logits = engine.graph_inference(input_ids=input_ids, storage_ids=storage_ids, position_ids=position_ids, attn_mask=attn_mask)
print(logits)


extra_input_ids = torch.Tensor(
[
    [    1472,21429, 29899, 6451, 22545,  1078,   505, 1063],
]
).long().cuda()

extra_position_ids = torch.Tensor(
[
    [    4, 2, 7, 3, 4, 6, 8, 8],
]
).long().cuda()
extra_storage_ids = torch.Tensor(
[
    8, 9, 10,  11, 12,  13,  14, 15
]
).long().cuda()

extra_attn_mask = torch.full((extra_input_ids.shape[1], MAX_LENGTH), torch.tensor(torch.finfo(DTYPE).min, device=DEVICE), device=DEVICE)

extra_attn_mask[..., 0] = 0.0
extra_attn_mask[..., 1] = 0.0
extra_attn_mask[..., 4] = 0.0
extra_attn_mask[..., 7] = 0.0
extra_attn_mask[..., 8] = 0.0
extra_attn_mask = extra_attn_mask[None, None,:,:]

logits = engine.graph_inference(input_ids=extra_input_ids, storage_ids=extra_storage_ids, position_ids=extra_position_ids, attn_mask=extra_attn_mask)
print(logits)