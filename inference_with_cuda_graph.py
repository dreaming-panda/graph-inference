import torch
from Engine import InferenceEngine
from Llama_utils import _make_causal_mask
DTYPE = torch.float32
DEVICE = "cuda:0"
MAX_LENGTH = 256

MODEL_NAME = "JackFram/llama-68m"
static_input_ids = torch.Tensor(
[
    [    1, 21429, 29899,  6451, 22545,  1078,   505, 1063],
]
).long().cuda()

static_position_ids = torch.Tensor(
[
    [    1, 2, 2,  3, 3,  3,   4, 5],
]
).long().cuda()

static_storage_ids = torch.Tensor(
[
    0, 1, 2,  3, 4,  5,  6, 7
]
).long().cuda()
static_attn_mask = torch.full((static_input_ids.shape[1], MAX_LENGTH), torch.tensor(torch.finfo(DTYPE).min, device=DEVICE), device=DEVICE)
static_attn_mask[:,:static_input_ids.shape[1]] = _make_causal_mask(input_ids_shape=static_input_ids.shape, dtype=DTYPE, device=DEVICE)

static_attn_mask[7][6] = torch.finfo(DTYPE).min
static_attn_mask[5][4] = torch.finfo(DTYPE).min
static_attn_mask = static_attn_mask[None, None,:,:]

input_ids = static_input_ids.clone()
storage_ids=static_storage_ids.clone()
position_ids = static_position_ids.clone()
attention_mask=static_attn_mask.clone()

engine = InferenceEngine(max_length=MAX_LENGTH, model_name_or_path=MODEL_NAME, dtype=DTYPE, device=DEVICE)



s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        logits = engine.model_run(input_ids=static_input_ids, storage_ids=static_storage_ids, position_ids=static_position_ids, attention_mask=static_attn_mask)
    s.synchronize()
    engine.clear_kv()
torch.cuda.current_stream().wait_stream(s)


graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_logits = engine.model_run(input_ids=static_input_ids, storage_ids=static_storage_ids, position_ids=static_position_ids, attention_mask=static_attn_mask)

static_input_ids.copy_(input_ids)
static_storage_ids.copy_(storage_ids)
static_position_ids.copy_(position_ids)
static_attn_mask.copy_(attention_mask)

graph.replay()
print(static_logits.clone())


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



static_input_ids.copy_(extra_input_ids)
static_storage_ids.copy_(extra_storage_ids)
static_position_ids.copy_(extra_position_ids)
static_attn_mask.copy_(extra_attn_mask)

graph.replay()
print(static_logits.clone())

k, v = engine.get_kv_cache()
print(k[1][...,:16, :])













