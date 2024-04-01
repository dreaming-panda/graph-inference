import deepspeed
from transformers import LlamaForCausalLM
from deepspeed.inference.config import DeepSpeedTPConfig
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import time
# parser = argparse.ArgumentParser(description="deepspeed llama benchmark")
# parser.add_argument("--batch", type=int, help="batch size for inference")
# parser.add_argument("--tp", type=int, help="tp size")
# args = parser.parse_args()
TP = 1
BSZ = 1
GEN_LEN = 1
PROMPT_LEN = 64
T = 1000
tp_config = DeepSpeedTPConfig(
    tp_size=TP
)
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
if TP > 1:
    model = deepspeed.init_inference(model, 
                                 tensor_parallel=tp_config, 
                                 dtype=torch.float16,
                                 enable_cuda_graph=True,
                                replace_with_kernel_inject=True)
else:
    model = deepspeed.init_inference(model,  
                dtype=torch.float16, enable_cuda_graph=True, replace_with_kernel_inject=True)
prompt_ids = torch.randint(low=10, high=30000, size=(BSZ, PROMPT_LEN)).long().cuda()
input_ids = torch.randint(low=10, high=30000, size=(BSZ, GEN_LEN)).long().cuda()
outputs = model(prompt_ids, return_dict=True, use_cache=True)
past_key_values = outputs.past_key_values

for _ in range(10):
        outputs = model(input_ids, use_cache=True, past_key_values=past_key_values, return_dict=True)

torch.cuda.synchronize()
t1 = time.time()
for _ in range(T):
        outputs = model(input_ids, use_cache=True, past_key_values=past_key_values, return_dict=True)


torch.cuda.synchronize()
t2 = time.time()
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(TP, GEN_LEN, (t2 - t1) / T)


