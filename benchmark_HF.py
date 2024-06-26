import torch
from transformers import LlamaForCausalLM
import argparse
import time
from torch.profiler import profile, record_function, ProfilerActivity
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf",help='model')
parser.add_argument('--T', type=int, default=200, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')

parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--D', type=int, default=8, help='dec length')
parser.add_argument('--L', type=int, default=1, help='dec length')
args = parser.parse_args()
print(args)

#target_model = LlamaForCausalLM_Attn.from_pretrained(args.target, torch_dtype=torch.float16, device_map="auto")
draft_model = LlamaForCausalLM.from_pretrained(args.model)


# draft_model = deepspeed.init_inference(draft_model,  
#                 dtype=torch.float16, enable_cuda_graph=True)
T = args.T
B = args.B
P = args.P
LEN = [1, 32, 64, 128, 192, 256]
prefix = torch.randint(low=3, high=30000, size=(B, P))
past_key_values = draft_model(input_ids = prefix, use_cache=True).past_key_values

PERFORMANCE = []
with torch.no_grad():
    for l in LEN:

        sentence = torch.randint(low=3, high=30000, size=(B,  l))
        total_time = 0.0
        for _ in range(20):
            output = draft_model(input_ids = sentence, use_cache=True, past_key_values=past_key_values)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(T):
                output = draft_model(input_ids = sentence, use_cache=True, past_key_values=past_key_values)
        torch.cuda.synchronize()
        t2 = time.time()
        total_time += (t2 - t1)
        print("Length :{}, inference time:{}".format(l, total_time / T))
    
    
