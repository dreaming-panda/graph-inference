import torch
from transformers import LlamaForCausalLM
import argparse
import time
import deepspeed
#from Llama import LlamaForCausalLM_Attn
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="TheBloke/Llama-2-70B-AWQ",help='model')
parser.add_argument('--T', type=int, default=100, help='repeat times')
parser.add_argument('--B', type=int, default=1, help='batch size')
parser.add_argument('--P', type=int, default=128, help='prefix length')
args = parser.parse_args()
print(args)

#target_model = LlamaForCausalLM_Attn.from_pretrained(args.target, torch_dtype=torch.float16, device_map="auto")
draft_model = LlamaForCausalLM.from_pretrained(args.model).cuda()


# draft_model = deepspeed.init_inference(draft_model,  
#                 dtype=torch.float16, enable_cuda_graph=True)
T = args.T
B = args.B
P = args.P
LEN = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128]
prefix = torch.randint(low=3, high=30000, size=(B, P)).cuda()
past_key_values = draft_model(input_ids = prefix, use_cache=True).past_key_values


PERFORMANCE = []

for l in LEN:

    sentence = torch.randint(low=3, high=30000, size=(B,  l)).cuda()
    total_time = 0.0
    for _ in range(3):
        output = draft_model(input_ids = sentence, use_cache=True, past_key_values=past_key_values)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(T):
    
        output = draft_model(input_ids = sentence, use_cache=True, past_key_values=past_key_values)
        
       
    torch.cuda.synchronize()
    t2 = time.time()
    total_time += (t2 - t1)
    PERFORMANCE.append(total_time / T)

for i, l in enumerate(LEN):
    print("Length :{}, inference time:{}".format(l, PERFORMANCE[i]))








