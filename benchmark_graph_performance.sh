#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-1.3B --M 256 --P 128  --D 1 >> benchmark/1.3B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-1.3B --M 256 --P 128  --D 2 >> benchmark/1.3B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-1.3B --M 256 --P 128  --D 4 >> benchmark/1.3B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-1.3B --M 256 --P 128  --D 8 >> benchmark/1.3B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-1.3B --M 256 --P 128  --D 16 >> benchmark/1.3B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-1.3B --M 256 --P 128  --D 32 >> benchmark/1.3B_benchmark.log


#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-2.7B --M 256 --P 128  --D 1 >> benchmark/2.7B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-2.7B --M 256 --P 128  --D 2 >> benchmark/2.7B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-2.7B --M 256 --P 128  --D 4 >> benchmark/2.7B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-2.7B --M 256 --P 128  --D 8 >> benchmark/2.7B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-2.7B --M 256 --P 128  --D 16 >> benchmark/2.7B_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model princeton-nlp/Sheared-LLaMA-2.7B --M 256 --P 128  --D 32 >> benchmark/2.7B_benchmark.log

#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-68m --M 256 --P 128  --D 1 >> benchmark/68m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-68m --M 256 --P 128  --D 2 >> benchmark/68m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-68m --M 256 --P 128  --D 4 >> benchmark/68m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-68m --M 256 --P 128  --D 8 >> benchmark/68m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-68m --M 256 --P 128  --D 16 >> benchmark/68m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-68m --M 256 --P 128  --D 32 >> benchmark/68m_benchmark.log

#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-160m --M 256 --P 128  --D 1 >> benchmark/160m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-160m --M 256 --P 128  --D 2 >> benchmark/160m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-160m --M 256 --P 128  --D 4 >> benchmark/160m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-160m --M 256 --P 128  --D 8 >> benchmark/160m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-160m --M 256 --P 128  --D 16 >> benchmark/160m_benchmark.log
#CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model JackFram/llama-160m --M 256 --P 128  --D 32 >> benchmark/160m_benchmark.log

# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 1 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 2 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 4 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 8 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 16 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 24 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 32 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 48 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 64 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 80 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 96 >> benchmark/7B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model meta-llama/Llama-2-7b-hf --M 256 --P 128  --D 128 >> benchmark/7B_benchmark.log


# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 1 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 2 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 4 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 8 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 16 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 24 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 32 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 48 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 64 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 80 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 96 >> benchmark/33B_benchmark.log
# CUDA_VISIBLE_DEVICES=0 python bencmark_graph_inference.py --model lmsys/vicuna-33b-v1.3 --M 256 --P 128  --D 128 >> benchmark/33B_benchmark.log

#CUDA_VISIBLE_DEVICES=0 python benchmark_awq.py --model TheBloke/vicuna-33B-AWQ >> benchmark/33BAWQ_benchmark.log
CUDA_VISIBLE_DEVICES=0 python benchmark_HF.py --model meta-llama/Llama-2-7b-hf >> benchmark/HF_7B_benchmark.log
CUDA_VISIBLE_DEVICES=0 python benchmark_HF.py --model meta-llama/Llama-2-13b-hf >> benchmark/HF_13B_benchmark.log
CUDA_VISIBLE_DEVICES=0 python benchmark_HF.py --model deepseek-ai/deepseek-coder-33b-base >> benchmark/HF_33B_deepseek_benchmark.log

CUDA_VISIBLE_DEVICES=0,1 python benchmark_HF.py --model meta-llama/Llama-2-70b-hf >> benchmark/70B_benchmark.log













