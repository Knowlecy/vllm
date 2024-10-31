git clone https://github.com/Knowlecy/vllm.git
cd vllm

for tokens in 2048 1024 512 256
do
    cd ..
    vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
        --trust-remote-code \
        --enable-chunked-prefill \
        --max-num-batched-tokens $tokens \
        --max_model_len 128000 \
        --quantization awq \
        --tensor_parallel_size 2 \
        --pipeline_parallel_size 2 &
    server_pid=$!

    timeout 600 bash -c 'until curl http://127.0.0.1:8000/v1/models; do sleep 1; done' || exit 1

    cd vllm
    echo "$tokens awq benchmark serving" >> benchmark_serving.txt
    python3 benchmarks/benchmark_serving.py \
        --backend vllm \
        --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
        --dataset-name hf \
        --num-prompts 50 \
        --host 127.0.0.1 \
        2>&1 | tee -a benchmark_serving.txt

    kill $server_pid
done

cd ..
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-batched-tokens 128 \
    --max-num-seqs 128 \
    --max_model_len 128000 \
    --quantization awq \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 2 &
server_pid=$!

timeout 600 bash -c 'until curl http://127.0.0.1:8000/v1/models; do sleep 1; done' || exit 1

cd vllm
echo "128 awq benchmark serving" >> benchmark_serving.txt
python3 benchmarks/benchmark_serving.py \
    --backend vllm \
    --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --dataset-name hf \
    --num-prompts 50 \
    --host 127.0.0.1 \
    2>&1 | tee -a benchmark_serving.txt

kill $server_pid

cd ..
vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-batched-tokens 64 \
    --max-num-seqs 64 \
    --max_model_len 128000 \
    --quantization awq \
    --tensor_parallel_size 2 \
    --pipeline_parallel_size 2 &
server_pid=$!

timeout 600 bash -c 'until curl http://127.0.0.1:8000/v1/models; do sleep 1; done' || exit 1

cd vllm
echo "64 awq benchmark serving" >> benchmark_serving.txt
python3 benchmarks/benchmark_serving.py \
    --backend vllm \
    --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --dataset-name hf \
    --num-prompts 50 \
    --host 127.0.0.1 \
    2>&1 | tee -a benchmark_serving.txt

kill $server_pid
