import torch

def benchmark_matmul_tanh(N, num_iters=100):
    # allocate matrices on gpu
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    x = torch.randn(N, 1, device='cuda', dtype=torch.float32)

    # warmup iterations to avoid first-run jit overheads
    for _ in range(10):
        _ = torch.tanh(A @ x)

    # cuda event timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    real_end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()  # ensure all previous operations are done
    start_event.record()

    for _ in range(num_iters):
        y = torch.tanh(A @ x)

    end_event.record()
    torch.cuda.synchronize()  # ensure all operations are finished
    real_end_event.record()

    elapsed_time_ms = start_event.elapsed_time(end_event)  # time in ms
    avg_latency_ms = elapsed_time_ms / num_iters

    extra_wrapup_time = end_event.elapsed_time(real_end_event)  # time in ms

    return avg_latency_ms

# run for different N values
Ns = [1000, 5000, 10000, 20000, 30000, 40000]  # adjust as needed
for N in Ns:
    latency = benchmark_matmul_tanh(N)
    print(f"N={N}, latency={latency:.3f} ms")