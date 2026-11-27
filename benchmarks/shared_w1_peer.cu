#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int VectorBytes = sizeof(uint4);
constexpr int Threads = 128;

__device__ uint64_t gpu_time() {
    uint64_t ts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts) :: "memory");
    return ts;
}

__device__ uint32_t load_ready(const uint32_t* address) {
    uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                 : "=r"(value) : "l"(address) : "memory");
    return value;
}

__device__ void publish(uint32_t* address, uint32_t value) {
    asm volatile("st.release.sys.global.u32 [%0], %1;"
                 :: "l"(address), "r"(value) : "memory");
}

__device__ void wait_ready(const uint32_t* address, uint32_t epoch) {
    const auto started = clock64();
    while (load_ready(address) != epoch) {
        if (clock64() - started > 2000000000ULL) {
            asm volatile("trap;");
        }
        __nanosleep(64);
    }
}

__device__ void wait_reuse(const uint32_t* entered, uint32_t previous_epoch) {
    const auto started = clock64();
    // With two banks the receiver may already have entered the new epoch.
    while (static_cast<int32_t>(load_ready(entered) - previous_epoch) < 0) {
        if (clock64() - started > 2000000000ULL) {
            asm volatile("trap;");
        }
        __nanosleep(64);
    }
}

__global__ void gather_pair(const uint4* local, const uint4* peer,
                            uint4* paired, int vectors, uint64_t* trace) {
    if (trace && threadIdx.x == 0) {
        trace[blockIdx.x * 2] = gpu_time();
    }
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < vectors) {
        paired[index] = local[index];
        paired[vectors + index] = peer[index];
    }
    if (trace) {
        __syncthreads();
        if (threadIdx.x == 0) {
            trace[blockIdx.x * 2 + 1] = gpu_time();
        }
    }
}

__device__ void transfer_halves(const uint4* own_source, const uint4* remote_source,
                                uint4* own_target, uint4* remote_target,
                                unsigned scratch, unsigned barrier, int bytes) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier), "r"(bytes * 2) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(scratch), "l"(own_source), "r"(bytes), "r"(barrier) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(scratch + bytes), "l"(remote_source), "r"(bytes), "r"(barrier) : "memory");
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; "
            "selp.u32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier) : "memory");
    } while (!ready);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(own_target), "r"(scratch), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(remote_target), "r"(scratch + bytes), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
    __threadfence_system();
}

template <bool Bulk, bool DoubleBuffered = false>
__global__ void exchange_halves(const uint4* projections, uint4* local, uint4* peer,
                                 int half_vectors, int shard, uint64_t* trace) {
    const int token = blockIdx.x;
    const int tokens = gridDim.x;
    const int row_vectors = 2 * half_vectors;
    const int bank_vectors = tokens * row_vectors;
    const int flag_offset = (DoubleBuffered ? 2 : 1) * bank_vectors;
    auto* entered = reinterpret_cast<uint32_t*>(local + flag_offset);
    auto* arrived = entered + tokens;
    auto* peer_entered = reinterpret_cast<uint32_t*>(peer + flag_offset);
    auto* peer_arrived = peer_entered + tokens;
    __shared__ uint32_t epoch;
    __shared__ uint64_t transfer_barrier;
    extern __shared__ __align__(16) uint8_t scratch[];

    if (threadIdx.x == 0) {
        if (trace) {
            trace[token * 4] = gpu_time();
        }
        epoch = load_ready(entered + token) + 1;
        // At this point the previous W2 has already consumed this row.
        publish(entered + token, epoch);
        if constexpr (DoubleBuffered) {
            // Entry to epoch N-1 retires the bank last used in epoch N-2.
            wait_reuse(peer_entered + token, epoch - 1);
        } else {
            wait_ready(peer_entered + token, epoch);
        }
        if (trace) {
            trace[token * 4 + 1] = gpu_time();
        }
    }
    __syncthreads();

    const int bank = DoubleBuffered ? (epoch & 1) : 0;
    const int dst = bank * bank_vectors + token * row_vectors + shard * half_vectors;
    if constexpr (Bulk) {
        if (threadIdx.x == 0) {
            transfer_halves(projections + token * half_vectors,
                            projections + (tokens + token) * half_vectors,
                            local + dst, peer + dst, __cvta_generic_to_shared(scratch),
                            __cvta_generic_to_shared(&transfer_barrier),
                            half_vectors * VectorBytes);
        }
    } else {
        for (int vector = threadIdx.x; vector < half_vectors; vector += blockDim.x) {
            local[dst + vector] = projections[token * half_vectors + vector];
            peer[dst + vector] = projections[(tokens + token) * half_vectors + vector];
            __threadfence_system();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (trace) {
            trace[token * 4 + 2] = gpu_time();
        }
        publish(peer_arrived + bank * tokens + token, epoch);
        wait_ready(arrived + bank * tokens + token, epoch);
        if (trace) {
            trace[token * 4 + 3] = gpu_time();
        }
    }
}

__global__ void return_owned_rows(const uint4* results, uint4* local, uint4* peer,
                                  int row_vectors, bool owner) {
    const int row = blockIdx.x;
    const int tokens = gridDim.x;
    const int bank_vectors = tokens * row_vectors;
    auto* local_entered = reinterpret_cast<uint32_t*>(local + 2 * bank_vectors);
    auto* peer_entered = reinterpret_cast<uint32_t*>(peer + 2 * bank_vectors);
    __shared__ uint64_t complete;
    extern __shared__ __align__(16) uint8_t scratch[];
    if (threadIdx.x == 0) {
        const uint32_t epoch = load_ready(local_entered + row) + 1;
        publish(local_entered + row, epoch);
        const int bank = epoch & 1;
        if (owner) {
            wait_reuse(peer_entered + row, epoch - 1);
            const auto* source = results + (tokens + row) * row_vectors;
            auto* target = peer + bank * bank_vectors + row * row_vectors;
            const int half = row_vectors / 2;
            transfer_halves(source, source + half, target, target + half,
                            __cvta_generic_to_shared(scratch),
                            __cvta_generic_to_shared(&complete), half * VectorBytes);
            publish(peer_entered + (bank + 1) * tokens + row, epoch);
        } else {
            wait_ready(local_entered + (bank + 1) * tokens + row, epoch);
        }
    }
}
}  // namespace

extern "C" int gc_shared_gather_pair(const void* local, const void* peer,
                                     void* paired, int bytes, uint64_t* trace, cudaStream_t stream) {
    const int vectors = bytes / VectorBytes;
    gather_pair<<<(vectors + Threads - 1) / Threads, Threads, 0, stream>>>(
        static_cast<const uint4*>(local), static_cast<const uint4*>(peer),
        static_cast<uint4*>(paired), vectors, trace);
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int gc_shared_exchange_halves(const void* projections, void* local,
                                          void* peer, int tokens, int half_bytes,
                                          int shard, uint64_t* trace, cudaStream_t stream) {
    exchange_halves<false><<<tokens, half_bytes / VectorBytes, 0, stream>>>(
        static_cast<const uint4*>(projections), static_cast<uint4*>(local),
        static_cast<uint4*>(peer), half_bytes / VectorBytes, shard, trace);
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int gc_shared_exchange_bulk(const void* projections, void* local,
                                        void* peer, int tokens, int half_bytes,
                                        int shard, uint64_t* trace, cudaStream_t stream) {
    exchange_halves<true><<<tokens, 32, 2 * half_bytes, stream>>>(
        static_cast<const uint4*>(projections), static_cast<uint4*>(local),
        static_cast<uint4*>(peer), half_bytes / VectorBytes, shard, trace);
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int gc_shared_exchange_double_buffered(const void* projections, void* local,
                                                   void* peer, int tokens, int half_bytes,
                                                   int shard, uint64_t* trace, cudaStream_t stream) {
    exchange_halves<true, true><<<tokens, 32, 2 * half_bytes, stream>>>(
        static_cast<const uint4*>(projections), static_cast<uint4*>(local),
        static_cast<uint4*>(peer), half_bytes / VectorBytes, shard, trace);
    return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int gc_shared_return_owned_rows(const void* results, void* local, void* peer,
                                            int tokens, int row_bytes, int owner,
                                            cudaStream_t stream) {
    return_owned_rows<<<tokens, 32, owner ? row_bytes : 0, stream>>>(
        static_cast<const uint4*>(results), static_cast<uint4*>(local),
        static_cast<uint4*>(peer), row_bytes / VectorBytes, owner != 0);
    return static_cast<int>(cudaPeekAtLastError());
}
