#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

namespace {
namespace cg = cooperative_groups;
constexpr int K = 2048;
constexpr int RowBytes = K + K / 32;
constexpr int VectorBytes = sizeof(uint4);
constexpr int Trials = 100;
enum class Path { GlobalTma, DsmVector, DsmBulk };

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

template <class T>
T* allocate(size_t count) {
    T* value = nullptr;
    check(cudaMalloc(&value, count * sizeof(T)), "allocate");
    return value;
}

__device__ uint64_t timer() {
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
    return now;
}

__device__ void spill_to_global(const uint4* source, uint4* target, unsigned bytes) {
    const unsigned scratch = __cvta_generic_to_shared(source);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(target), "r"(scratch), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
    __threadfence();
}

__device__ void expect(unsigned barrier, unsigned bytes) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier), "r"(bytes) : "memory");
}

__device__ void wait_for_copy(unsigned barrier) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; "
            "selp.u32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier) : "memory");
    } while (!ready);
}

__device__ void read_from_global(uint4* target, const uint4* source,
                                 unsigned bytes, uint64_t* completed) {
    const unsigned scratch = __cvta_generic_to_shared(target);
    const unsigned barrier = __cvta_generic_to_shared(completed);
    expect(barrier, bytes);
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(scratch), "l"(source), "r"(bytes), "r"(barrier) : "memory");
    wait_for_copy(barrier);
}

__device__ unsigned cluster_address(unsigned local, unsigned destination) {
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote) : "r"(local), "r"(destination));
    return remote;
}

__device__ void push_to_peer(const uint4* source, uint4* target,
                             unsigned bytes, unsigned barrier, unsigned peer) {
    const unsigned src = __cvta_generic_to_shared(source);
    const unsigned dst = cluster_address(__cvta_generic_to_shared(target), peer);
    const unsigned ready = cluster_address(barrier, peer);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 :: "r"(dst), "r"(src), "r"(bytes), "r"(ready) : "memory");
}

template <Path Copy>
__global__ void handoff(const uint4* w1, uint4* staging, uint4* w2,
                        uint64_t* trace, int vectors, int trial) {
    const auto cluster = cg::this_cluster();
    const unsigned members = cluster.num_blocks();
    const unsigned shard = cluster.block_rank();
    const unsigned expert = blockIdx.x / members;
    const unsigned shard_size = vectors / members;
    const unsigned scratch_offset = expert * vectors + shard * shard_size;
    extern __shared__ __align__(16) uint4 scratch[];
    __shared__ uint64_t complete;
    const unsigned barrier = __cvta_generic_to_shared(&complete);
    uint4* gathered = scratch + shard_size;
    if constexpr (Copy == Path::DsmBulk) {
        if (threadIdx.x == 0) {
            expect(barrier, (members - 1) * shard_size * VectorBytes);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    // Stand in for a completed W1 epilogue. The staging read is untimed.
    for (unsigned index = threadIdx.x; index < shard_size; index += blockDim.x) {
        scratch[index] = w1[scratch_offset + index];
    }
    __syncthreads();
    const unsigned point = (trial * gridDim.x + blockIdx.x) * 2;
    if (threadIdx.x == 0) {
        trace[point] = timer();
        if constexpr (Copy == Path::GlobalTma) {
            spill_to_global(scratch, staging + scratch_offset, shard_size * VectorBytes);
        }
    }
    cluster.sync();
    if constexpr (Copy == Path::DsmVector) {
        for (unsigned step = 0; step < members; ++step) {
            const unsigned owner = (shard + step) % members;
            const uint4* peer = cluster.map_shared_rank(scratch, owner);
            uint4* target = gathered + owner * shard_size;
            for (unsigned index = threadIdx.x; index < shard_size; index += blockDim.x) {
                target[index] = peer[index];
            }
        }
    } else if constexpr (Copy == Path::DsmBulk) {
        const unsigned peer = (shard + threadIdx.x) % members;
        uint4* shard_target = gathered + shard * shard_size;
        if (threadIdx.x > 0 && threadIdx.x < members) {
            push_to_peer(scratch, shard_target, shard_size * VectorBytes, barrier, peer);
        }
        for (unsigned index = threadIdx.x; index < shard_size; index += blockDim.x) {
            shard_target[index] = scratch[index];
        }
        if (threadIdx.x == 0) {
            wait_for_copy(barrier);
        }
    } else {
        if (threadIdx.x == 0) {
            read_from_global(gathered, staging + expert * vectors, vectors * VectorBytes, &complete);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        trace[point + 1] = timer();
    }
    // Keep the copies live and validate every byte without timing the sink.
    for (unsigned index = threadIdx.x; index < vectors; index += blockDim.x) {
        w2[blockIdx.x * vectors + index] = gathered[index];
    }
    // Keep each distributed source alive until every block is done reading.
    cluster.sync();
}

uint32_t contents(size_t value) {
    return static_cast<uint32_t>(value * 2654435761ULL) ^ 0x8d23d1edU;
}

double percentile(const std::vector<double>& sorted, size_t percent) {
    const size_t index = (percent * sorted.size() + 99) / 100 - 1;
    return sorted.at(index);
}

template <Path Copy>
void run(int tokens, int experts, int members, int threads) {
    const unsigned blocks = experts * members;
    const unsigned vectors = tokens * RowBytes / VectorBytes;
    const size_t outputs = static_cast<size_t>(blocks) * vectors;
    auto* input = allocate<uint4>(experts * vectors);
    auto* staging = allocate<uint4>(experts * vectors);
    auto* output = allocate<uint4>(outputs);
    auto* trace = allocate<uint64_t>(Trials * blocks * 2);
    std::vector<uint32_t> from_host(experts * vectors * 4);
    for (size_t index = 0; index < from_host.size(); ++index) {
        from_host[index] = contents(index);
    }
    check(cudaMemcpy(input, from_host.data(), from_host.size() * sizeof(uint32_t),
                     cudaMemcpyHostToDevice), "copy input");

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = members;
    attribute.val.clusterDim.y = attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(blocks);
    launch.blockDim = dim3(threads);
    launch.dynamicSmemBytes = (vectors + vectors / members) * VectorBytes;
    launch.attrs = &attribute;
    launch.numAttrs = 1;
    check(cudaFuncSetAttribute(handoff<Copy>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               launch.dynamicSmemBytes), "dynamic shared memory");
    for (int trial = 0; trial < Trials; ++trial) {
        check(cudaLaunchKernelEx(&launch, handoff<Copy>, input, staging, output,
                                 trace, vectors, trial), "launch");
    }
    check(cudaDeviceSynchronize(), "synchronize");

    std::vector<uint64_t> measurements(Trials * blocks * 2);
    check(cudaMemcpy(measurements.data(), trace, measurements.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost), "copy timestamps");
    std::vector<uint32_t> to_host(outputs * 4);
    check(cudaMemcpy(to_host.data(), output, to_host.size() * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost), "copy results");
    for (unsigned block = 0; block < blocks; ++block) {
        const auto from = from_host.begin() + block / members * vectors * 4;
        const auto to = to_host.begin() + block * vectors * 4;
        if (!std::equal(from, from + vectors * 4, to)) {
            std::fprintf(stderr, "handoff mismatch mode=%d block=%u\n", int(Copy), block);
            std::exit(1);
        }
    }
    // Exclude warmup launches and report per-round slowest cluster.
    std::vector<double> times;
    for (int trial = 12; trial < Trials; ++trial) {
        uint64_t longest = 0;
        for (int expert = 0; expert < experts; ++expert) {
            uint64_t first = UINT64_MAX;
            uint64_t last = 0;
            for (int member = 0; member < members; ++member) {
                const size_t point = (trial * blocks + expert * members + member) * 2;
                first = std::min(first, measurements[point]);
                last = std::max(last, measurements[point + 1]);
            }
            longest = std::max(longest, last - first);
        }
        times.push_back(longest / 1000.0);
    }
    std::sort(times.begin(), times.end());
    std::printf("{\"mode\":\"%s\",\"rows\":%d,\"experts\":%d,\"ctas\":%d,"
                "\"threads\":%d,\"p50_us\":%.3f,\"p95_us\":%.3f,\"exact\":true}\n",
                Copy == Path::DsmBulk ? "distributed_bulk" :
                Copy == Path::DsmVector ? "distributed_shared" : "global_tma", tokens, experts,
                members, threads, percentile(times, 50), percentile(times, 95));
    check(cudaFree(input), "free input");
    check(cudaFree(staging), "free staging");
    check(cudaFree(output), "free output");
    check(cudaFree(trace), "free timestamps");
}
}  // namespace

int main() {
    for (int experts : {8, 16}) {
        for (int tokens : {16, 32, 48}) {
            for (int members : {4, 8}) {
                for (int threads : {128, 256}) {
                    run<Path::GlobalTma>(tokens, experts, members, threads);
                    run<Path::DsmVector>(tokens, experts, members, threads);
                    run<Path::DsmBulk>(tokens, experts, members, threads);
                }
            }
        }
    }
}
