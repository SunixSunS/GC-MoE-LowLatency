#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int Peers = 4;
constexpr int Tokens = 32;
constexpr int Hidden = 6144;
constexpr int RowPitch = 8192;
constexpr int KTile = 128;
constexpr int Groups = 8;
constexpr int ScratchBytes = Groups * Peers * KTile;

void driver_check(CUresult result, const char* action) {
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        cuGetErrorName(result, &name);
        std::fprintf(stderr, "%s: %s\n", action, name ? name : "unknown CUDA error");
        std::exit(1);
    }
}

void runtime_check(cudaError_t result, const char* action) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", action, cudaGetErrorString(result));
        std::exit(1);
    }
}

__host__ __device__ unsigned char value(int peer, int token, int col) {
    return static_cast<unsigned char>(peer * 67 + token * 31 + col * 7 + col / KTile);
}

struct GatherMap {
    CUdeviceptr base = 0;
    size_t segment_bytes = 0;
    int segment_rows = 0;
    CUtensorMap descriptor{};

    explicit GatherMap(bool remote) {
        std::array<CUmemAllocationProp, Peers> props{};
        segment_bytes = Tokens * RowPitch;
        for (int peer = 0; peer < Peers; ++peer) {
            auto& prop = props[peer];
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = remote ? peer : 0;
            size_t bytes = 0;
            driver_check(cuMemGetAllocationGranularity(&bytes, &prop,
                                                       CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
                         "get allocation granularity");
            segment_bytes = std::max(segment_bytes, bytes);
        }
        segment_rows = static_cast<int>(segment_bytes / RowPitch);
        if (segment_rows * RowPitch != segment_bytes) {
            std::fprintf(stderr, "segment size is not an integral number of rows\n");
            std::exit(1);
        }
        driver_check(cuMemAddressReserve(&base, Peers * segment_bytes, 0, 0, 0), "reserve peer rows");
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = 0;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        std::vector<unsigned char> host(Tokens * RowPitch);
        for (int peer = 0; peer < Peers; ++peer) {
            CUmemGenericAllocationHandle allocation{};
            driver_check(cuMemCreate(&allocation, segment_bytes, &props[peer], 0), "allocate peer rows");
            const CUdeviceptr segment = base + peer * segment_bytes;
            driver_check(cuMemMap(segment, segment_bytes, 0, allocation, 0), "map peer rows");
            driver_check(cuMemRelease(allocation), "release allocation reference");
            driver_check(cuMemSetAccess(segment, segment_bytes, &access, 1), "access mapped peer rows");
            for (int token = 0; token < Tokens; ++token) {
                for (int col = 0; col < Hidden; ++col) {
                    host[token * RowPitch + col] = value(peer, token, col);
                }
            }
            driver_check(cuMemcpyHtoD(segment, host.data(), host.size()), "fill mapped peer rows");
        }
        const cuuint64_t sizes[] = {Hidden, static_cast<cuuint64_t>((Peers - 1) * segment_rows + Tokens)};
        const cuuint64_t strides[] = {RowPitch};
        const cuuint32_t box[] = {KTile, 1};
        const cuuint32_t elements[] = {1, 1};
        driver_check(cuTensorMapEncodeTiled(&descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                                            reinterpret_cast<void*>(base), sizes, strides, box, elements,
                                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                     "encode multi-peer TensorMap");
    }

    ~GatherMap() {
        for (int peer = 0; peer < Peers; ++peer) {
            driver_check(cuMemUnmap(base + peer * segment_bytes, segment_bytes), "unmap peer rows");
        }
        driver_check(cuMemAddressFree(base, Peers * segment_bytes), "release peer addresses");
    }

    GatherMap(const GatherMap&) = delete;
    GatherMap& operator=(const GatherMap&) = delete;
};

__device__ int token_for(int block, int group, int peer) {
    return (block + group * Peers + peer * 3) % Tokens;
}

__device__ void issue_tile(const CUtensorMap* map, int segment_rows, int tile,
                           unsigned char* scratch, unsigned barrier) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier), "r"(ScratchBytes) : "memory");
    for (int group = 0; group < Groups; ++group) {
        const unsigned dst = __cvta_generic_to_shared(scratch + group * Peers * KTile);
        const int row0 = token_for(blockIdx.x, group, 0);
        const int row1 = segment_rows + token_for(blockIdx.x, group, 1);
        const int row2 = segment_rows * 2 + token_for(blockIdx.x, group, 2);
        const int row3 = segment_rows * 3 + token_for(blockIdx.x, group, 3);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
            :: "r"(dst), "l"(map), "r"(tile * KTile), "r"(row0), "r"(row1),
               "r"(row2), "r"(row3), "r"(barrier) : "memory");
    }
}

__device__ void wait_tile(unsigned barrier, int parity) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
            "p, [%1], %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(parity) : "memory");
    } while (!ready);
}

__device__ std::uint64_t global_timer() {
    std::uint64_t time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time) :: "memory");
    return time;
}

template <int Stages, bool Validate, int Tiles = Hidden / KTile>
__global__ void gather(const __grid_constant__ CUtensorMap map, int segment_rows,
                       unsigned* errors, unsigned char* consumed, std::uint64_t* elapsed = nullptr) {
    static_assert(Tiles % Stages == 0 && Tiles <= Hidden / KTile);
    __shared__ __align__(128) unsigned char scratch[Stages][ScratchBytes];
    __shared__ unsigned long long complete[Stages];
    if (threadIdx.x == 0) {
        for (int stage = 0; stage < Stages; ++stage) {
            const unsigned barrier = __cvta_generic_to_shared(&complete[stage]);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    std::uint64_t start = 0;
    if constexpr (Tiles < Hidden / KTile) {
        if (threadIdx.x == 0) {
            start = global_timer();
        }
    }
    for (int tile_base = 0; tile_base < Tiles; tile_base += Stages) {
        if (threadIdx.x == 0) {
            for (int stage = 0; stage < Stages; ++stage) {
                issue_tile(&map, segment_rows, tile_base + stage, scratch[stage],
                           __cvta_generic_to_shared(&complete[stage]));
            }
            for (int stage = 0; stage < Stages; ++stage) {
                wait_tile(__cvta_generic_to_shared(&complete[stage]), (tile_base / Stages) & 1);
            }
        }
        if constexpr (Validate) {
            __syncthreads();
            unsigned incorrect = 0;
            for (int stage = 0; stage < Stages; ++stage) {
                for (int index = threadIdx.x; index < ScratchBytes; index += blockDim.x) {
                    const int row = index / KTile;
                    const int peer = row % Peers;
                    const int token = token_for(blockIdx.x, row / Peers, peer);
                    const int col = (tile_base + stage) * KTile + index % KTile;
                    incorrect += scratch[stage][index] != value(peer, token, col);
                }
            }
            if (incorrect) {
                atomicAdd(errors, incorrect);
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        if constexpr (Tiles < Hidden / KTile) {
            elapsed[blockIdx.x] = global_timer() - start;
        }
        consumed[blockIdx.x] = scratch[Stages - 1][0];
    }
}

template <int Tiles, bool Validate>
__global__ void gather_raw(const unsigned char* src, int segment_rows, unsigned* errors,
                           unsigned char* consumed, std::uint64_t* elapsed) {
    __shared__ __align__(128) unsigned char scratch[Tiles][ScratchBytes];
    const auto start = global_timer();
    for (int tile = 0; tile < Tiles; ++tile) {
        for (int vector = threadIdx.x; vector < ScratchBytes / 16; vector += blockDim.x) {
            const int byte = vector * 16;
            const int row = byte / KTile;
            const int peer = row % Peers;
            const int token = token_for(blockIdx.x, row / Peers, peer);
            const auto* source = src + (peer * segment_rows + token) * RowPitch +
                                 tile * KTile + byte % KTile;
            const unsigned target = __cvta_generic_to_shared(scratch[tile] + byte);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(target), "l"(source) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncwarp();
    if (threadIdx.x == 0) {
        elapsed[blockIdx.x] = global_timer() - start;
        consumed[blockIdx.x] = scratch[Tiles - 1][0];
    }
    if constexpr (Validate) {
        unsigned incorrect = 0;
        for (int tile = 0; tile < Tiles; ++tile) {
            for (int index = threadIdx.x; index < ScratchBytes; index += blockDim.x) {
                const int row = index / KTile;
                const int peer = row % Peers;
                const int token = token_for(blockIdx.x, row / Peers, peer);
                incorrect += scratch[tile][index] != value(peer, token, tile * KTile + index % KTile);
            }
        }
        if (incorrect) {
            atomicAdd(errors, incorrect);
        }
    }
}

template <int Stages>
float measure(const GatherMap& inputs, int blocks) {
    constexpr int Trials = 60;
    unsigned char* consumed = nullptr;
    runtime_check(cudaMalloc(&consumed, blocks), "allocate consumer output");
    cudaEvent_t start{}, end{};
    runtime_check(cudaEventCreate(&start), "create start event");
    runtime_check(cudaEventCreate(&end), "create end event");
    for (int warmup = 0; warmup < 6; ++warmup) {
        gather<Stages, false><<<blocks, 32>>>(inputs.descriptor, inputs.segment_rows, nullptr, consumed);
    }
    runtime_check(cudaDeviceSynchronize(), "warm up gather");
    std::array<float, Trials> samples{};
    for (float& us : samples) {
        runtime_check(cudaEventRecord(start), "record start");
        gather<Stages, false><<<blocks, 32>>>(inputs.descriptor, inputs.segment_rows, nullptr, consumed);
        runtime_check(cudaEventRecord(end), "record end");
        runtime_check(cudaEventSynchronize(end), "wait for gather");
        runtime_check(cudaEventElapsedTime(&us, start, end), "measure elapsed time");
        us *= 1000.0f;
    }
    runtime_check(cudaEventDestroy(start), "destroy start event");
    runtime_check(cudaEventDestroy(end), "destroy end event");
    runtime_check(cudaFree(consumed), "free consumer output");
    std::sort(samples.begin(), samples.end());
    return samples[Trials / 2];
}

template <int Stages>
void validate(const GatherMap& inputs) {
    unsigned* errors = nullptr;
    unsigned char* consumed = nullptr;
    runtime_check(cudaMalloc(&errors, sizeof(unsigned)), "allocate error counter");
    runtime_check(cudaMalloc(&consumed, 4), "allocate validation output");
    runtime_check(cudaMemset(errors, 0, sizeof(unsigned)), "clear error counter");
    gather<Stages, true><<<4, 32>>>(inputs.descriptor, inputs.segment_rows, errors, consumed);
    unsigned count = 0;
    runtime_check(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost), "validate output");
    if (count) {
        std::fprintf(stderr, "%u byte mismatches in gathered activations\n", count);
        std::exit(1);
    }
    runtime_check(cudaFree(consumed), "free validation output");
    runtime_check(cudaFree(errors), "free error counter");
}

template <int Stages>
void run(const GatherMap& local, const GatherMap& fabric) {
    validate<Stages>(local);
    validate<Stages>(fabric);
    for (int blocks : {1, 16, 64, 128}) {
        const float local_us = measure<Stages>(local, blocks);
        const float fabric_us = measure<Stages>(fabric, blocks);
        std::printf("{\"stages\":%d,\"ctas\":%d,\"local_p50_us\":%.3f,"
                    "\"multi_peer_p50_us\":%.3f,\"bytewise_equal\":true}\n",
                    Stages, blocks, local_us, fabric_us);
    }
}

template <int Tiles, bool Raw>
float measure_prefix(const GatherMap& inputs, int blocks) {
    constexpr int Trials = 60;
    constexpr int Warmups = 6;
    unsigned char* consumed = nullptr;
    std::uint64_t* elapsed = nullptr;
    runtime_check(cudaMalloc(&consumed, blocks), "allocate prefix output");
    runtime_check(cudaMalloc(&elapsed, blocks * Trials * sizeof(std::uint64_t)), "allocate prefix timers");
    if constexpr (Raw) {
        unsigned* errors = nullptr;
        runtime_check(cudaMalloc(&errors, sizeof(unsigned)), "allocate raw validation counter");
        runtime_check(cudaMemset(errors, 0, sizeof(unsigned)), "clear raw validation counter");
        gather_raw<Tiles, true><<<blocks, 32>>>(
            reinterpret_cast<const unsigned char*>(inputs.base), inputs.segment_rows,
            errors, consumed, elapsed);
        unsigned count = 0;
        runtime_check(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost),
                      "validate raw gather");
        runtime_check(cudaFree(errors), "free raw validation counter");
        if (count) {
            std::fprintf(stderr, "%u byte mismatches in raw prefix gathers\n", count);
            std::exit(1);
        }
    }
    for (int trial = -Warmups; trial < Trials; ++trial) {
        auto* output_time = elapsed + std::max(0, trial) * blocks;
        if constexpr (Raw) {
            gather_raw<Tiles, false><<<blocks, 32>>>(
                reinterpret_cast<const unsigned char*>(inputs.base), inputs.segment_rows,
                nullptr, consumed, output_time);
        } else {
            gather<Tiles, false, Tiles><<<blocks, 32>>>(
                inputs.descriptor, inputs.segment_rows, nullptr, consumed, output_time);
        }
    }
    std::vector<std::uint64_t> timers(blocks * Trials);
    runtime_check(cudaMemcpy(timers.data(), elapsed, timers.size() * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToHost), "read prefix timers");
    runtime_check(cudaFree(elapsed), "free prefix timers");
    runtime_check(cudaFree(consumed), "free prefix output");
    std::array<std::uint64_t, Trials> samples{};
    for (int trial = 0; trial < Trials; ++trial) {
        const auto begin = timers.begin() + trial * blocks;
        samples[trial] = *std::max_element(begin, begin + blocks);
    }
    std::sort(samples.begin(), samples.end());
    return samples[Trials / 2] / 1000.0f;
}

template <int Tiles>
void run_prefix(const GatherMap& local, const GatherMap& fabric) {
    for (int blocks : {1, 16, 64, 128}) {
        const float local_us = measure_prefix<Tiles, false>(local, blocks);
        const float fabric_us = measure_prefix<Tiles, false>(fabric, blocks);
        const float raw_local_us = measure_prefix<Tiles, true>(local, blocks);
        const float raw_fabric_us = measure_prefix<Tiles, true>(fabric, blocks);
        std::printf("{\"prefix_tiles\":%d,\"ctas\":%d,\"local_p50_us\":%.3f,"
                    "\"multi_peer_p50_us\":%.3f,\"raw_local_p50_us\":%.3f,"
                    "\"raw_multi_peer_p50_us\":%.3f,\"bytewise_equal\":true}\n",
                    Tiles, blocks, local_us, fabric_us, raw_local_us, raw_fabric_us);
    }
}

}  // namespace

int main() {
    runtime_check(cudaSetDevice(0), "select target GPU");
    runtime_check(cudaFree(nullptr), "initialize CUDA runtime");
    int devices = 0;
    runtime_check(cudaGetDeviceCount(&devices), "count GPUs");
    if (devices < Peers) {
        std::fprintf(stderr, "this benchmark needs four visible GPUs\n");
        return 1;
    }
    for (int peer = 1; peer < Peers; ++peer) {
        int access = 0;
        runtime_check(cudaDeviceCanAccessPeer(&access, 0, peer), "check native peer access");
        if (!access) {
            std::fprintf(stderr, "GPU 0 cannot directly read GPU %d\n", peer);
            return 1;
        }
    }
    GatherMap local(false);
    GatherMap fabric(true);
    run<1>(local, fabric);
    run<4>(local, fabric);
    run<8>(local, fabric);
    run_prefix<1>(local, fabric);
    run_prefix<2>(local, fabric);
    run_prefix<4>(local, fabric);
}
