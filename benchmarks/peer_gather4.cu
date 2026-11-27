// Measure peer activation loads of four noncontiguous rows into one SMEM tile.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int Rows = 128;
constexpr int Columns = 6144;
constexpr int Chunk = 128;
constexpr int Transfers = Columns / Chunk;
constexpr int StageBytes = 4 * Chunk;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA runtime: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

void check(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* description = nullptr;
        cuGetErrorString(status, &description);
        std::fprintf(stderr, "CUDA driver: %s\n", description);
        std::exit(1);
    }
}

__host__ __device__ unsigned coordinate(int block, int row) {
    return (block * 7 + row * (row + 2)) % Rows;
}

__device__ void wait_stage(unsigned barrier, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
            " p, [%1], %2; selp.u32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(barrier), "r"(phase) : "memory");
    } while (!ready);
}

template <bool Gather>
__device__ void issue(const CUtensorMap* map, const uint8_t* source,
                      unsigned chunk, unsigned dst, unsigned barrier) {
    const unsigned x = chunk * Chunk;
    unsigned rows[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        rows[i] = coordinate(blockIdx.x, i);
    }
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(barrier), "r"(StageBytes) : "memory");
    if constexpr (Gather) {
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
            ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
            :: "r"(dst), "l"(map), "r"(x), "r"(rows[0]), "r"(rows[1]),
               "r"(rows[2]), "r"(rows[3]), "r"(barrier) : "memory");
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t* row = source + rows[i] * Columns + x;
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %2, [%3];"
                :: "r"(dst + i * Chunk), "l"(row), "r"(Chunk), "r"(barrier) : "memory");
        }
    }
}

template <bool Gather, int Depth>
__global__ void load_activation(const __grid_constant__ CUtensorMap map,
                                const uint8_t* source, uint32_t* digest) {
    __shared__ __align__(128) uint8_t stages[Depth * StageBytes];
    __shared__ __align__(8) uint64_t barriers[Depth];
    if (threadIdx.x != 0) return;
    unsigned base = __cvta_generic_to_shared(stages);
    unsigned bar_base = __cvta_generic_to_shared(barriers);
    #pragma unroll
    for (int i = 0; i < Depth; ++i) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(bar_base + i * 8) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    #pragma unroll
    for (int i = 0; i < Depth; ++i) {
        issue<Gather>(&map, source, i, base + i * StageBytes, bar_base + i * 8);
    }
    uint32_t hash = 0;
    for (int chunk = 0; chunk < Transfers; ++chunk) {
        const int slot = chunk % Depth;
        const unsigned barrier = bar_base + slot * 8;
        wait_stage(barrier, (chunk / Depth) & 1);
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            const volatile uint8_t* input = stages + slot * StageBytes + row * Chunk;
            hash = hash * 33u + input[0] + 256u * input[Chunk - 1];
        }
        if (chunk + Depth < Transfers) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            issue<Gather>(&map, source, chunk + Depth, base + slot * StageBytes, barrier);
        }
    }
    digest[blockIdx.x] = hash;
}

template <bool Gather, int Depth>
float measure(CUtensorMap map, uint8_t* source, uint32_t* digest, int ctas,
              const std::vector<uint32_t>& expected) {
    cudaStream_t stream;
    cudaEvent_t before, after;
    cudaGraph_t graph;
    cudaGraphExec_t replay;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&before));
    check(cudaEventCreate(&after));
    load_activation<Gather, Depth><<<ctas, 32, 0, stream>>>(map, source, digest);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    std::vector<uint32_t> got(ctas);
    check(cudaMemcpy(got.data(), digest, ctas * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (got != expected) {
        std::fprintf(stderr, "Peer activation checksum differs, gather=%d, depth=%d\n", Gather, Depth);
        std::exit(1);
    }
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    check(cudaEventRecordWithFlags(before, stream, cudaEventRecordExternal));
    load_activation<Gather, Depth><<<ctas, 32, 0, stream>>>(map, source, digest);
    check(cudaEventRecordWithFlags(after, stream, cudaEventRecordExternal));
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiateWithFlags(&replay, graph, 0));
    std::vector<float> samples;
    for (int i = 0; i < 48; ++i) {
        check(cudaGraphLaunch(replay, stream));
        check(cudaStreamSynchronize(stream));
        float ms;
        check(cudaEventElapsedTime(&ms, before, after));
        if (i >= 8) samples.push_back(ms * 1000.0f);
    }
    std::sort(samples.begin(), samples.end());
    const float median = (samples[19] + samples[20]) * 0.5f;
    check(cudaGraphExecDestroy(replay));
    check(cudaGraphDestroy(graph));
    check(cudaEventDestroy(before));
    check(cudaEventDestroy(after));
    check(cudaStreamDestroy(stream));
    return median;
}
}  // namespace

int main(int argc, char** argv) {
    const int ctas = argc > 1 ? std::atoi(argv[1]) : 68;
    const int peer = argc > 2 ? std::atoi(argv[2]) : 1;
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    if (ctas < 1 || ctas > 152 || peer < 0 || peer >= devices) return 2;

    std::vector<uint8_t> input(Rows * Columns);
    uint32_t state = 12345;
    for (auto& value : input) {
        state = state * 1664525u + 1013904223u;
        value = state >> 24;
    }
    check(cudaSetDevice(peer));
    uint8_t* source = nullptr;
    check(cudaMalloc(&source, input.size()));
    check(cudaMemcpy(source, input.data(), input.size(), cudaMemcpyHostToDevice));
    check(cudaSetDevice(0));
    if (peer != 0) {
        int access;
        check(cudaDeviceCanAccessPeer(&access, 0, peer));
        if (!access) return 3;
        check(cudaDeviceEnablePeerAccess(peer, 0));
    }
    uint32_t* digest = nullptr;
    check(cudaMalloc(&digest, ctas * sizeof(uint32_t)));

    CUtensorMap map{};
    const cuuint64_t dims[] = {Columns, Rows};
    const cuuint64_t strides[] = {Columns};
    const cuuint32_t box[] = {Chunk, 1};
    const cuuint32_t elements[] = {1, 1};
    check(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                                 source, dims, strides, box, elements,
                                 CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    std::vector<uint32_t> expected(ctas, 0);
    for (int cta = 0; cta < ctas; ++cta) {
        for (int k = 0; k < Transfers; ++k) {
            for (int r = 0; r < 4; ++r) {
                const int pos = coordinate(cta, r) * Columns + k * Chunk;
                expected[cta] = expected[cta] * 33u + input[pos] + 256u * input[pos + Chunk - 1];
            }
        }
    }
    const float bulk4 = measure<false, 4>(map, source, digest, ctas, expected);
    const float gather4 = measure<true, 4>(map, source, digest, ctas, expected);
    const float gather8 = measure<true, 8>(map, source, digest, ctas, expected);
    const float bulk8 = measure<false, 8>(map, source, digest, ctas, expected);
    std::printf("COMPARE ctas=%d peer=%d bulk4_us=%.3f gather4_us=%.3f "
                "bulk8_us=%.3f gather8_us=%.3f\n", ctas, peer, bulk4, gather4, bulk8, gather8);
    check(cudaFree(digest));
    check(cudaSetDevice(peer));
    check(cudaFree(source));
}
