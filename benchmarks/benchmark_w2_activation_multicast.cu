#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr unsigned K = 128;
constexpr unsigned Stages = 16;
constexpr unsigned WeightBytes = 8192;
constexpr unsigned Trials = 72;

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

template <class T>
T* allocate(size_t count) {
    T* buffer = nullptr;
    check(cudaMalloc(&buffer, count * sizeof(T)), "allocate");
    return buffer;
}

__device__ uint64_t now() {
    uint64_t stamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stamp) :: "memory");
    return stamp;
}

__global__ void init_weights(uint32_t* weights, size_t words) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < words; index += gridDim.x * blockDim.x) {
        uint32_t value = uint32_t(index) ^ 0xa13f9705u;
        value = (value ^ (value >> 16)) * 0x7feb352du;
        value = (value ^ (value >> 15)) * 0x846ca68bu;
        weights[index] = value ^ (value >> 16);
    }
}

__host__ __device__ char activation_byte(size_t index) {
    return char((index * 0x9e3779b9ULL) >> 16);
}

__global__ void init_activation_copies(char* input, size_t bytes, unsigned copies) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < bytes * copies; index += gridDim.x * blockDim.x) {
        input[index] = activation_byte(index % bytes);
    }
}

template <bool Multicast, bool LoadWeights>
__global__ void copy_tiles(__grid_constant__ const CUtensorMap map, const char* weights,
                           char* output, uint64_t* elapsed, unsigned rows,
                           unsigned depth, unsigned trial) {
    auto cluster = cooperative_groups::this_cluster();
    const unsigned rank = cluster.block_rank();
    const unsigned source = (blockIdx.x / 4 * 2 + rank % 2) * rows * Stages;
    const unsigned activation_bytes = rows * K;
    const unsigned payload_bytes = activation_bytes + (LoadWeights ? WeightBytes : 0);
    extern __shared__ __align__(128) char tiles[];
    __shared__ uint64_t barriers[Stages];
    const unsigned destination = __cvta_generic_to_shared(tiles);
    const unsigned barrier_base = __cvta_generic_to_shared(barriers);
    if (threadIdx.x == 0) {
        for (unsigned k = 0; k < Stages; ++k) {
            const unsigned barrier = barrier_base + k * sizeof(uint64_t);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                         :: "r"(barrier), "r"(payload_bytes) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    cluster.sync();
    if (threadIdx.x == 0) {
        const uint64_t start = now();
        for (unsigned base = 0; base < Stages; base += depth) {
            for (unsigned k = base; k < base + depth; ++k) {
                const unsigned activation = destination + k * activation_bytes;
                const unsigned barrier = barrier_base + k * sizeof(uint64_t);
                const unsigned row = source + k * rows;
                if constexpr (Multicast) {
                    if (rank < 2) {
                        const uint16_t mask = 5u << rank;
                        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
                                     "mbarrier::complete_tx::bytes.multicast::cluster "
                                     "[%0], [%1, {0, %2}], [%3], %4;"
                                     :: "r"(activation), "l"(&map), "r"(row), "r"(barrier),
                                        "h"(mask) : "memory");
                    }
                } else {
                    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                                 "[%0], [%1, {0, %2}], [%3];"
                                 :: "r"(activation), "l"(&map), "r"(row), "r"(barrier) : "memory");
                }
                if constexpr (LoadWeights) {
                    const unsigned target = destination + Stages * activation_bytes + k * WeightBytes;
                    const size_t weight_tile = (size_t(trial % 16) * gridDim.x + blockIdx.x) * Stages + k;
                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                 "[%0], [%1], %2, [%3];" :: "r"(target),
                                 "l"(weights + weight_tile * WeightBytes), "r"(WeightBytes),
                                 "r"(barrier) : "memory");
                }
            }
            for (unsigned k = base; k < base + depth; ++k) {
                const unsigned barrier = barrier_base + k * sizeof(uint64_t);
                unsigned ready;
                do {
                    asm volatile("{ .reg .pred p; "
                                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], 0; "
                                 "selp.u32 %0, 1, 0, p; }"
                                 : "=r"(ready) : "r"(barrier) : "memory");
                } while (!ready);
            }
        }
        elapsed[trial * gridDim.x + blockIdx.x] = now() - start;
    }
    __syncthreads();
    if (trial == Trials - 1) {
        for (unsigned index = threadIdx.x; index < Stages * activation_bytes; index += blockDim.x) {
            output[blockIdx.x * Stages * activation_bytes + index] = tiles[index];
        }
    }
    cluster.sync();
}

template <bool Multicast, bool LoadWeights>
void run(unsigned clusters, unsigned rows, unsigned depth, int peer_device = -1) {
    const unsigned blocks = clusters * 4;
    const unsigned input_rows = clusters * 2 * rows * Stages;
    std::vector<char> expected(input_rows * K);
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = activation_byte(index);
    }
    int compute_device = 0;
    check(cudaGetDevice(&compute_device), "get compute device");
    if (peer_device >= 0) check(cudaSetDevice(peer_device), "set activation device");
    const unsigned copies = peer_device >= 0 ? Trials : 1;
    auto* input = allocate<char>(expected.size() * copies);
    if (peer_device >= 0) {
        init_activation_copies<<<1024, 256>>>(input, expected.size(), copies);
        check(cudaDeviceSynchronize(), "initialize peer activation copies");
        check(cudaSetDevice(compute_device), "restore compute device");
    } else {
        check(cudaMemcpy(input, expected.data(), expected.size(), cudaMemcpyHostToDevice), "init data");
    }
    auto* output = allocate<char>(expected.size() * 2);
    auto* elapsed = allocate<uint64_t>(Trials * blocks);
    char* weights = nullptr;
    if constexpr (LoadWeights) {
        const size_t bytes = size_t(16) * blocks * Stages * WeightBytes;
        weights = allocate<char>(bytes);
        init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights), bytes / sizeof(uint32_t));
        check(cudaGetLastError(), "init weights");
    }
    const cuuint64_t dimensions[]{K, input_rows};
    const cuuint64_t strides[]{K};
    const cuuint32_t box[]{K, rows};
    const cuuint32_t element_strides[]{1, 1};
    cudaLaunchAttribute cluster{};
    cluster.id = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim = {4, 1, 1};
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(blocks);
    launch.blockDim = dim3(64);
    launch.dynamicSmemBytes = Stages * (rows * K + (LoadWeights ? WeightBytes : 0));
    launch.attrs = &cluster;
    launch.numAttrs = 1;
    check(cudaFuncSetAttribute(copy_tiles<Multicast, LoadWeights>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               launch.dynamicSmemBytes), "configure scratch");
    for (unsigned trial = 0; trial < Trials; ++trial) {
        CUtensorMap map{};
        char* current = input + (trial % copies) * expected.size();
        if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, current, dimensions,
                                  strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
            std::fprintf(stderr, "TensorMap encoding failed\n");
            std::exit(1);
        }
        check(cudaLaunchKernelEx(&launch, copy_tiles<Multicast, LoadWeights>, map,
                                 weights, output, elapsed, rows, depth, trial), "launch");
    }
    check(cudaDeviceSynchronize(), "synchronize");
    std::vector<uint64_t> elapsed_host(Trials * blocks);
    std::vector<char> actual(expected.size() * 2);
    check(cudaMemcpy(actual.data(), output, actual.size(), cudaMemcpyDeviceToHost), "read output");
    check(cudaMemcpy(elapsed_host.data(), elapsed, elapsed_host.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost), "read timings");
    for (unsigned block = 0; block < blocks; ++block) {
        const unsigned count = rows * Stages * K;
        const auto source = expected.begin() + (block / 4 * 2 + block % 2) * count;
        if (!std::equal(source, source + count, actual.begin() + block * count)) {
            std::fprintf(stderr, "activation bytes differ in block %u\n", block);
            std::exit(1);
        }
    }
    std::vector<uint64_t> samples;
    for (unsigned trial = 8; trial < Trials; ++trial) {
        auto first = elapsed_host.begin() + trial * blocks;
        samples.push_back(*std::max_element(first, first + blocks));
    }
    std::sort(samples.begin(), samples.end());
    std::printf("{\"multicast\":%s,\"weights\":%s,\"peer_activation\":%s,\"clusters\":%u,\"rows_per_cta\":%u,"
                "\"depth\":%u,\"p50_us\":%.3f,\"p95_us\":%.3f,\"exact\":true}\n",
                Multicast ? "true" : "false", LoadWeights ? "true" : "false",
                peer_device >= 0 ? "true" : "false", clusters, rows,
                depth, samples[31] / 1000.0, samples[60] / 1000.0);
    check(cudaFree(output), "free output");
    check(cudaFree(elapsed), "free timings");
    if (weights != nullptr) check(cudaFree(weights), "free weights");
    if (peer_device >= 0) check(cudaSetDevice(peer_device), "set activation device for cleanup");
    check(cudaFree(input), "free input");
    if (peer_device >= 0) check(cudaSetDevice(compute_device), "restore compute device after cleanup");
}

struct WeightShape {
    unsigned experts;
    unsigned rows;
    unsigned k_tiles;
    unsigned bytes_per_tile;

    __host__ __device__ unsigned bytes_per_row() const { return bytes_per_tile / 128; }
    __host__ __device__ unsigned scale_period() const { return bytes_per_tile == 8192 ? 1 : 4; }
    unsigned weight_bytes() const { return experts * rows * k_tiles * bytes_per_row(); }
    unsigned scale_bytes() const { return experts * rows * k_tiles / scale_period() * sizeof(int32_t); }
};

struct WeightSample {
    uint64_t start;
    uint64_t stop;
    uint32_t first_word;
    uint32_t first_scale;
};

__device__ void wait_for_tile(unsigned barrier, unsigned parity) {
    unsigned ready;
    do {
        asm volatile("{ .reg .pred p; "
                     "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
                     "selp.u32 %0, 1, 0, p; }"
                     : "=r"(ready) : "r"(barrier), "r"(parity) : "memory");
    } while (!ready);
}

enum class WeightCache { EvictFirst, Normal, EvictLast };

template <bool TensorMap, bool ExpertInterleaved, unsigned ScaleGroup = 1, unsigned TileRows = 128,
          unsigned Producers = 1, bool PackedScales = false, unsigned PackedScaleK = 1,
          WeightCache Cache = WeightCache::EvictFirst>
__global__ void read_weights(__grid_constant__ const CUtensorMap weight_map,
                             __grid_constant__ const CUtensorMap scale_map,
                             const char* weight_ptr, const char* scale_ptr,
                             WeightShape shape, unsigned depth, WeightSample* samples) {
    constexpr unsigned MaxDepth = 24;
    constexpr unsigned WeightSlot = TileRows * 128;
    constexpr unsigned ScaleSlot = TileRows * 4;
    constexpr uint64_t CacheHint = Cache == WeightCache::Normal ? 0x1000000000000000ULL :
                                   Cache == WeightCache::EvictLast ? 0x14f0000000000000ULL :
                                                                    0x12f0000000000000ULL;
    static_assert(ScaleGroup == 1 || TensorMap);
    static_assert(Producers == 1 || ScaleGroup == 1);
    static_assert(!PackedScales || (TensorMap && TileRows == 128 && Producers == 1 && ScaleGroup == 1));
    static_assert(PackedScaleK == 1 || (PackedScales && PackedScaleK == 4));
    const unsigned weight_bytes = TileRows * shape.bytes_per_row();
    extern __shared__ __align__(128) char scratch[];
    __shared__ uint64_t complete[MaxDepth];
    if constexpr (Producers == 1 && !PackedScales) {
        if (threadIdx.x != 0) return;
    }
    const unsigned unaligned = __cvta_generic_to_shared(scratch);
    const unsigned alignment = (0u - unaligned) & 1023u;
    const unsigned base_smem = unaligned + alignment;
    const unsigned base_barrier = __cvta_generic_to_shared(complete);
    if (threadIdx.x == 0) {
        for (unsigned slot = 0; slot < depth; ++slot) {
            const unsigned bar = base_barrier + slot * sizeof(uint64_t);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if constexpr (Producers > 1 || PackedScales) __syncthreads();
    // Each CTA walks its output tiles. Producer warps own disjoint K slots.
    const unsigned n_tiles = shape.rows / TileRows;
    const uint64_t start = now();
    const unsigned producer = threadIdx.x / 32;
    unsigned iteration = producer;
    if (threadIdx.x % 32 == 0 && producer < Producers) {
        for (unsigned job = blockIdx.x; job < shape.experts * n_tiles; job += gridDim.x) {
            const unsigned expert = ExpertInterleaved ? job % shape.experts : job / n_tiles;
            const unsigned row = TileRows * (ExpertInterleaved ? job / shape.experts : job % n_tiles);
            for (unsigned k = producer; k < shape.k_tiles; k += Producers, iteration += Producers) {
                const unsigned slot = iteration % depth;
                const unsigned parity = (iteration / depth) % 2;
                const unsigned bar = base_barrier + slot * sizeof(uint64_t);
                if (iteration >= depth) wait_for_tile(bar, parity ^ 1);
                const unsigned scale_period = shape.scale_period();
                const bool scale = !PackedScales && k % (scale_period * ScaleGroup) == 0;
                const unsigned bytes = weight_bytes + (scale ? ScaleSlot * ScaleGroup : 0);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                             :: "r"(bar), "r"(bytes) : "memory");
                const unsigned weight = base_smem + slot * WeightSlot;
                const unsigned scale_dest = base_smem + depth * WeightSlot + slot * ScaleSlot;
                const unsigned scale_row = expert * (shape.k_tiles / scale_period) + k / scale_period;
                if constexpr (TensorMap) {
                    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                                 "[%0], [%1, {0, %2, %3, %4}], [%5], %6;"
                                 :: "r"(weight), "l"(&weight_map), "r"(row), "r"(k), "r"(expert),
                                    "r"(bar), "l"(CacheHint) : "memory");
                    if (scale) {
                        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                                     "[%0], [%1, {%2, %3}], [%4], %5;"
                                     :: "r"(scale_dest), "l"(&scale_map), "r"(row), "r"(scale_row),
                                        "r"(bar), "l"(CacheHint) : "memory");
                    }
                } else {
                    const size_t offset = ((expert * shape.k_tiles + k) * shape.rows + row) * shape.bytes_per_row();
                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                                 "[%0], [%1], %2, [%3], %4;"
                                 :: "r"(weight), "l"(weight_ptr + offset), "r"(weight_bytes),
                                    "r"(bar), "l"(CacheHint) : "memory");
                    if (scale) {
                        const size_t offset = (scale_row * shape.rows + row) * sizeof(int32_t);
                        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                                     "[%0], [%1], %2, [%3], %4;"
                                     :: "r"(scale_dest), "l"(scale_ptr + offset), "r"(ScaleSlot),
                                        "r"(bar), "l"(CacheHint) : "memory");
                    }
                }
            }
        }
        const unsigned rounds = iteration - producer;
        const unsigned drain = rounds > depth ? rounds - depth : 0;
        for (unsigned k = drain + producer; k < iteration; k += Producers) {
            wait_for_tile(base_barrier + (k % depth) * sizeof(uint64_t), (k / depth) % 2);
        }
    }
    if constexpr (PackedScales) {
        if (producer == 1) {
            // The scale warp runs independently. This measures memory and
            // unpack issue costs without adding its wait to the weight loader.
            const unsigned lane = threadIdx.x % 32;
            const auto* packed = reinterpret_cast<const uint32_t*>(scale_ptr);
            unsigned scale_iteration = 0;
            constexpr uint32_t Mask = 0x03030303u;
            constexpr uint32_t Base = 0x78787878u;
            for (unsigned job = blockIdx.x; job < shape.experts * n_tiles; job += gridDim.x) {
                const unsigned expert = ExpertInterleaved ? job % shape.experts : job / n_tiles;
                const unsigned row = TileRows * (ExpertInterleaved ? job / shape.experts : job % n_tiles);
                for (unsigned k = 0; k < shape.k_tiles; k += PackedScaleK, scale_iteration += PackedScaleK) {
                    uint4 codes{};
                    if constexpr (PackedScaleK == 1) {
                        const size_t index = ((expert * shape.k_tiles + k) * shape.rows + row) / 4 + lane;
                        codes.x = packed[index];
                    } else {
                        const size_t tile = (expert * n_tiles + row / TileRows) * (shape.k_tiles / 4) + k / 4;
                        codes = reinterpret_cast<const uint4*>(packed)[tile * 32 + lane];
                    }
                    const uint32_t words[]{codes.x, codes.y, codes.z, codes.w};
                    #pragma unroll
                    for (unsigned tile = 0; tile < PackedScaleK; ++tile) {
                        const uint32_t bits = words[tile];
                        const unsigned dst = base_smem + depth * WeightSlot +
                            ((scale_iteration + tile) % depth) * ScaleSlot + lane * sizeof(uint4);
                        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(dst),
                                     "r"(Base + (bits & Mask)), "r"(Base + ((bits >> 2) & Mask)),
                                     "r"(Base + ((bits >> 4) & Mask)), "r"(Base + ((bits >> 6) & Mask)) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
            }
        }
    }
    if constexpr (Producers > 1 || PackedScales) __syncthreads();
    if (threadIdx.x != 0) return;
    const uint64_t stop = now();
    const unsigned last_slot = (iteration - 1) % depth;
    const unsigned scale_slot = ((iteration - 1) / shape.scale_period() * shape.scale_period()) % depth;
    const uint32_t first = reinterpret_cast<uint32_t*>(scratch + alignment + last_slot * WeightSlot)[0];
    const uint32_t first_scale = reinterpret_cast<uint32_t*>(
        scratch + alignment + depth * WeightSlot + scale_slot * ScaleSlot)[0];
    samples[blockIdx.x] = {start, stop, first, first_scale};
}

CUtensorMapL2promotion l2_promotion(unsigned bytes) {
    switch (bytes) {
        case 0: return CU_TENSOR_MAP_L2_PROMOTION_NONE;
        case 64: return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
        case 128: return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
        case 256: return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
        default: std::abort();
    }
}

CUtensorMap weight_descriptor(char* ptr, const WeightShape& shape, unsigned promotion, unsigned tile_rows = 128) {
    CUtensorMap map{};
    const size_t row_bytes = shape.bytes_per_row();
    const cuuint64_t dims[]{128, shape.rows, shape.k_tiles, shape.experts};
    const cuuint64_t strides[]{row_bytes, row_bytes * shape.rows, row_bytes * shape.rows * shape.k_tiles};
    const cuuint32_t box[]{128, tile_rows, 1, 1};
    const cuuint32_t steps[]{1, 1, 1, 1};
    const CUtensorMapDataType dtype = shape.bytes_per_tile == 8192 ?
        CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if (cuTensorMapEncodeTiled(&map, dtype, 4, ptr, dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, l2_promotion(promotion),
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        std::fprintf(stderr, "weight TensorMap encoding failed\n");
        std::exit(1);
    }
    return map;
}

CUtensorMap scale_descriptor(char* ptr, const WeightShape& shape, unsigned promotion,
                             unsigned scale_group = 1, unsigned tile_rows = 128) {
    CUtensorMap map{};
    const cuuint64_t dims[]{shape.rows, shape.experts * shape.k_tiles / shape.scale_period()};
    const cuuint64_t strides[]{shape.rows * sizeof(int32_t)};
    const cuuint32_t box[]{tile_rows, scale_group};
    const cuuint32_t steps[]{1, 1};
    if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, ptr, dims, strides, box,
                              steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                              l2_promotion(promotion), CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        std::fprintf(stderr, "weight scale TensorMap encoding failed\n");
        std::exit(1);
    }
    return map;
}

template <bool ExpertInterleaved, unsigned TileRows = 128>
void check_last_weight_tiles(const WeightShape& shape, unsigned ctas, const char* weight_ptr, const char* scale_ptr,
                             const WeightSample* output) {
    const unsigned n_tiles = shape.rows / TileRows;
    for (unsigned cta = 0; cta < ctas; ++cta) {
        const unsigned job = cta + (shape.experts * n_tiles - 1 - cta) / ctas * ctas;
        const unsigned expert = ExpertInterleaved ? job % shape.experts : job / n_tiles;
        const unsigned row = TileRows * (ExpertInterleaved ? job / shape.experts : job % n_tiles);
        const size_t offset = ((expert * shape.k_tiles + shape.k_tiles - 1) * shape.rows + row) * shape.bytes_per_row();
        uint32_t value;
        check(cudaMemcpy(&value, weight_ptr + offset, sizeof(value), cudaMemcpyDeviceToHost), "read reference word");
        if (output[cta].first_word != value) {
            std::fprintf(stderr, "weight payload mismatch at CTA %u: got %08x expected %08x\n",
                         cta, output[cta].first_word, value);
            std::exit(1);
        }
        const size_t scale_row = (expert + 1) * (shape.k_tiles / shape.scale_period()) - 1;
        const size_t scale_offset = (scale_row * shape.rows + row) * sizeof(uint32_t);
        check(cudaMemcpy(&value, scale_ptr + scale_offset, sizeof(value), cudaMemcpyDeviceToHost),
              "read reference scale");
        if (output[cta].first_scale != value) {
            std::fprintf(stderr, "scale payload mismatch at CTA %u: got %08x expected %08x\n",
                         cta, output[cta].first_scale, value);
            std::exit(1);
        }
    }
}

template <bool TensorMap, bool ExpertInterleaved, unsigned ScaleGroup = 1, unsigned TileRows = 128,
          unsigned Producers = 1, bool PackedScales = false, unsigned PackedScaleK = 1,
          WeightCache Cache = WeightCache::EvictFirst>
void sample_weights(const WeightShape& shape, unsigned ctas, unsigned depth, unsigned weight_bytes,
                    unsigned scale_bytes, char* weights, char* scales, unsigned copies,
                    unsigned promotion = 256, const char* compressed_scales = nullptr) {
    const unsigned smem = TileRows * (128 + 4) * depth + 1024;
    check(cudaFuncSetAttribute(read_weights<TensorMap, ExpertInterleaved, ScaleGroup, TileRows,
                                            Producers, PackedScales, PackedScaleK, Cache>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "configure scratch");
    WeightSample* device = allocate<WeightSample>(Trials * ctas);
    for (unsigned trial = 0; trial < Trials; ++trial) {
        char* weight_ptr = weights + (trial % copies) * size_t(weight_bytes);
        char* scale_ptr = scales + (trial % copies) * size_t(scale_bytes);
        const char* scale_input = PackedScales ?
            compressed_scales + (trial % copies) * size_t(scale_bytes / 4) : scale_ptr;
        read_weights<TensorMap, ExpertInterleaved, ScaleGroup, TileRows, Producers,
                     PackedScales, PackedScaleK, Cache><<<ctas, 128, smem>>>(
            weight_descriptor(weight_ptr, shape, promotion, TileRows),
            scale_descriptor(scale_ptr, shape, promotion, ScaleGroup, TileRows),
            weight_ptr, scale_input, shape, depth, device + trial * ctas);
        check(cudaGetLastError(), "launch weight pass");
    }
    std::vector<WeightSample> output(Trials * ctas);
    check(cudaMemcpy(output.data(), device, output.size() * sizeof(WeightSample), cudaMemcpyDeviceToHost), "read samples");
    check_last_weight_tiles<ExpertInterleaved, TileRows>(
        shape, ctas, weights + ((Trials - 1) % copies) * size_t(weight_bytes),
        scales + ((Trials - 1) % copies) * size_t(scale_bytes), output.data() + (Trials - 1) * ctas);
    std::vector<uint64_t> times;
    for (unsigned trial = 8; trial < Trials; ++trial) {
        uint64_t first = UINT64_MAX;
        uint64_t last = 0;
        for (unsigned cta = 0; cta < ctas; ++cta) {
            const WeightSample sample = output[trial * ctas + cta];
            first = std::min(first, sample.start);
            last = std::max(last, sample.stop);
        }
        times.push_back(last - first);
    }
    std::sort(times.begin(), times.end());
    const unsigned bytes = weight_bytes + scale_bytes;
    const char* policy = Cache == WeightCache::Normal ? "normal" :
                         Cache == WeightCache::EvictLast ? "evict_last" : "evict_first";
    std::printf("{\"experts\":%u,\"weight_bits\":%u,\"n\":%u,\"k\":%u,\"copies\":%u,"
                "\"ctas\":%u,\"depth\":%u,\"tensormap\":%s,\"expert_interleaved\":%s,\"promotion_bytes\":%u,"
                "\"scale_group\":%u,\"tile_rows\":%u,\"producers\":%u,\"bytes\":%u,"
                "\"packed_scales\":%s,\"scale_unpack_k\":%u,\"input_bytes\":%u,\"cache_policy\":\"%s\","
                "\"p50_us\":%.3f,\"p95_us\":%.3f,\"logical_tb_per_s\":%.4f}\n",
                shape.experts, shape.bytes_per_tile == 8192 ? 4 : 8, shape.rows, shape.k_tiles * 128,
                copies, ctas, depth, TensorMap ? "true" : "false", ExpertInterleaved ? "true" : "false", promotion,
                ScaleGroup, TileRows, Producers, bytes, PackedScales ? "true" : "false",
                PackedScales ? PackedScaleK : 0,
                weight_bytes + scale_bytes / (PackedScales ? 4 : 1), policy,
                times[31] / 1000.0, times[60] / 1000.0,
                bytes / (1000.0 * times[31]));
    check(cudaFree(device), "free samples");
}

void measure_weight_residency(const WeightShape& shape) {
    constexpr unsigned MaxCopies = 8;
    const unsigned weight_bytes = shape.weight_bytes();
    const unsigned scale_bytes = shape.scale_bytes();
    const unsigned ctas = std::min(128u, shape.experts * shape.rows / 128u);
    auto* weights = allocate<char>(MaxCopies * size_t(weight_bytes));
    auto* scales = allocate<char>(MaxCopies * size_t(scale_bytes));
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights), MaxCopies * size_t(weight_bytes) / 4);
    check(cudaGetLastError(), "initialize weight residency payload");
    init_weights<<<128, 256>>>(reinterpret_cast<uint32_t*>(scales), MaxCopies * size_t(scale_bytes) / 4);
    check(cudaGetLastError(), "initialize weight residency scales");
    // Match the descriptors and occupancy while varying only the number of
    // disjoint allocations read between visits to one weight matrix.
    for (const unsigned copies : {1u, 2u, 4u, MaxCopies}) {
        sample_weights<true, false>(shape, ctas, 8, weight_bytes, scale_bytes, weights, scales, copies);
        sample_weights<true, false, 1, 128, 1, false, 1, WeightCache::Normal>(
            shape, ctas, 8, weight_bytes, scale_bytes, weights, scales, copies);
        sample_weights<true, false, 1, 128, 1, false, 1, WeightCache::EvictLast>(
            shape, ctas, 8, weight_bytes, scale_bytes, weights, scales, copies);
    }
    check(cudaFree(scales), "free residency scales");
    check(cudaFree(weights), "free residency weights");
}

enum class WeightSweep { Full, Promotion, ScaleGroup, SharedWidth, Producers };

__global__ void expand_scale_exponents(const uint32_t* packed, uint4* expanded, size_t groups) {
    constexpr uint32_t Mask = 0x03030303u;
    constexpr uint32_t Base = 0x78787878u;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < groups; index += gridDim.x * blockDim.x) {
        const uint32_t bits = packed[index];
        expanded[index] = make_uint4(Base + (bits & Mask), Base + ((bits >> 2) & Mask),
                                     Base + ((bits >> 4) & Mask), Base + ((bits >> 6) & Mask));
    }
}

__global__ void group_packed_scale_tiles(const uint32_t* packed, uint4* grouped,
                                         WeightShape shape, size_t count) {
    const unsigned n_tiles = shape.rows / 128;
    const unsigned k_groups = shape.k_tiles / 4;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += gridDim.x * blockDim.x) {
        const unsigned lane = index % 32;
        const unsigned group = index / 32 % k_groups;
        const unsigned n = index / 32 / k_groups % n_tiles;
        const unsigned expert = index / 32 / k_groups / n_tiles;
        const size_t offset = (size_t(expert) * shape.k_tiles + group * 4) * (shape.rows / 4) + n * 32 + lane;
        const unsigned stride = shape.rows / 4;
        grouped[index] = make_uint4(packed[offset], packed[offset + stride],
                                   packed[offset + 2 * stride], packed[offset + 3 * stride]);
    }
}

void run_packed_scale_warp(const WeightShape& shape) {
    const unsigned weight_bytes = shape.weight_bytes(), scale_bytes = shape.scale_bytes();
    const unsigned copies = std::max(4u, 512u * 1024 * 1024 / (weight_bytes + scale_bytes));
    const size_t scale_groups = copies * size_t(scale_bytes) / sizeof(uint4);
    auto* weights = allocate<char>(copies * size_t(weight_bytes));
    auto* scales = allocate<uint4>(scale_groups);
    auto* packed = allocate<uint32_t>(scale_groups);
    auto* grouped = allocate<uint4>(scale_groups / 4);
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights), copies * size_t(weight_bytes) / 4);
    init_weights<<<1024, 256>>>(packed, scale_groups);
    expand_scale_exponents<<<1024, 256>>>(packed, scales, scale_groups);
    group_packed_scale_tiles<<<1024, 256>>>(packed, grouped, shape, scale_groups / 4);
    check(cudaGetLastError(), "initialize paired scale encodings");
    for (unsigned depth : {8, 10, 12}) {
        sample_weights<true, false>(shape, 128, depth, weight_bytes, scale_bytes, weights,
                                    reinterpret_cast<char*>(scales), copies);
        sample_weights<true, false, 1, 128, 1, true>(shape, 128, depth, weight_bytes, scale_bytes, weights,
            reinterpret_cast<char*>(scales), copies, 256, reinterpret_cast<char*>(packed));
        sample_weights<true, false, 1, 128, 1, true, 4>(shape, 128, depth, weight_bytes, scale_bytes, weights,
            reinterpret_cast<char*>(scales), copies, 256, reinterpret_cast<char*>(grouped));
    }
    check(cudaFree(grouped), "free grouped scale sectors");
    check(cudaFree(packed), "free packed scale sectors");
    check(cudaFree(scales), "free full scales");
    check(cudaFree(weights), "free weight stream");
}

// Shared FP8 uses one packed scale vector per four K128 stages. Put each
// vector immediately after the group containing its first weight tile.
constexpr unsigned SharedWeightTile = 128 * 128;
constexpr unsigned SharedScaleTile = 128 * sizeof(uint32_t);

template <unsigned GroupK>
__global__ void pack_shared_weight_groups(__grid_constant__ const CUtensorMap weight_map,
                                          __grid_constant__ const CUtensorMap scale_map,
                                          unsigned groups_per_tile, char* destination) {
    extern __shared__ __align__(128) char scratch[];
    __shared__ uint64_t complete;
    const unsigned address = __cvta_generic_to_shared(scratch);
    const unsigned padding = (0u - address) & 1023u;
    const unsigned row_tile = blockIdx.x / groups_per_tile;
    const unsigned k = (blockIdx.x % groups_per_tile) * GroupK;
    const bool with_scale = k % 4 == 0;
    const unsigned bytes = GroupK * SharedWeightTile + (with_scale ? SharedScaleTile : 0);
    if (threadIdx.x == 0) {
        const unsigned bar = __cvta_generic_to_shared(&complete);
        const unsigned row = row_tile * 128;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(bar), "r"(bytes) : "memory");
        for (unsigned stage = 0; stage < GroupK; ++stage) {
            const unsigned dst = address + padding + stage * SharedWeightTile;
            asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1, {0, %2, %3, 0}], [%4];"
                         :: "r"(dst), "l"(&weight_map), "r"(row), "r"(k + stage), "r"(bar) : "memory");
        }
        if (with_scale) {
            const unsigned dst = address + padding + GroupK * SharedWeightTile;
            asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1, {%2, %3}], [%4];"
                         :: "r"(dst), "l"(&scale_map), "r"(row), "r"(k / 4), "r"(bar) : "memory");
        }
        wait_for_tile(bar, 0);
    }
    __syncthreads();
    const auto* source = reinterpret_cast<const uint4*>(scratch + padding);
    const size_t tile_base = row_tile * size_t(groups_per_tile * GroupK) * (SharedWeightTile + SharedScaleTile / 4);
    auto* output = reinterpret_cast<uint4*>(
        destination + tile_base + k * SharedWeightTile + ((k + 3) / 4) * SharedScaleTile);
    for (unsigned word = threadIdx.x; word < bytes / sizeof(uint4); word += blockDim.x) {
        output[word] = source[word];
    }
}

template <unsigned GroupK>
__global__ void read_packed_shared_weights(const char* packed, unsigned groups_per_tile,
                                          unsigned depth, WeightSample* samples) {
    constexpr unsigned SlotBytes = GroupK * SharedWeightTile + 1024;
    extern __shared__ __align__(128) char scratch[];
    __shared__ uint64_t complete[12 / GroupK];
    if (threadIdx.x != 0) return;
    const unsigned address = __cvta_generic_to_shared(scratch);
    const unsigned padding = (0u - address) & 1023u;
    const unsigned bars = __cvta_generic_to_shared(complete);
    const size_t tile_base = blockIdx.x * size_t(groups_per_tile * GroupK) *
        (SharedWeightTile + SharedScaleTile / 4);
    for (unsigned slot = 0; slot < depth; ++slot) {
        const unsigned bar = bars + slot * sizeof(uint64_t);
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    const uint64_t start = now();
    for (unsigned group = 0; group < groups_per_tile; ++group) {
        const unsigned slot = group % depth;
        const unsigned parity = (group / depth) % 2;
        const unsigned bar = bars + slot * sizeof(uint64_t);
        if (group >= depth) wait_for_tile(bar, parity ^ 1);
        const unsigned k = group * GroupK;
        const unsigned bytes = GroupK * SharedWeightTile + (k % 4 == 0 ? SharedScaleTile : 0);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(bar), "r"(bytes) : "memory");
        const unsigned dst = address + padding + slot * SlotBytes;
        const char* src = packed + tile_base + k * SharedWeightTile + ((k + 3) / 4) * SharedScaleTile;
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                     "[%0], [%1], %2, [%3], %4;"
                     :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(0x12f0000000000000ULL) : "memory");
    }
    for (unsigned group = groups_per_tile - depth; group < groups_per_tile; ++group) {
        wait_for_tile(bars + (group % depth) * sizeof(uint64_t), (group / depth) % 2);
    }
    const uint64_t stop = now();
    const char* tail = scratch + padding + ((groups_per_tile - 1) % depth) * SlotBytes;
    const char* scale = scratch + padding + ((groups_per_tile - 4 / GroupK) % depth) * SlotBytes;
    samples[blockIdx.x] = {start, stop, *reinterpret_cast<const uint32_t*>(tail + (GroupK - 1) * SharedWeightTile),
                           *reinterpret_cast<const uint32_t*>(scale + GroupK * SharedWeightTile)};
}

template <unsigned GroupK>
void run_packed_shared_weights(const WeightShape& shape) {
    const unsigned weight_bytes = shape.weight_bytes(), scale_bytes = shape.scale_bytes();
    const size_t bytes = size_t(weight_bytes) + scale_bytes;
    const unsigned copies = 512u * 1024 * 1024 / bytes;
    const unsigned ctas = shape.rows / 128, groups_per_tile = shape.k_tiles / GroupK;
    constexpr unsigned PackScratch = GroupK * SharedWeightTile + SharedScaleTile + 1024;
    char* weights = allocate<char>(copies * size_t(weight_bytes));
    char* scales = allocate<char>(copies * size_t(scale_bytes));
    char* packed = allocate<char>(copies * bytes);
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights), copies * size_t(weight_bytes) / 4);
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(scales), copies * size_t(scale_bytes) / 4);
    check(cudaFuncSetAttribute(pack_shared_weight_groups<GroupK>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               PackScratch), "configure packing scratch");
    for (unsigned copy = 0; copy < copies; ++copy) {
        const auto weight_map = weight_descriptor(weights + copy * size_t(weight_bytes), shape, 256);
        const auto scale_map = scale_descriptor(scales + copy * size_t(scale_bytes), shape, 256);
        pack_shared_weight_groups<GroupK><<<ctas * groups_per_tile, 128, PackScratch>>>(
            weight_map, scale_map, groups_per_tile, packed + copy * bytes);
    }
    check(cudaGetLastError(), "prepack shared weights");
    WeightSample* stamps = allocate<WeightSample>(Trials * ctas);
    std::vector<WeightSample> samples(Trials * ctas);
    for (unsigned stages : {4, 8, 10, 12}) {
        if (stages % GroupK) continue;
        const unsigned depth = stages / GroupK;
        sample_weights<true, false>(shape, ctas, stages, weight_bytes, scale_bytes, weights, scales, copies);
        const unsigned smem = depth * (GroupK * SharedWeightTile + 1024) + 1024;
        check(cudaFuncSetAttribute(read_packed_shared_weights<GroupK>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   smem), "configure grouped TMA scratch");
        for (unsigned trial = 0; trial < Trials; ++trial) {
            read_packed_shared_weights<GroupK><<<ctas, 128, smem>>>(
                packed + (trial % copies) * bytes, groups_per_tile, depth, stamps + trial * ctas);
        }
        check(cudaGetLastError(), "read prepacked shared weights");
        check(cudaMemcpy(samples.data(), stamps, samples.size() * sizeof(WeightSample), cudaMemcpyDeviceToHost),
              "collect prepacked timings");
        const size_t last = (Trials - 1) % copies;
        check_last_weight_tiles<false>(shape, ctas, weights + last * weight_bytes, scales + last * scale_bytes,
                                       samples.data() + (Trials - 1) * ctas);
        std::vector<uint64_t> times;
        for (unsigned trial = 8; trial < Trials; ++trial) {
            uint64_t first = UINT64_MAX, end = 0;
            for (unsigned cta = 0; cta < ctas; ++cta) {
                first = std::min(first, samples[trial * ctas + cta].start);
                end = std::max(end, samples[trial * ctas + cta].stop);
            }
            times.push_back(end - first);
        }
        std::sort(times.begin(), times.end());
        std::printf("{\"packed_shared\":true,\"n\":%u,\"k\":%u,\"ctas\":%u,\"depth\":%u,"
                    "\"group_k\":%u,\"copies\":%u,\"bytes\":%zu,\"p50_us\":%.3f,\"p95_us\":%.3f}\n",
                    shape.rows, shape.k_tiles * 128, ctas, stages, GroupK, copies, bytes,
                    times[31] / 1000.0, times[60] / 1000.0);
    }
    check(cudaFree(stamps), "free timestamps");
    check(cudaFree(packed), "free prepacked weights");
    check(cudaFree(scales), "free initial scales");
    check(cudaFree(weights), "free initial weights");
}

void run_weight_shape(const WeightShape& shape, WeightSweep sweep) {
    const unsigned weight_bytes = shape.weight_bytes();
    const unsigned scale_bytes = shape.scale_bytes();
    const unsigned copies = std::max(4u, 512u * 1024 * 1024 / (weight_bytes + scale_bytes));
    auto* weights = allocate<char>(copies * size_t(weight_bytes));
    auto* scales = allocate<char>(copies * size_t(scale_bytes));
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights), copies * size_t(weight_bytes) / sizeof(uint32_t));
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(scales), copies * size_t(scale_bytes) / sizeof(uint32_t));
    check(cudaGetLastError(), "init weight stream");
    if (sweep == WeightSweep::Producers) {
        for (unsigned depth : {8, 12}) {
            sample_weights<true, false, 1, 128, 1>(shape, 128, depth, weight_bytes, scale_bytes, weights, scales, copies);
            sample_weights<true, false, 1, 128, 2>(shape, 128, depth, weight_bytes, scale_bytes, weights, scales, copies);
        }
        for (unsigned depth : {12, 24}) {
            sample_weights<true, false, 1, 64, 1>(shape, 128, depth, weight_bytes, scale_bytes, weights, scales, copies);
            sample_weights<true, false, 1, 64, 2>(shape, 128, depth, weight_bytes, scale_bytes, weights, scales, copies);
        }
    } else if (sweep == WeightSweep::SharedWidth) {
        for (unsigned depth : {4, 8, 12}) {
            sample_weights<true, false, 1, 128>(shape, shape.rows / 128, depth,
                                               weight_bytes, scale_bytes, weights, scales, copies);
            sample_weights<true, false, 1, 64>(shape, shape.rows / 64, depth,
                                              weight_bytes, scale_bytes, weights, scales, copies);
            sample_weights<true, false, 1, 32>(shape, std::min(152u, shape.rows / 32), depth,
                                              weight_bytes, scale_bytes, weights, scales, copies);
        }
    } else if (sweep == WeightSweep::ScaleGroup) {
        constexpr unsigned Ctas = 128;
        for (unsigned depth : {8, 12}) {
            sample_weights<true, false, 1>(shape, Ctas, depth, weight_bytes, scale_bytes, weights, scales, copies);
            sample_weights<true, false, 2>(shape, Ctas, depth, weight_bytes, scale_bytes, weights, scales, copies);
            sample_weights<true, false, 4>(shape, Ctas, depth, weight_bytes, scale_bytes, weights, scales, copies);
        }
    } else if (sweep == WeightSweep::Promotion) {
        const unsigned ctas = shape.experts == 1 ? shape.rows / 128 : 128;
        for (unsigned depth : {8, 12}) {
            for (unsigned promotion : {256, 0, 64, 128}) {
                sample_weights<true, false>(shape, ctas, depth, weight_bytes, scale_bytes,
                                            weights, scales, copies, promotion);
            }
        }
    } else {
        const auto cta_sizes = shape.experts == 1 ? std::vector<unsigned>{shape.rows / 128} :
                                                   std::vector<unsigned>{96, 128, 144};
        for (unsigned ctas : cta_sizes) {
            for (unsigned depth : {4, 8, 12}) {
                sample_weights<false, false>(shape, ctas, depth, weight_bytes, scale_bytes, weights, scales, copies);
                sample_weights<true, false>(shape, ctas, depth, weight_bytes, scale_bytes, weights, scales, copies);
                if (shape.experts > 1) {
                    sample_weights<true, true>(shape, ctas, depth, weight_bytes, scale_bytes, weights, scales, copies);
                }
            }
        }
    }
    check(cudaFree(weights), "free weight stream");
    check(cudaFree(scales), "free scale stream");
}

void run_split_weights(unsigned experts, unsigned w1_ctas, unsigned w2_ctas, unsigned depth, bool concurrent) {
    constexpr unsigned Copies = 4;
    const WeightShape shapes[]{{experts, 4096, 48, 8192}, {experts, 6144, 16, 8192}};
    const unsigned blocks[]{w1_ctas, w2_ctas};
    const unsigned smem = (128 * 128 + 128 * 4) * depth + 1024;
    cudaStream_t streams[2]{};
    cudaEvent_t start{}, done[2]{};
    char* weights[2]{};
    char* scales[2]{};
    WeightSample* stamps[2]{};
    CUtensorMap weight_maps[2][Copies], scale_maps[2][Copies];

    check(cudaFuncSetAttribute(read_weights<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               smem), "configure split scratch");
    check(cudaEventCreateWithFlags(&start, cudaEventDisableTiming), "create start event");
    for (unsigned phase = 0; phase < 2; ++phase) {
        const auto& shape = shapes[phase];
        const unsigned weight_bytes = shape.weight_bytes();
        const unsigned scale_bytes = shape.scale_bytes();
        weights[phase] = allocate<char>(Copies * size_t(weight_bytes));
        scales[phase] = allocate<char>(Copies * size_t(scale_bytes));
        stamps[phase] = allocate<WeightSample>(Trials * blocks[phase]);
        init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights[phase]), Copies * size_t(weight_bytes) / 4);
        init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(scales[phase]), Copies * size_t(scale_bytes) / 4);
        check(cudaGetLastError(), "initialize split weights");
        if (concurrent) {
            check(cudaStreamCreateWithFlags(&streams[phase], cudaStreamNonBlocking), "create weight stream");
            check(cudaEventCreateWithFlags(&done[phase], cudaEventDisableTiming), "create end event");
        }
        for (unsigned copy = 0; copy < Copies; ++copy) {
            weight_maps[phase][copy] = weight_descriptor(weights[phase] + copy * size_t(weight_bytes), shape, 256);
            scale_maps[phase][copy] = scale_descriptor(scales[phase] + copy * size_t(scale_bytes), shape, 256);
        }
    }
    for (unsigned trial = 0; trial < Trials; ++trial) {
        if (concurrent) check(cudaEventRecord(start), "record shared start");
        const unsigned copy = trial % Copies;
        for (unsigned phase = 0; phase < 2; ++phase) {
            if (concurrent) check(cudaStreamWaitEvent(streams[phase], start), "wait for shared start");
            read_weights<true, false><<<blocks[phase], 128, smem, streams[phase]>>>(
                weight_maps[phase][copy], scale_maps[phase][copy], nullptr, nullptr,
                shapes[phase], depth, stamps[phase] + trial * blocks[phase]);
            check(cudaGetLastError(), "launch split weight flow");
            if (concurrent) check(cudaEventRecord(done[phase], streams[phase]), "record split completion");
        }
        if (concurrent) {
            for (auto event : done) check(cudaStreamWaitEvent(nullptr, event), "join split streams");
        }
    }
    check(cudaDeviceSynchronize(), "wait for split weight flows");
    std::vector<WeightSample> output[2];
    for (unsigned phase = 0; phase < 2; ++phase) {
        output[phase].resize(Trials * blocks[phase]);
        check(cudaMemcpy(output[phase].data(), stamps[phase], output[phase].size() * sizeof(WeightSample),
                         cudaMemcpyDeviceToHost), "read split timestamps");
        check_last_weight_tiles<false>(shapes[phase], blocks[phase],
                                      weights[phase] + ((Trials - 1) % Copies) * size_t(shapes[phase].weight_bytes()),
                                      scales[phase] + ((Trials - 1) % Copies) * size_t(shapes[phase].scale_bytes()),
                                      output[phase].data() + (Trials - 1) * blocks[phase]);
        check(cudaFree(stamps[phase]), "free split timestamps");
        check(cudaFree(scales[phase]), "free split scales");
        check(cudaFree(weights[phase]), "free split weights");
        if (concurrent) {
            check(cudaEventDestroy(done[phase]), "destroy end event");
            check(cudaStreamDestroy(streams[phase]), "destroy weight stream");
        }
    }
    check(cudaEventDestroy(start), "destroy start event");
    std::vector<uint64_t> times, overlaps;
    for (unsigned trial = 8; trial < Trials; ++trial) {
        uint64_t first[]{UINT64_MAX, UINT64_MAX}, last[]{0, 0};
        for (unsigned phase = 0; phase < 2; ++phase) {
            for (unsigned cta = 0; cta < blocks[phase]; ++cta) {
                const auto sample = output[phase][trial * blocks[phase] + cta];
                first[phase] = std::min(first[phase], sample.start);
                last[phase] = std::max(last[phase], sample.stop);
            }
        }
        times.push_back(std::max(last[0], last[1]) - std::min(first[0], first[1]));
        const auto overlap_start = std::max(first[0], first[1]);
        const auto overlap_end = std::min(last[0], last[1]);
        overlaps.push_back(overlap_end > overlap_start ? overlap_end - overlap_start : 0);
    }
    std::sort(times.begin(), times.end());
    std::sort(overlaps.begin(), overlaps.end());
    const unsigned bytes = shapes[0].weight_bytes() + shapes[0].scale_bytes() +
                           shapes[1].weight_bytes() + shapes[1].scale_bytes();
    std::printf("{\"experts\":%u,\"w1_ctas\":%u,\"w2_ctas\":%u,\"depth\":%u,\"concurrent\":%s,"
                "\"bytes\":%u,\"p50_us\":%.3f,\"p95_us\":%.3f,\"overlap_p50_us\":%.3f,"
                "\"logical_tb_per_s\":%.4f}\n",
                experts, w1_ctas, w2_ctas, depth, concurrent ? "true" : "false", bytes,
                times[31] / 1000.0, times[60] / 1000.0, overlaps[31] / 1000.0, bytes / (1000.0 * times[31]));
}

__global__ void pull_peer_input(const uint4* source, uint4* destination, unsigned vectors, WeightSample* stamps) {
    const unsigned lane = threadIdx.x;
    const uint64_t start = now();
    for (unsigned i = blockIdx.x * blockDim.x + lane; i < vectors; i += gridDim.x * blockDim.x) {
        destination[i] = source[i];
    }
    __syncthreads();
    if (lane == 0) stamps[blockIdx.x] = {start, now(), 0, 0};
}

void run_peer_weight_contention(unsigned experts, unsigned dispatch_ctas, unsigned peer_bytes,
                                const uint4* remote, unsigned peer_stride, unsigned copies) {
    constexpr unsigned Depth = 8;
    constexpr unsigned WeightCtas = 128;
    constexpr unsigned Scratch = (128 * 128 + 128 * 4) * Depth + 1024;
    const WeightShape shape{experts, 4096, 48, 8192};
    const unsigned weight_bytes = shape.weight_bytes();
    const unsigned scale_bytes = shape.scale_bytes();
    auto* weights = allocate<char>(copies * size_t(weight_bytes));
    auto* scales = allocate<char>(copies * size_t(scale_bytes));
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights), copies * size_t(weight_bytes) / sizeof(uint32_t));
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(scales), copies * size_t(scale_bytes) / sizeof(uint32_t));
    check(cudaGetLastError(), "initialize contention weights");
    auto* local = allocate<uint4>(std::max(1u, peer_bytes / unsigned(sizeof(uint4))));
    auto* weight_stamps = allocate<WeightSample>(Trials * WeightCtas);
    auto* peer_stamps = allocate<WeightSample>(Trials * std::max(1u, dispatch_ctas));
    check(cudaFuncSetAttribute(read_weights<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               Scratch), "configure contention scratch");
    cudaStream_t weight_stream{}, peer_stream{};
    cudaEvent_t start{}, weight_done{}, peer_done{};
    check(cudaStreamCreateWithFlags(&weight_stream, cudaStreamNonBlocking), "create contention weight stream");
    check(cudaStreamCreateWithFlags(&peer_stream, cudaStreamNonBlocking), "create contention peer stream");
    for (auto* event : {&start, &weight_done, &peer_done}) {
        check(cudaEventCreateWithFlags(event, cudaEventDisableTiming), "create contention event");
    }
    for (unsigned trial = 0; trial < Trials; ++trial) {
        const unsigned copy = trial % copies;
        check(cudaEventRecord(start), "record contention start");
        check(cudaStreamWaitEvent(weight_stream, start), "order weight stream");
        auto* weight_ptr = weights + copy * size_t(weight_bytes);
        auto* scale_ptr = scales + copy * size_t(scale_bytes);
        if (dispatch_ctas) {
            check(cudaStreamWaitEvent(peer_stream, start), "order peer stream");
            // Give Dispatch the earlier launch, as in the decode path.
            pull_peer_input<<<dispatch_ctas, 512, 0, peer_stream>>>(
                remote + copy * size_t(peer_stride), local, peer_bytes / sizeof(uint4),
                peer_stamps + trial * dispatch_ctas);
            check(cudaGetLastError(), "launch peer pulls");
            check(cudaEventRecord(peer_done, peer_stream), "record peer completion");
        }
        read_weights<true, false><<<WeightCtas, 128, Scratch, weight_stream>>>(
            weight_descriptor(weight_ptr, shape, 256), scale_descriptor(scale_ptr, shape, 256),
            weight_ptr, scale_ptr, shape, Depth, weight_stamps + trial * WeightCtas);
        check(cudaGetLastError(), "launch contention weight reads");
        check(cudaEventRecord(weight_done, weight_stream), "record weight completion");
        check(cudaStreamWaitEvent(nullptr, weight_done), "join weight stream");
        if (dispatch_ctas) check(cudaStreamWaitEvent(nullptr, peer_done), "join peer stream");
    }
    check(cudaDeviceSynchronize(), "wait for contention work");
    std::vector<WeightSample> w(Trials * WeightCtas), p(Trials * dispatch_ctas);
    check(cudaMemcpy(w.data(), weight_stamps, w.size() * sizeof(WeightSample), cudaMemcpyDeviceToHost),
          "read contention weight stamps");
    const unsigned last_copy = (Trials - 1) % copies;
    check_last_weight_tiles<false>(shape, WeightCtas, weights + last_copy * size_t(weight_bytes),
                                  scales + last_copy * size_t(scale_bytes), w.data() + (Trials - 1) * WeightCtas);
    if (dispatch_ctas) {
        check(cudaMemcpy(p.data(), peer_stamps, p.size() * sizeof(WeightSample), cudaMemcpyDeviceToHost),
              "read contention peer stamps");
        std::vector<uint4> actual(peer_bytes / sizeof(uint4)), expected(actual.size());
        check(cudaMemcpy(actual.data(), local, peer_bytes, cudaMemcpyDeviceToHost), "read peer copy");
        check(cudaSetDevice(1), "select source for checking peer copy");
        check(cudaMemcpy(expected.data(), remote + last_copy * size_t(peer_stride), peer_bytes, cudaMemcpyDeviceToHost),
              "read peer source");
        if (std::memcmp(actual.data(), expected.data(), peer_bytes) != 0) {
            std::fprintf(stderr, "peer contention copy mismatch\n");
            std::exit(1);
        }
        check(cudaSetDevice(0), "restore local device after peer validation");
    }
    std::vector<uint64_t> times, peer_times, overlaps;
    for (unsigned trial = 8; trial < Trials; ++trial) {
        uint64_t w0 = UINT64_MAX, w1 = 0;
        uint64_t p0 = UINT64_MAX, p1 = 0;
        for (unsigned cta = 0; cta < WeightCtas; ++cta) {
            const auto sample = w[trial * WeightCtas + cta];
            w0 = std::min(w0, sample.start);
            w1 = std::max(w1, sample.stop);
        }
        for (unsigned cta = 0; cta < dispatch_ctas; ++cta) {
            const auto sample = p[trial * dispatch_ctas + cta];
            p0 = std::min(p0, sample.start);
            p1 = std::max(p1, sample.stop);
        }
        times.push_back(w1 - w0);
        peer_times.push_back(dispatch_ctas ? p1 - p0 : 0);
        const auto begin = std::max(w0, p0), end = std::min(w1, p1);
        overlaps.push_back(end > begin ? end - begin : 0);
    }
    for (auto* values : {&times, &peer_times, &overlaps}) std::sort(values->begin(), values->end());
    std::printf("{\"experts\":%u,\"weight_ctas\":%u,\"dispatch_ctas\":%u,\"peer_bytes\":%u,"
                "\"weight_p50_us\":%.3f,\"weight_p95_us\":%.3f,\"peer_p50_us\":%.3f,\"overlap_p50_us\":%.3f}\n",
                experts, WeightCtas, dispatch_ctas, peer_bytes, times[31] / 1000.0, times[60] / 1000.0,
                peer_times[31] / 1000.0, overlaps[31] / 1000.0);
    for (auto stream : {weight_stream, peer_stream}) check(cudaStreamDestroy(stream), "destroy contention stream");
    for (auto event : {start, weight_done, peer_done}) check(cudaEventDestroy(event), "destroy contention event");
    check(cudaFree(peer_stamps), "free peer stamps");
    check(cudaFree(weight_stamps), "free weight stamps");
    check(cudaFree(local), "free peer destination");
    check(cudaFree(scales), "free contention scales");
    check(cudaFree(weights), "free contention weights");
}

void run_peer_weight_cases() {
    constexpr unsigned Copies = 4;
    constexpr unsigned Stride = 64 * 1024 * 1024 / sizeof(uint4);
    int peer = 0;
    check(cudaDeviceCanAccessPeer(&peer, 0, 1), "check contention peer link");
    if (!peer) {
        std::fprintf(stderr, "contention benchmark needs two peer-accessible GPUs\n");
        std::exit(1);
    }
    check(cudaSetDevice(1), "select peer source GPU");
    auto* remote = allocate<uint4>(Copies * Stride);
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(remote), Copies * size_t(Stride) * 4);
    check(cudaDeviceSynchronize(), "initialize remote input");
    check(cudaSetDevice(0), "select local weight GPU");
    check(cudaDeviceEnablePeerAccess(1, 0), "enable remote reads");
    for (unsigned experts : {8, 16}) {
        run_peer_weight_contention(experts, 0, 0, remote, Stride, Copies);
        for (unsigned ctas : {12, 24}) {
            for (unsigned kib : {768, 1536, 3072}) {
                run_peer_weight_contention(experts, ctas, kib * 1024, remote, Stride, Copies);
            }
        }
    }
    check(cudaDeviceDisablePeerAccess(1), "disable remote reads");
    check(cudaSetDevice(1), "select peer source for cleanup");
    check(cudaFree(remote), "free peer source");
}
constexpr unsigned ActivationScaleBytes = 512;
constexpr unsigned RoutedWeightStageBytes = WeightBytes + 512;

__device__ void copy_stage_bytes(unsigned dst, const char* src, unsigned bytes, unsigned barrier) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(barrier) : "memory");
}

template <bool Packed>
__global__ void read_w2_activation_packet(__grid_constant__ const CUtensorMap acts,
                                          const char* scale_or_packet, const char* weights,
                                          char* output, uint64_t* latency, unsigned rows,
                                          unsigned depth, unsigned trial) {
    extern __shared__ __align__(128) char storage[];
    __shared__ uint64_t completion[Stages];
    const unsigned act_bytes = rows * K;
    const unsigned packet_bytes = act_bytes + ActivationScaleBytes;
    // Each activation destination starts at the same TMA swizzle alignment.
    const unsigned packet_pitch = (packet_bytes + 1023) & ~1023u;
    const unsigned base_address = __cvta_generic_to_shared(storage);
    const unsigned data = (base_address + 1023) & ~1023u;
    char* scratch = storage + data - base_address;
    const unsigned bars = __cvta_generic_to_shared(completion);
    const unsigned cta_tile = blockIdx.x * Stages;
    if (threadIdx.x == 0) {
        for (unsigned k = 0; k < Stages; ++k) {
            const unsigned barrier = bars + k * sizeof(uint64_t);
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                         :: "r"(barrier), "r"(packet_bytes + RoutedWeightStageBytes) : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    const uint64_t start = now();
    for (unsigned base = 0; base < Stages; base += depth) {
        if (threadIdx.x == 0 || threadIdx.x == 32) {
            for (unsigned k = base; k < base + depth; ++k) {
                const unsigned barrier = bars + k * sizeof(uint64_t);
                const unsigned tile = cta_tile + k;
                const unsigned dst = data + k * packet_pitch;
                if (threadIdx.x == 0) {
                    if constexpr (Packed) {
                        copy_stage_bytes(dst, scale_or_packet + tile * packet_bytes, packet_bytes, barrier);
                    } else {
                        asm volatile("cp.async.bulk.tensor.2d.shared::cta.global."
                                     "mbarrier::complete_tx::bytes [%0], [%1, {0, %2}], [%3];"
                                     :: "r"(dst), "l"(&acts), "r"(tile * rows), "r"(barrier) : "memory");
                        copy_stage_bytes(dst + act_bytes, scale_or_packet + tile * ActivationScaleBytes,
                                         ActivationScaleBytes, barrier);
                    }
                } else {
                    const size_t weight_tile = (trial % 16 * gridDim.x + blockIdx.x) * Stages + k;
                    const unsigned weight_dst = data + Stages * packet_pitch + k * RoutedWeightStageBytes;
                    copy_stage_bytes(weight_dst, weights + weight_tile * RoutedWeightStageBytes,
                                     RoutedWeightStageBytes, barrier);
                }
            }
            if (threadIdx.x == 0) {
                for (unsigned k = base; k < base + depth; ++k) {
                    wait_for_tile(bars + k * sizeof(uint64_t), 0);
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) latency[trial * gridDim.x + blockIdx.x] = now() - start;
    if (trial + 1 == Trials) {
        for (unsigned i = threadIdx.x; i < Stages * packet_bytes; i += blockDim.x) {
            output[cta_tile * packet_bytes + i] = scratch[i / packet_bytes * packet_pitch + i % packet_bytes];
        }
    }
}

template <bool Packed>
void measure_w2_activation_packets(unsigned blocks, unsigned rows, unsigned depth) {
    const size_t tiles = blocks * Stages;
    const unsigned act_bytes = rows * K;
    const unsigned packet_bytes = act_bytes + ActivationScaleBytes;
    const unsigned packet_pitch = (packet_bytes + 1023) & ~1023u;
    std::vector<char> activations(tiles * act_bytes), scales(tiles * ActivationScaleBytes);
    std::vector<char> expected(tiles * packet_bytes);
    for (size_t tile = 0; tile < tiles; ++tile) {
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned byte = 0; byte < K; ++byte) {
                const char value = activation_byte(tile * act_bytes + row * K + byte);
                activations[tile * act_bytes + row * K + byte] = value;
                // The 128B TMA swizzle XORs the 16B vector index with the row modulo eight.
                const unsigned swizzled = byte ^ ((row % 8) * 16);
                expected[tile * packet_bytes + row * K + swizzled] = value;
            }
        }
        for (unsigned byte = 0; byte < ActivationScaleBytes; ++byte) {
            const char value = activation_byte(tile * ActivationScaleBytes + byte + 0x4c012);
            scales[tile * ActivationScaleBytes + byte] = value;
            expected[tile * packet_bytes + act_bytes + byte] = value;
        }
    }
    auto* activation_device = allocate<char>(activations.size());
    const auto& payload = Packed ? expected : scales;
    auto* scale_or_packet = allocate<char>(payload.size());
    auto* output = allocate<char>(expected.size());
    auto* latency = allocate<uint64_t>(Trials * blocks);
    const size_t weight_bytes = 16 * tiles * RoutedWeightStageBytes;
    auto* weights = allocate<char>(weight_bytes);
    init_weights<<<1024, 256>>>(reinterpret_cast<uint32_t*>(weights), weight_bytes / 4);
    check(cudaGetLastError(), "initialize rotating weight payload");
    check(cudaMemcpy(activation_device, activations.data(), activations.size(), cudaMemcpyHostToDevice),
          "initialize activation plane");
    check(cudaMemcpy(scale_or_packet, payload.data(), payload.size(), cudaMemcpyHostToDevice),
          "initialize scale or packed activation plane");
    const cuuint64_t dimensions[]{K, tiles * rows};
    const cuuint64_t strides[]{K};
    const cuuint32_t box[]{K, rows}, element_strides[]{1, 1};
    CUtensorMap map{};
    if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, activation_device,
                              dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                              CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
        std::fprintf(stderr, "activation TensorMap encoding failed\n");
        std::exit(1);
    }
    const unsigned smem = Stages * (packet_pitch + RoutedWeightStageBytes) + 1023;
    check(cudaFuncSetAttribute(read_w2_activation_packet<Packed>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
          "allocate W2 stage storage");
    for (unsigned trial = 0; trial < Trials; ++trial) {
        read_w2_activation_packet<Packed><<<blocks, 64, smem>>>(
            map, scale_or_packet, weights, output, latency, rows, depth, trial);
        check(cudaGetLastError(), "launch W2 staging");
    }
    std::vector<char> actual(expected.size());
    std::vector<uint64_t> times(Trials * blocks);
    check(cudaMemcpy(actual.data(), output, actual.size(), cudaMemcpyDeviceToHost), "verify W2 staging");
    check(cudaMemcpy(times.data(), latency, times.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost),
          "read W2 staging timing");
    const auto difference = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (difference.first != actual.end()) {
        const size_t offset = difference.first - actual.begin();
        std::fprintf(stderr, "W2 stage mismatch packed=%d rows=%u tile=%zu offset=%zu actual=%u expected=%u\n",
                     Packed, rows, offset / packet_bytes, offset % packet_bytes,
                     static_cast<unsigned char>(*difference.first), static_cast<unsigned char>(*difference.second));
        std::exit(1);
    }
    std::vector<uint64_t> samples;
    for (unsigned trial = 8; trial < Trials; ++trial) {
        auto first = times.begin() + trial * blocks;
        samples.push_back(*std::max_element(first, first + blocks));
    }
    std::sort(samples.begin(), samples.end());
    std::printf("{\"packed\":%s,\"ctas\":%u,\"rows_per_cta\":%u,\"depth\":%u,"
                "\"scale_bytes\":%u,\"p50_us\":%.3f,\"p95_us\":%.3f,\"exact\":true}\n",
                Packed ? "true" : "false", blocks, rows, depth, ActivationScaleBytes,
                samples[31] / 1000.0, samples[60] / 1000.0);
    check(cudaFree(weights), "free W2 weight payload");
    check(cudaFree(latency), "free W2 timings");
    check(cudaFree(output), "free W2 stage result");
    check(cudaFree(scale_or_packet), "free W2 payload");
    check(cudaFree(activation_device), "free W2 activation plane");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "weight-residency") == 0) {
        int device = 0;
        check(cudaGetDevice(&device), "get residency device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "read device L2 capacity");
        std::printf("{\"l2_bytes\":%d,\"sms\":%d}\n", properties.l2CacheSize, properties.multiProcessorCount);
        for (const unsigned experts : {8u, 16u}) {
            measure_weight_residency({experts, 4096, 48, 8192});
            measure_weight_residency({experts, 6144, 16, 8192});
        }
        measure_weight_residency({1, 4096, 48, 16384});
        measure_weight_residency({1, 6144, 16, 16384});
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "w2-activation-packets") == 0) {
        for (unsigned blocks : {128, 152}) {
            for (unsigned rows : {8, 16}) {
                for (unsigned depth : {4, 8, 16}) {
                    measure_w2_activation_packets<false>(blocks, rows, depth);
                    measure_w2_activation_packets<true>(blocks, rows, depth);
                }
            }
        }
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "packed-scale-warp") == 0) {
        for (unsigned experts : {8, 16}) {
            run_packed_scale_warp({experts, 4096, 48, 8192});
            run_packed_scale_warp({experts, 6144, 16, 8192});
        }
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "packed-shared-weights") == 0) {
        for (const auto shape : {WeightShape{1, 4096, 48, 16384}, WeightShape{1, 6144, 16, 16384}}) {
            run_packed_shared_weights<2>(shape);
            run_packed_shared_weights<4>(shape);
        }
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "weight-producers") == 0) {
        for (unsigned experts : {8, 16}) {
            run_weight_shape({experts, 4096, 48, 8192}, WeightSweep::Producers);
            run_weight_shape({experts, 6144, 16, 8192}, WeightSweep::Producers);
        }
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "shared-width") == 0) {
        run_weight_shape({1, 4096, 48, 16384}, WeightSweep::SharedWidth);
        run_weight_shape({1, 6144, 16, 16384}, WeightSweep::SharedWidth);
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "peer-weights") == 0) {
        run_peer_weight_cases();
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "group-scales") == 0) {
        for (unsigned experts : {8, 16}) {
            run_weight_shape({experts, 4096, 48, 8192}, WeightSweep::ScaleGroup);
            run_weight_shape({experts, 6144, 16, 8192}, WeightSweep::ScaleGroup);
        }
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "split-weights") == 0) {
        for (unsigned experts : {8, 16}) {
            for (unsigned depth : {8, 12}) {
                run_split_weights(experts, 128, 128, depth, false);
                for (unsigned w1_ctas : {72, 80, 88, 96, 104, 112}) {
                    run_split_weights(experts, w1_ctas, 152 - w1_ctas, depth, true);
                }
            }
        }
        return 0;
    }
    const bool sweep_promotion = argc == 2 && std::strcmp(argv[1], "promotion") == 0;
    if (sweep_promotion || (argc == 2 && std::strcmp(argv[1], "weights") == 0)) {
        const auto sweep = sweep_promotion ? WeightSweep::Promotion : WeightSweep::Full;
        for (unsigned experts : {8, 16}) {
            run_weight_shape({experts, 4096, 48, 8192}, sweep);
            run_weight_shape({experts, 6144, 16, 8192}, sweep);
        }
        run_weight_shape({1, 4096, 48, 16384}, sweep);
        run_weight_shape({1, 6144, 16, 16384}, sweep);
        return 0;
    }
    const bool peer_activation = argc == 2 && std::strcmp(argv[1], "peer-activation") == 0;
    if (peer_activation) {
        int accessible = 0;
        check(cudaDeviceCanAccessPeer(&accessible, 0, 1), "query peer access");
        if (!accessible) {
            std::fprintf(stderr, "peer activation test needs two peer-accessible GPUs\n");
            return 1;
        }
        check(cudaSetDevice(0), "set compute device");
        check(cudaDeviceEnablePeerAccess(1, 0), "enable peer access in this context");
    }
    const int source = peer_activation ? 1 : -1;
    for (unsigned clusters : {8, 32}) {
        for (unsigned rows : {8, 16, 24}) {
            for (unsigned depth : {4, 8}) {
                run<false, false>(clusters, rows, depth, source);
                run<true, false>(clusters, rows, depth, source);
                run<false, true>(clusters, rows, depth, source);
                run<true, true>(clusters, rows, depth, source);
            }
        }
    }
    if (peer_activation) check(cudaDeviceDisablePeerAccess(1), "disable peer access in this context");
}
