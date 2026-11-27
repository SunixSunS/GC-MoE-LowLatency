#include <cuda/atomic>
#include <cstddef>
#include <cstdint>

namespace {
constexpr unsigned FullWarp = 0xffffffffu;
constexpr unsigned QueueDone = 1u << 31;
constexpr unsigned RankedExperts = GCMOE_RANKS * GCMOE_LOCAL_EXPERTS;
constexpr unsigned RouteWidth = 8;
constexpr unsigned RowBytes = 6144;
constexpr unsigned ScaleWords = RowBytes / 128;
constexpr unsigned RowsPerExpert = GCMOE_BLOCK_M * GCMOE_EXPERT_SPAN;
constexpr unsigned Warps = GCMOE_THREADS / 32;

struct SymBuffer {
    int64_t base;
    int64_t offsets[72];
    uint32_t rank;
};
static_assert(sizeof(SymBuffer) == 592 && offsetof(SymBuffer, rank) == 584);
static_assert(GCMOE_RANKS <= 32 && GCMOE_BLOCK_M <= 128);
static_assert(GCMOE_CAPACITY >= 32 * RouteWidth);

template <class T>
__device__ T* at(char* base, unsigned offset) {
    return reinterpret_cast<T*>(base + offset);
}

__device__ char* peer_base(const SymBuffer& peers, unsigned rank) {
    return reinterpret_cast<char*>(peers.base + peers.offsets[rank]);
}

__device__ uint64_t acquire_packet(const uint64_t* address) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(value) : "l"(address) : "memory");
    return value;
}

__device__ void publish_packet(uint64_t* address, uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;"
                 :: "l"(address), "l"(value) : "memory");
}

__device__ void publish_row(unsigned* address, unsigned increment) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;"
                 :: "l"(address), "r"(increment) : "memory");
}

__device__ void trace_cta(char* local, unsigned field) {
    if constexpr (GCMOE_TRACE_OFFSET != 0) {
        auto* trace = at<uint64_t>(local, GCMOE_TRACE_OFFSET);
        trace += (blockIdx.x * GCMOE_TRACE_FIELDS + field) * 2;
        uint64_t now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now) :: "memory");
        trace[0] = now;
        trace[1] = 0;
    }
}

__device__ void trace_exit(char* local) {
    if constexpr (GCMOE_TRACE_OFFSET != 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            trace_cta(local, 3);
        }
    }
}

__device__ unsigned* spill_slot(char* dst, unsigned expert, unsigned rank, unsigned ordinal) {
    const unsigned index = (expert * GCMOE_RANKS + rank) * GCMOE_CAPACITY + ordinal;
    return at<unsigned>(dst, GCMOE_SOURCE_INDICES) + index;
}

__device__ void publish_routes(
    char* local, const SymBuffer& peers, unsigned rank, unsigned tokens,
    unsigned epoch, unsigned bank, unsigned* words) {
    const unsigned tid = threadIdx.x;
    const unsigned retired = bank ^ 1;
    if (tid == 0) {
        *at<unsigned>(local, GCMOE_ROLE3_EPOCH) = epoch;
    }
    if (tid < GCMOE_LOCAL_EXPERTS) {
        at<uint64_t>(local, GCMOE_TOTALS)[retired * GCMOE_LOCAL_EXPERTS + tid] = 0;
    }
    for (unsigned expert = tid; expert < RankedExperts; expert += blockDim.x) {
        words[expert] = 0;
    }
    __syncthreads();

    bool wrote_spill = false;
    auto* routes = at<const uint64_t>(local, GCMOE_TOPK);
    for (unsigned route = tid; route < tokens * RouteWidth; route += blockDim.x) {
        const uint64_t expert = routes[route];
        if (expert < RankedExperts) {
            const unsigned ordinal = atomicAdd(words + expert, 1u << 16) >> 16;
            if (ordinal < 2) {
                atomicOr(words + expert, route << (ordinal * 8));
            } else {
                char* dst = peer_base(peers, expert / GCMOE_LOCAL_EXPERTS);
                *spill_slot(dst, expert % GCMOE_LOCAL_EXPERTS, rank, ordinal) = route;
                wrote_spill = true;
            }
        }
    }
    if (wrote_spill) {
        __threadfence_system();
    }
    __syncthreads();

    // Publish one packet even for an empty span. The epoch stamps the count
    // and inline siblings together. Only spill entries use the old route table.
    for (unsigned expert = tid; expert < RankedExperts; expert += blockDim.x) {
        char* dst = peer_base(peers, expert / GCMOE_LOCAL_EXPERTS);
        const unsigned slot = (bank * GCMOE_RANKS + rank) * GCMOE_LOCAL_EXPERTS
                              + expert % GCMOE_LOCAL_EXPERTS;
        publish_packet(at<uint64_t>(dst, GCMOE_PACKETS) + slot,
                       static_cast<uint64_t>(epoch) << 32 | words[expert]);
    }
}

__device__ void start_pull(unsigned scratch, const char* source, unsigned bar) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar), "r"(RowBytes) : "memory");
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(scratch), "l"(source), "r"(RowBytes), "r"(bar) : "memory");
}

__device__ void finish_pull(char* target, unsigned scratch, unsigned bar, unsigned phase) {
    unsigned ready;
    do {
        asm volatile(
            "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
            "selp.u32 %0, 1, 0, p; }"
            : "=r"(ready) : "r"(bar), "r"(phase) : "memory");
    } while (!ready);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                 :: "l"(target), "r"(scratch), "r"(RowBytes) : "memory");
    asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ uint64_t wait_packet(char* local, unsigned bank, unsigned source,
                                unsigned expert, unsigned epoch) {
    const unsigned slot = (bank * GCMOE_RANKS + source) * GCMOE_LOCAL_EXPERTS + expert;
    const auto* address = at<uint64_t>(local, GCMOE_PACKETS) + slot;
    const auto started = clock64();
    uint64_t packet;
    do {
        packet = acquire_packet(address);
        if (clock64() - started > GCMOE_TIMEOUT_CLOCKS) {
            asm volatile("trap;");
        }
    } while (static_cast<unsigned>(packet >> 32) != epoch);
    return packet;
}

__device__ void publish_total(char* local, unsigned bank, unsigned expert, unsigned count) {
    if (count > RowsPerExpert) {
        asm volatile("trap;");
    }
    const uint64_t ready = static_cast<uint64_t>(gridDim.x * GCMOE_RANKS) << 32;
    publish_packet(at<uint64_t>(local, GCMOE_TOTALS) + bank * GCMOE_LOCAL_EXPERTS
                   + expert, ready | count);
    trace_cta(local, 2);
}

__device__ unsigned read_route(char* local, unsigned expert, unsigned source,
                               unsigned ordinal, unsigned packed) {
    return ordinal < 2 ? (packed >> (ordinal * 8)) & 255
                       : *spill_slot(local, expert, source, ordinal);
}

template <bool StoreMetadata>
__device__ void pull_payload(char* local, const SymBuffer& peers, unsigned source,
                             unsigned route, unsigned row, unsigned increment,
                             unsigned scratch, unsigned barrier, unsigned phase) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned token = route / RouteWidth;
    const unsigned tile = row / GCMOE_BLOCK_M;
    const unsigned tile_row = row % GCMOE_BLOCK_M;
    char* remote = peer_base(peers, source);
    if (lane == 0) {
        start_pull(scratch, remote + GCMOE_INPUT_X + token * RowBytes, barrier);
    }
    const unsigned sf_row = tile * 128 + (tile_row % 32) * 4 + tile_row / 32;
    for (unsigned col = lane; col < ScaleWords; col += 32) {
        at<unsigned>(local, GCMOE_L1_SF)[col * GCMOE_SF_ROWS + sf_row] =
            at<const unsigned>(remote, GCMOE_INPUT_SF)[token * ScaleWords + col];
    }
    __syncwarp();
    if (lane == 0) {
        at<unsigned>(local, GCMOE_L1_WEIGHT)[row] = at<const unsigned>(remote, GCMOE_SCORES)[route];
        if constexpr (StoreMetadata) {
            auto* metadata = at<unsigned>(local, GCMOE_METADATA) + row * 3;
            metadata[0] = source;
            metadata[1] = token;
            metadata[2] = route % RouteWidth;
        }
        finish_pull(local + GCMOE_L1_X + row * RowBytes, scratch, barrier, phase);
        publish_row(at<unsigned>(local, GCMOE_L1_FULL) + tile, increment);
    }
    __syncwarp();
}

__device__ void pull_row(char* local, const SymBuffer& peers, unsigned expert,
                         unsigned source, unsigned ordinal, unsigned packed, unsigned row,
                         unsigned increment, unsigned scratch, unsigned barrier, unsigned phase) {
    unsigned route = 0;
    if (threadIdx.x % 32 == 0) {
        route = read_route(local, expert, source, ordinal, packed);
    }
    route = __shfl_sync(FullWarp, route, 0);
    pull_payload<true>(local, peers, source, route, row, increment, scratch, barrier, phase);
}

__device__ void plan_expert(char* local, unsigned expert, unsigned bank, unsigned epoch) {
    const unsigned lane = threadIdx.x;
    unsigned packet = 0;
    if (lane < GCMOE_RANKS) {
        packet = static_cast<unsigned>(wait_packet(local, bank, lane, expert, epoch));
    }
    const unsigned count = packet >> 16;
    unsigned end = count;
    for (unsigned step = 1; step < 32; step *= 2) {
        const unsigned previous = __shfl_up_sync(FullWarp, end, step);
        if (lane >= step) {
            end += previous;
        }
    }
    const unsigned total = __shfl_sync(FullWarp, end, 31);
    if (total > RowsPerExpert) {
        asm volatile("trap;");
    }
    auto* rows = at<unsigned>(local, GCMOE_METADATA) + (expert * RowsPerExpert + end - count) * 3;
    for (unsigned ordinal = 0; ordinal < count; ++ordinal) {
        const unsigned route = read_route(local, expert, lane, ordinal, packet);
        rows[0] = lane;
        rows[1] = route / RouteWidth;
        rows[2] = route % RouteWidth;
        rows += 3;
    }
    __syncwarp();
    if (lane == 0) {
        // This acquire/release edge covers both the count and the pull plan.
        publish_total(local, bank, expert, total);
    }
}

__device__ unsigned wait_total(char* local, unsigned bank, unsigned expert) {
    const uint64_t* address = at<uint64_t>(local, GCMOE_TOTALS) + bank * GCMOE_LOCAL_EXPERTS + expert;
    const unsigned expected = gridDim.x * GCMOE_RANKS;
    const auto started = clock64();
    uint64_t packet;
    do {
        packet = acquire_packet(address);
        if (clock64() - started > GCMOE_TIMEOUT_CLOCKS) {
            asm volatile("trap;");
        }
    } while (static_cast<unsigned>(packet >> 32) != expected);
    return static_cast<unsigned>(packet);
}

__device__ void pull_planned_rows(char* local, const SymBuffer& peers, unsigned bank,
                                  unsigned scratch, unsigned barrier) {
    const unsigned warp = threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    unsigned phase = 0;
    // Consecutive rows go to different SMs. Each assigned warp also owns
    // the same row coordinate of any additional M tile for this expert.
    for (unsigned work = warp * gridDim.x + blockIdx.x;
         work < GCMOE_BLOCK_M * GCMOE_LOCAL_EXPERTS; work += Warps * gridDim.x) {
        const unsigned expert = work / GCMOE_BLOCK_M;
        unsigned total = 0;
        if (lane == 0) {
            total = wait_total(local, bank, expert);
        }
        total = __shfl_sync(FullWarp, total, 0);
        for (unsigned index = work % GCMOE_BLOCK_M; index < total; index += GCMOE_BLOCK_M) {
            const unsigned row = expert * RowsPerExpert + index;
            unsigned source = 0;
            unsigned route = 0;
            if (lane == 0) {
                const auto* metadata = at<const unsigned>(local, GCMOE_METADATA) + row * 3;
                source = metadata[0];
                route = metadata[1] * RouteWidth + metadata[2];
            }
            source = __shfl_sync(FullWarp, source, 0);
            route = __shfl_sync(FullWarp, route, 0);
            const unsigned increment = index + 1 == total ? GCMOE_BLOCK_M - index % GCMOE_BLOCK_M : 1;
            pull_payload<false>(local, peers, source, route, row, increment, scratch, barrier, phase);
            phase ^= 1;
        }
    }
}

// Per-expert queue, with one publisher warp and a pool of TMA worker
// warps. A late source blocks neither earlier packets nor their copies.
__device__ void receive_sources(char* local, unsigned rank, unsigned expert, unsigned epoch,
                                unsigned bank, unsigned* work, unsigned& state) {
    const unsigned lane = threadIdx.x;
    const unsigned source = (lane + rank + expert) % GCMOE_RANKS;
    const unsigned packet_index = (bank * GCMOE_RANKS + source) * GCMOE_LOCAL_EXPERTS + expert;
    const auto* address = at<uint64_t>(local, GCMOE_PACKETS) + packet_index;
    cuda::atomic_ref<unsigned, cuda::thread_scope_block> progress(state);
    unsigned pending = FullWarp >> (32 - GCMOE_RANKS);
    unsigned offset = 0;
    const auto started = clock64();
    while (pending) {
        uint64_t packet = 0;
        if (pending & (1u << lane)) {
            packet = acquire_packet(address);
        }
        const bool valid = (pending & (1u << lane)) && static_cast<unsigned>(packet >> 32) == epoch;
        const unsigned ready = __ballot_sync(FullWarp, valid);
        if (ready) {
            const unsigned count = valid ? static_cast<unsigned>(packet) >> 16 : 0;
            unsigned end = count;
            for (unsigned step = 1; step < 32; step *= 2) {
                const unsigned previous = __shfl_up_sync(FullWarp, end, step);
                if (lane >= step) {
                    end += previous;
                }
            }
            const unsigned next = offset + __shfl_sync(FullWarp, end, 31);
            if (next > RowsPerExpert) {
                asm volatile("trap;");
            }
            for (unsigned ordinal = 0; ordinal < count; ++ordinal) {
                const unsigned index = offset + end - count + ordinal;
                work[index] = source << 24 | ordinal << 16 | (static_cast<unsigned>(packet) & 0xffff);
            }
            __syncwarp();
            if (lane == 0 && next != offset) {
                progress.store(next, cuda::memory_order_release);
            }
            pending &= ~ready;
            offset = next;
        } else if (clock64() - started > GCMOE_TIMEOUT_CLOCKS) {
            asm volatile("trap;");
        }
    }

    if (lane == 0) {
        publish_total(local, bank, expert, offset);
        const unsigned tail = offset % GCMOE_BLOCK_M;
        if (tail) {
            const unsigned tail_block = expert * GCMOE_EXPERT_SPAN + offset / GCMOE_BLOCK_M;
            publish_row(at<unsigned>(local, GCMOE_L1_FULL) + tail_block, GCMOE_BLOCK_M - tail);
        }
        progress.store(QueueDone | offset, cuda::memory_order_release);
    }
}

__device__ void pull_queued_rows(char* local, const SymBuffer& peers, unsigned expert,
                                 unsigned scratch, unsigned barrier,
                                 unsigned* work, unsigned& state) {
    const unsigned warp = threadIdx.x / 32;
    const unsigned lane = threadIdx.x % 32;
    cuda::atomic_ref<unsigned, cuda::thread_scope_block> progress(state);
    unsigned phase = 0;
    for (unsigned index = warp - 1;; index += Warps - 1) {
        unsigned available = 0;
        unsigned route = 0;
        if (lane == 0) {
            do {
                available = progress.load(cuda::memory_order_acquire);
            } while (!(available & QueueDone) && index >= available);
            if (index < (available & ~QueueDone)) {
                route = work[index];
            }
        }
        available = __shfl_sync(FullWarp, available, 0);
        if (index >= (available & ~QueueDone)) {
            break;
        }
        route = __shfl_sync(FullWarp, route, 0);
        const unsigned source = route >> 24;
        const unsigned ordinal = (route >> 16) & 255;
        const unsigned row = expert * RowsPerExpert + index;
        pull_row(local, peers, expert, source, ordinal, route, row, 1, scratch, barrier, phase);
        phase ^= 1;
    }
}
}  // namespace

extern "C" __global__ __launch_bounds__(GCMOE_THREADS)
void gc_decode_dispatch(char* local, const __grid_constant__ SymBuffer peers,
                        unsigned rank, unsigned tokens, float, float) {
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid / 32;
    const unsigned lane = tid % 32;
    if (tid == 0) {
        trace_cta(local, 0);
    }
    if constexpr (GCMOE_PROGRAMMATIC) {
        if (tid == 0) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    const unsigned epoch = *at<volatile unsigned>(local, GCMOE_RECV_PARITY) + 1;
    const unsigned bank = (epoch - 1) & 1;

    __shared__ unsigned route_words[RankedExperts];
    __shared__ unsigned span_begin[32];
    __shared__ unsigned span_end[32];
    __shared__ unsigned inline_routes[32];
    __shared__ unsigned ready_rows;
    __shared__ unsigned row_queue[GCMOE_ARRIVAL_ORDER ? RowsPerExpert : 1];
    __shared__ uint64_t barriers[Warps];
    extern __shared__ __align__(16) char scratch_rows[];
    const unsigned barrier = __cvta_generic_to_shared(&barriers[warp]);
    if (lane == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barrier) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    if (blockIdx.x == 0) {
        publish_routes(local, peers, rank, tokens, epoch, bank, route_words);
        if constexpr (GCMOE_TRACE_OFFSET != 0) {
            __syncthreads();
            if (tid == 0) {
                trace_cta(local, 1);
            }
        }
    }

    const unsigned expert = blockIdx.x % GCMOE_LOCAL_EXPERTS;
    const unsigned scratch = __cvta_generic_to_shared(scratch_rows + warp * RowBytes);
    if constexpr (GCMOE_PULL_PLAN) {
        if (warp == 0 && blockIdx.x == expert) {
            plan_expert(local, expert, bank, epoch);
        }
        pull_planned_rows(local, peers, bank, scratch, barrier);
        trace_exit(local);
        return;
    }
    if constexpr (GCMOE_ARRIVAL_ORDER) {
        if (blockIdx.x == expert) {
            if (tid == 0) {
                ready_rows = 0;
            }
            __syncthreads();
            if (warp == 0) {
                receive_sources(local, rank, expert, epoch, bank, row_queue, ready_rows);
            } else {
                pull_queued_rows(local, peers, expert, scratch, barrier, row_queue, ready_rows);
            }
        }
        trace_exit(local);
        return;
    }
    if (warp == 0) {
        uint64_t packet = 0;
        if (lane < GCMOE_RANKS) {
            packet = wait_packet(local, bank, lane, expert, epoch);
        }
        const unsigned rows = static_cast<unsigned>(packet) >> 16;
        unsigned inclusive = rows;
        for (unsigned step = 1; step < 32; step *= 2) {
            const unsigned previous = __shfl_up_sync(FullWarp, inclusive, step);
            if (lane >= step) {
                inclusive += previous;
            }
        }
        span_begin[lane] = inclusive - rows;
        span_end[lane] = inclusive;
        inline_routes[lane] = static_cast<unsigned>(packet) & 0xffff;
        // The first CTA serving this expert is its sole count publisher.
        if (lane == 31 && blockIdx.x == expert) {
            publish_total(local, bank, expert, inclusive);
        }
    }
    __syncthreads();

    const unsigned total = span_end[31];
    const unsigned first = blockIdx.x / GCMOE_LOCAL_EXPERTS * Warps + warp;
    const unsigned stride = ((gridDim.x - 1 - expert) / GCMOE_LOCAL_EXPERTS + 1) * Warps;
    const unsigned begin = span_begin[lane];
    const unsigned end = span_end[lane];
    const unsigned packet = inline_routes[lane];
    unsigned phase = 0;
    for (unsigned index = first; index < total; index += stride) {
        const unsigned source = __ffs(__ballot_sync(FullWarp, index >= begin && index < end)) - 1;
        const unsigned ordinal = index - __shfl_sync(FullWarp, begin, source);
        const unsigned packed = __shfl_sync(FullWarp, packet, source);
        const unsigned row = expert * RowsPerExpert + index;
        const unsigned increment = index + 1 == total ? GCMOE_BLOCK_M - row % GCMOE_BLOCK_M : 1;
        pull_row(local, peers, expert, source, ordinal, packed, row, increment,
                 scratch, barrier, phase);
        phase ^= 1;
    }
    trace_exit(local);
}
