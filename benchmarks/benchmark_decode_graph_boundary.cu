// Measure CUDA Graph event-to-event time and the corresponding device kernel spans.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr unsigned Replays = 128;
constexpr unsigned Warmup = 16;

struct Span {
    uint64_t begin;
    uint64_t end;
};

void check(cudaError_t status, const char* step) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(status));
    std::exit(1);
}

__device__ uint64_t now() {
    uint64_t result;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
    return result;
}

__global__ void role_span(Span* spans, unsigned role, uint64_t delay_ns, bool trigger) {
    const uint64_t start = now();
    if (trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    while (now() - start < delay_ns) {}
    spans[role] = {start, now()};
}

std::array<cudaGraphNode_t, 3> add_roles(cudaGraph_t graph, cudaGraphNode_t entry,
                                         Span* spans, bool programmatic, bool busy) {
    std::array<cudaGraphNode_t, 3> nodes{};
    const uint64_t delays[]{12000, 58000, 32000};
    for (unsigned role = 0; role < nodes.size(); ++role) {
        auto duration = busy ? delays[role] : uint64_t{0};
        auto trigger = programmatic && role < 2;
        void* arguments[]{&spans, &role, &duration, &trigger};
        cudaGraphNodeParams params{};
        params.type = cudaGraphNodeTypeKernel;
        params.kernel.func = reinterpret_cast<void*>(role_span);
        params.kernel.gridDim = dim3(1);
        params.kernel.blockDim = dim3(1);
        params.kernel.kernelParams = arguments;
        const cudaGraphNode_t* dependencies = &entry;
        unsigned count = entry ? 1 : 0;
        cudaGraphEdgeData edges[2]{};
        if (role != 0) {
            dependencies = nodes.data();
            count = role;
            if (programmatic) {
                edges[role - 1].type = cudaGraphDependencyTypeProgrammatic;
                edges[role - 1].from_port = cudaGraphKernelNodePortProgrammatic;
            }
        }
        check(cudaGraphAddNode(&nodes[role], graph, dependencies, programmatic ? edges : nullptr, count, &params),
              "add role node");
    }
    return nodes;
}

void run(bool nested, bool programmatic, bool busy) {
    Span* device_spans = nullptr;
    check(cudaMalloc(&device_spans, 3 * sizeof(Span)), "allocate spans");
    cudaEvent_t start{}, end{};
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&end), "create end event");
    cudaGraph_t parent{}, child{};
    check(cudaGraphCreate(&parent, 0), "create parent graph");
    cudaGraphNode_t start_node{}, end_node{};
    check(cudaGraphAddEventRecordNode(&start_node, parent, nullptr, 0, start), "add start event");
    if (nested) {
        check(cudaGraphCreate(&child, 0), "create child graph");
        add_roles(child, nullptr, device_spans, programmatic, busy);
        cudaGraphNode_t child_node{};
        check(cudaGraphAddChildGraphNode(&child_node, parent, &start_node, 1, child), "add child graph");
        check(cudaGraphAddEventRecordNode(&end_node, parent, &child_node, 1, end), "add end event");
    } else {
        const auto roles = add_roles(parent, start_node, device_spans, programmatic, busy);
        check(cudaGraphAddEventRecordNode(&end_node, parent, roles.data(), roles.size(), end), "add end event");
    }
    cudaGraphExec_t graph{};
    check(cudaGraphInstantiate(&graph, parent, 0), "instantiate graph");
    std::vector<double> timed, on_device, outside;
    uint64_t previous_end = 0;
    for (unsigned iter = 0; iter < Replays + Warmup; ++iter) {
        check(cudaGraphLaunch(graph, nullptr), "launch graph");
        check(cudaEventSynchronize(end), "wait for graph");
        if (iter < Warmup) continue;
        float elapsed_ms;
        check(cudaEventElapsedTime(&elapsed_ms, start, end), "measure graph");
        std::array<Span, 3> spans{};
        check(cudaMemcpy(spans.data(), device_spans, sizeof(spans), cudaMemcpyDeviceToHost), "read spans");
        uint64_t first = UINT64_MAX, last = 0;
        for (auto span : spans) {
            if (span.begin <= previous_end || span.end < span.begin) {
                std::fprintf(stderr, "invalid role span\n");
                std::exit(1);
            }
            first = std::min(first, span.begin);
            last = std::max(last, span.end);
        }
        previous_end = last;
        timed.push_back(elapsed_ms * 1000.0);
        on_device.push_back((last - first) / 1000.0);
        outside.push_back(timed.back() - on_device.back());
    }
    for (auto* values : {&timed, &on_device, &outside}) std::sort(values->begin(), values->end());
    std::printf("{\"nested\":%s,\"programmatic\":%s,\"busy\":%s,"
                "\"event_p50_us\":%.3f,\"role_span_p50_us\":%.3f,\"outside_roles_p50_us\":%.3f,"
                "\"event_p95_us\":%.3f}\n",
                nested ? "true" : "false", programmatic ? "true" : "false", busy ? "true" : "false",
                timed[Replays / 2 - 1], on_device[Replays / 2 - 1], outside[Replays / 2 - 1], timed[121]);
    check(cudaGraphExecDestroy(graph), "destroy graph executable");
    check(cudaGraphDestroy(parent), "destroy parent graph");
    if (child) check(cudaGraphDestroy(child), "destroy child graph");
    check(cudaEventDestroy(end), "destroy end event");
    check(cudaEventDestroy(start), "destroy start event");
    check(cudaFree(device_spans), "free spans");
}

__global__ void grid_span(Span* spans, unsigned retire_early, unsigned duration_ns, bool dispatch) {
    if (threadIdx.x != 0) return;
    const uint64_t begin = now();
    if (dispatch) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const unsigned duration = blockIdx.x < retire_early ? 2000 : duration_ns;
    while (now() - begin < duration) {}
    spans[blockIdx.x] = {begin, now()};
}

cudaGraphNode_t add_grid(cudaGraph_t graph, cudaGraphNode_t dependency, Span* spans,
                         unsigned blocks, unsigned retire_early, bool dispatch) {
    cudaGraphNode_t node{};
    unsigned duration = dispatch ? 15000 : 45000;
    void* args[]{&spans, &retire_early, &duration, &dispatch};
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeKernel;
    params.kernel.func = reinterpret_cast<void*>(grid_span);
    params.kernel.gridDim = dim3(blocks);
    params.kernel.blockDim = dim3(dispatch ? 512 : 384);
    // Match the dynamic-SMEM reservations of the physical MoE roles, so
    // two grids cannot overlap by sharing one SM's storage.
    params.kernel.sharedMemBytes = dispatch ? 108388 : 224148;
    params.kernel.kernelParams = args;
    cudaGraphEdgeData edge{};
    if (dependency) {
        edge.type = cudaGraphDependencyTypeProgrammatic;
        edge.from_port = cudaGraphKernelNodePortProgrammatic;
    }
    check(cudaGraphAddNode(&node, graph, dependency ? &dependency : nullptr,
                           dependency ? &edge : nullptr, dependency ? 1 : 0, &params), "add grid node");
    return node;
}

void run_grid_overlap(unsigned dispatch_ctas, unsigned compute_ctas, unsigned early, bool cooperative) {
    check(cudaFuncSetAttribute(grid_span, cudaFuncAttributeMaxDynamicSharedMemorySize, 224148),
          "configure grid storage");
    Span* device = nullptr;
    std::vector<Span> spans(dispatch_ctas + compute_ctas);
    check(cudaMalloc(&device, spans.size() * sizeof(Span)), "allocate grid spans");
    cudaGraph_t graph{};
    check(cudaGraphCreate(&graph, 0), "create grid graph");
    const auto dispatch = add_grid(graph, nullptr, device, dispatch_ctas, early, true);
    const auto compute = add_grid(graph, dispatch, device + dispatch_ctas, compute_ctas, 0, false);
    cudaLaunchAttributeValue attr{};
    attr.clusterDim.x = 2;
    attr.clusterDim.y = attr.clusterDim.z = 1;
    check(cudaGraphKernelNodeSetAttribute(compute, cudaKernelNodeAttributeClusterDimension, &attr),
          "set two-CTA compute cluster");
    if (cooperative) {
        cudaLaunchAttributeValue coop{};
        coop.cooperative = 1;
        check(cudaGraphKernelNodeSetAttribute(compute, cudaKernelNodeAttributeCooperative, &coop),
              "set cooperative compute grid");
    }
    cudaGraphExec_t executable{};
    cudaError_t status = cudaGraphInstantiate(&executable, graph, 0);
    std::vector<double> delay, overlap, role_span;
    std::vector<unsigned> started_early;
    for (unsigned iteration = 0; status == cudaSuccess && iteration < Replays + Warmup; ++iteration) {
        status = cudaGraphLaunch(executable, nullptr);
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        if (status != cudaSuccess || iteration < Warmup) continue;
        check(cudaMemcpy(spans.data(), device, spans.size() * sizeof(Span), cudaMemcpyDeviceToHost),
              "read grid spans");
        uint64_t dispatch_start = UINT64_MAX, dispatch_end = 0, compute_start = UINT64_MAX, compute_end = 0;
        unsigned count = 0;
        for (unsigned block = 0; block < dispatch_ctas; ++block) {
            const Span s = spans[block];
            if (s.begin == 0 || s.end < s.begin) std::abort();
            dispatch_start = std::min(dispatch_start, s.begin);
            dispatch_end = std::max(dispatch_end, s.end);
        }
        for (unsigned block = dispatch_ctas; block < spans.size(); ++block) {
            const Span s = spans[block];
            if (s.begin < dispatch_start || s.end < s.begin) std::abort();
            compute_start = std::min(compute_start, s.begin);
            compute_end = std::max(compute_end, s.end);
            count += s.begin < dispatch_end;
        }
        delay.push_back((compute_start - dispatch_start) / 1000.0);
        overlap.push_back(dispatch_end > compute_start ? (dispatch_end - compute_start) / 1000.0 : 0.0);
        role_span.push_back((std::max(dispatch_end, compute_end) - dispatch_start) / 1000.0);
        started_early.push_back(count);
    }
    std::printf("{\"dispatch_ctas\":%u,\"compute_ctas\":%u,\"retire_early\":%u,\"cooperative\":%s,\"status\":\"%s\"",
                dispatch_ctas, compute_ctas, early, cooperative ? "true" : "false", cudaGetErrorName(status));
    if (status == cudaSuccess) {
        for (auto* values : {&delay, &overlap, &role_span}) std::sort(values->begin(), values->end());
        std::sort(started_early.begin(), started_early.end());
        const unsigned median = Replays / 2 - 1;
        std::printf(",\"compute_start_us\":%.3f,\"overlap_us\":%.3f,\"role_span_us\":%.3f,\"compute_ctas_before_dispatch_end\":%u",
                    delay[median], overlap[median], role_span[median], started_early[median]);
    }
    std::puts("}");
    if (executable) check(cudaGraphExecDestroy(executable), "destroy grid executable");
    check(cudaGraphDestroy(graph), "destroy grid graph");
    check(cudaFree(device), "free grid spans");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "cooperative-overlap") == 0) {
        for (const unsigned dispatch : {24, 32}) {
            for (const unsigned compute : {128, 136, 144}) {
                for (const unsigned early : {0, 8}) {
                    run_grid_overlap(dispatch, compute, early, false);
                    run_grid_overlap(dispatch, compute, early, true);
                }
            }
        }
        return 0;
    }
    for (bool busy : {false, true}) {
        for (bool programmatic : {false, true}) {
            run(true, programmatic, busy);
            run(false, programmatic, busy);
        }
    }
}
