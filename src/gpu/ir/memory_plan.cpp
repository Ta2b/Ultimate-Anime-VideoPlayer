#include "gpu/ir/memory_plan.h"

#include <algorithm>

#include "gpu/ir/graph.h"

namespace ajp::gpu::ir {
namespace {

std::uint64_t round_up(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

bool lives_overlap(const Allocation& a, const Allocation& b) {
  return a.first <= b.last && b.first <= a.last;
}

}  // namespace

MemoryPlan plan_memory(const Graph& graph, std::uint64_t alignment) {
  MemoryPlan plan;
  const std::size_t count = graph.tensors.size();
  std::vector<bool> io(count, false);
  for (const TensorId id : graph.inputs) io[id] = true;
  for (const TensorId id : graph.outputs) io[id] = true;

  // Durées de vie, en indices de nœud.
  std::vector<std::size_t> first(count, kNoNode);
  std::vector<std::size_t> last(count, 0);
  std::vector<bool> constant_read(count, false);
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    for (const TensorId id : graph.nodes[n].outputs) {
      first[id] = n;
      last[id] = n;
    }
    for (const TensorId id : graph.nodes[n].inputs) {
      if (id == kNoTensor) continue;
      last[id] = std::max(last[id], n);
      constant_read[id] = graph.tensors[id].kind == TensorKind::Constant;
    }
  }

  for (TensorId id = 0; id < count; ++id) {
    const Tensor& tensor = graph.tensors[id];
    if (io[id]) {
      plan.io_bytes += storage_bytes(tensor);
    } else if (constant_read[id]) {
      plan.constant_bytes += storage_bytes(tensor);
    } else if (tensor.kind == TensorKind::Activation && first[id] != kNoNode) {
      const std::uint64_t bytes = round_up(storage_bytes(tensor), alignment);
      plan.allocations.push_back({.tensor = id, .bytes = bytes, .first = first[id], .last = last[id]});
      plan.unshared_bytes += bytes;
    }
  }

  // Du plus grand au plus petit ; à taille égale, dans l'ordre d'écriture.
  std::vector<std::size_t> order(plan.allocations.size());
  for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
  std::ranges::stable_sort(order, [&](std::size_t a, std::size_t b) {
    return plan.allocations[a].bytes > plan.allocations[b].bytes;
  });

  std::vector<std::size_t> placed;
  for (const std::size_t k : order) {
    Allocation& alloc = plan.allocations[k];
    // Tampons déjà placés dont la vie croise celle-ci, par décalage croissant.
    std::vector<const Allocation*> busy;
    for (const std::size_t p : placed) {
      if (lives_overlap(alloc, plan.allocations[p])) busy.push_back(&plan.allocations[p]);
    }
    std::ranges::sort(busy, [](const Allocation* a, const Allocation* b) { return a->offset < b->offset; });
    std::uint64_t offset = 0;
    for (const Allocation* other : busy) {
      if (offset + alloc.bytes <= other->offset) break;
      offset = std::max(offset, round_up(other->offset + other->bytes, alignment));
    }
    alloc.offset = offset;
    plan.arena_bytes = std::max(plan.arena_bytes, offset + alloc.bytes);
    placed.push_back(k);
  }
  return plan;
}

}  // namespace ajp::gpu::ir
