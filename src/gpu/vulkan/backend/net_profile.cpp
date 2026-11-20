#include "gpu/vulkan/backend/net_profile.h"

#include <algorithm>
#include <format>
#include <functional>
#include <map>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"
#include "gpu/vulkan/backend/net_measure.h"

namespace kirari::gpu::vk::net {
namespace {

// Les médianes de plusieurs catégories ne s'additionnent pas à la médiane du total.
template <class Row>
void distribute(std::vector<Row>& rows) {
  double sum = 0.0;
  for (const Row& row : rows) sum += row.ms;
  if (sum <= 0.0) return;
  for (Row& row : rows) row.share = row.ms / sum;
}

}  // namespace

ProfileReport summarize_profile(std::span<const Dispatch> dispatches, std::span<const ProfileRun> runs) {
  ProfileReport report;
  report.runs = runs.size();
  if (runs.empty()) return report;
  // Par shader : une somme par exécution, puis la médiane des sommes.
  std::map<std::string, std::vector<double>> sums;
  std::map<std::string, std::uint32_t> counts;
  for (const Dispatch& d : dispatches) ++counts[d.shader];
  std::vector<double> totals;
  std::vector<double> spans;
  for (const ProfileRun& run : runs) {
    std::map<std::string, double> sum;
    double total = 0.0;
    for (std::size_t k = 0; k < dispatches.size() && k < run.dispatch_ms.size(); ++k) {
      sum[dispatches[k].shader] += run.dispatch_ms[k];
      total += run.dispatch_ms[k];
    }
    for (const auto& [shader, ms] : sum) sums[shader].push_back(ms);
    totals.push_back(total);
    spans.push_back(run.span_ms);
  }
  report.total_ms = median(totals);
  report.span_ms = median(spans);
  for (const auto& [shader, values] : sums) {
    const double ms = median(values);
    report.shaders.push_back({.shader = shader, .dispatches = counts[shader], .ms = ms});
  }
  distribute(report.shaders);
  std::ranges::sort(report.shaders, std::greater<>{}, &ShaderTime::ms);
  return report;
}

std::string format_profile(const ProfileReport& report) {
  std::string text = std::format("profil des dispatchs ({} exécutions, médianes)\n", report.runs);
  text += std::format("{:<40} {:>9} {:>10} {:>7}\n", "shader", "dispatchs", "ms", "part");
  for (const ShaderTime& s : report.shaders) {
    text += std::format("{:<40} {:>9} {:>10.3f} {:>6.1f}%\n", s.shader, s.dispatches, s.ms, 100.0 * s.share);
  }
  text += std::format("total des dispatchs : {:.3f} ms ; de la première estampille à la dernière : {:.3f} ms",
                      report.total_ms, report.span_ms);
  return text;
}

std::vector<NodeTime> summarize_nodes(const ir::Graph& graph, std::span<const Dispatch> dispatches,
                                      std::span<const ProfileRun> runs) {
  std::vector<NodeTime> nodes;
  if (runs.empty()) return nodes;
  // Les nœuds dans l'ordre de leur premier dispatch ; slot[k] : la ligne du dispatch k.
  std::map<std::uint32_t, std::size_t> row_of;
  std::vector<std::size_t> slot(dispatches.size());
  for (std::size_t k = 0; k < dispatches.size(); ++k) {
    const Dispatch& d = dispatches[k];
    auto [it, fresh] = row_of.try_emplace(d.node, nodes.size());
    if (fresh) {
      NodeTime row{.node = d.node};
      if (d.node < graph.nodes.size()) {
        const ir::Node& n = graph.nodes[d.node];
        row.op = std::string(ir::name(n.op));
        if (!n.outputs.empty() && n.outputs[0] < graph.tensors.size()) {
          row.output = graph.tensors[n.outputs[0]].name;
          row.dims = graph.tensors[n.outputs[0]].dims;
        }
      }
      nodes.push_back(std::move(row));
    }
    NodeTime& row = nodes[it->second];
    ++row.dispatches;
    if (row.shaders.find(d.shader) == std::string::npos)
      row.shaders += (row.shaders.empty() ? "" : "+") + d.shader;
    slot[k] = it->second;
  }
  std::vector<std::vector<double>> sums(nodes.size());
  for (const ProfileRun& run : runs) {
    std::vector<double> sum(nodes.size(), 0.0);
    for (std::size_t k = 0; k < dispatches.size() && k < run.dispatch_ms.size(); ++k)
      sum[slot[k]] += run.dispatch_ms[k];
    for (std::size_t r = 0; r < nodes.size(); ++r) sums[r].push_back(sum[r]);
  }
  for (std::size_t r = 0; r < nodes.size(); ++r) nodes[r].ms = median(sums[r]);
  distribute(nodes);
  return nodes;
}

std::string format_node_profile(std::span<const NodeTime> nodes) {
  std::string text = "profil par nœud (ordre du programme, médianes)\n";
  text += std::format("{:>5} {:<13} {:<20} {:<44} {:>10} {:>7}\n", "nœud", "opérateur", "sortie", "shaders",
                      "ms", "part");
  for (const NodeTime& n : nodes) {
    text += std::format("{:>5} {:<13} {:<20} {:<44} {:>10.3f} {:>6.1f}%  {}\n", n.node, n.op,
                        ir::format_dims(n.dims), n.shaders, n.ms, 100.0 * n.share, n.output);
  }
  return text;
}

std::vector<ShaderCount> static_count(const ir::Graph& graph, const Program& program) {
  std::map<std::string, ShaderCount> by_shader;
  for (const Dispatch& d : program.dispatches) {
    ShaderCount& row = by_shader[d.shader];
    row.shader = d.shader;
    ++row.dispatches;
    const std::vector<ir::TensorId>* outputs =
        d.node < graph.nodes.size() ? &graph.nodes[d.node].outputs : nullptr;
    for (const Binding& b : d.bindings) {
      switch (b.kind) {
        case BindingKind::Tensor: {
          if (b.index >= graph.tensors.size()) break;
          const std::uint64_t bytes = ir::storage_bytes(graph.tensors[b.index]);
          const bool written = outputs != nullptr && std::ranges::contains(*outputs, b.index);
          (written ? row.written_bytes : row.read_bytes) += bytes;
          break;
        }
        case BindingKind::Blob:
          if (b.index < program.blobs.size()) row.read_bytes += program.blobs[b.index].size();
          break;
        case BindingKind::Dummy:
        case BindingKind::Scratch:
          break;
      }
    }
  }
  std::vector<ShaderCount> rows;
  rows.reserve(by_shader.size());
  for (auto& [shader, row] : by_shader) rows.push_back(std::move(row));
  std::ranges::sort(rows, std::greater<>{}, &ShaderCount::dispatches);
  return rows;
}

}  // namespace kirari::gpu::vk::net
