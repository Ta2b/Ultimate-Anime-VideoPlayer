#include "gpu/vulkan/backend/net_profile.h"

#include <algorithm>
#include <format>
#include <functional>
#include <map>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_measure.h"

namespace ajp::gpu::vk::net {

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
    report.shaders.push_back({.shader = shader,
                              .dispatches = counts[shader],
                              .ms = ms,
                              .share = report.total_ms > 0.0 ? ms / report.total_ms : 0.0});
  }
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

}  // namespace ajp::gpu::vk::net
