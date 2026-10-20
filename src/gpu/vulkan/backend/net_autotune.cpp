#include "gpu/vulkan/backend/net_autotune.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <variant>

#include "ajp_vk_backend_spirv.h"
#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_artifact.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/recording.h"

namespace ajp::gpu::vk::net {
namespace {

// Hachage incrémental d'entiers, FNV-1a sur leurs huit octets.
class Signature {
 public:
  Signature& add(std::int64_t v) {
    std::array<std::byte, 8> b{};
    for (std::size_t k = 0; k < 8; ++k) b[k] = static_cast<std::byte>((static_cast<std::uint64_t>(v) >> (8 * k)) & 0xFFU);
    hash_ = fnv1a(b, hash_);
    return *this;
  }
  Signature& add(float v) { return add(static_cast<std::int64_t>(std::bit_cast<std::uint32_t>(v))); }
  template <std::size_t N>
  Signature& add(const std::array<std::int64_t, N>& values) {
    for (const std::int64_t v : values) add(v);
    return *this;
  }
  Signature& add(const ir::Dims& dims) {
    add(static_cast<std::int64_t>(dims.size()));
    for (const std::int64_t d : dims) add(d);
    return *this;
  }
  std::uint64_t value() const { return hash_; }

 private:
  std::uint64_t hash_ = 0xCBF29CE484222325ULL;
};

// Fonctions des requêtes d'horodatage, hors des tables du socle.
struct QueryFunctions {
  PFN_vkCreateQueryPool create = nullptr;
  PFN_vkDestroyQueryPool destroy = nullptr;
  PFN_vkCmdResetQueryPool reset = nullptr;
  PFN_vkCmdWriteTimestamp write = nullptr;
  PFN_vkGetQueryPoolResults results = nullptr;
};

QueryFunctions query_functions(const VkContext& context) {
  const auto get = [&](const char* name) {
    return context.instance_functions().vkGetDeviceProcAddr(context.device(), name);
  };
  return {.create = reinterpret_cast<PFN_vkCreateQueryPool>(get("vkCreateQueryPool")),
          .destroy = reinterpret_cast<PFN_vkDestroyQueryPool>(get("vkDestroyQueryPool")),
          .reset = reinterpret_cast<PFN_vkCmdResetQueryPool>(get("vkCmdResetQueryPool")),
          .write = reinterpret_cast<PFN_vkCmdWriteTimestamp>(get("vkCmdWriteTimestamp")),
          .results = reinterpret_cast<PFN_vkGetQueryPoolResults>(get("vkGetQueryPoolResults"))};
}

// Période d'horodatage de la file compute en nanosecondes ; 0 sans horodatage.
double timestamp_period(const VkContext& context) {
  const InstanceFunctions& fn = context.instance_functions();
  VkPhysicalDeviceProperties properties{};
  fn.vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
  std::uint32_t count = 0;
  fn.vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  fn.vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &count, families.data());
  if (context.queue_family() >= count || families[context.queue_family()].timestampValidBits == 0) return 0.0;
  const QueryFunctions q = query_functions(context);
  if (q.create == nullptr || q.destroy == nullptr || q.reset == nullptr || q.write == nullptr ||
      q.results == nullptr) {
    return 0.0;
  }
  return static_cast<double>(properties.limits.timestampPeriod);
}

// Pool de deux requêtes, détruit avec l'objet.
class QueryPool {
 public:
  QueryPool(const VkContext& context, QueryFunctions fn) : context_(context), fn_(fn) {}
  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;
  QueryPool(QueryPool&&) = delete;
  QueryPool& operator=(QueryPool&&) = delete;
  ~QueryPool() {
    if (pool_ != VK_NULL_HANDLE) fn_.destroy(context_.device(), pool_, nullptr);
  }
  Done create() {
    const VkQueryPoolCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
    return check(fn_.create(context_.device(), &info, nullptr, &pool_), "vkCreateQueryPool");
  }
  VkQueryPool handle() const { return pool_; }
  const QueryFunctions& fn() const { return fn_; }

 private:
  const VkContext& context_;
  QueryFunctions fn_;
  VkQueryPool pool_ = VK_NULL_HANDLE;
};

double median(std::vector<double> values) {
  std::ranges::sort(values);
  const std::size_t n = values.size();
  return n % 2 == 1 ? values[n / 2] : 0.5 * (values[(n / 2) - 1] + values[n / 2]);
}

double deviation(const std::vector<double>& values) {
  double mean = 0.0;
  for (const double v : values) mean += v;
  mean /= static_cast<double>(values.size());
  double sum = 0.0;
  for (const double v : values) sum += (v - mean) * (v - mean);
  return std::sqrt(sum / static_cast<double>(values.size()));
}

std::string tuning_text(const KernelChoice& c) {
  std::string out;
  for (const std::uint32_t t : c.tuning) out += std::format("{}{}", out.empty() ? "" : ",", t);
  return out;
}

}  // namespace

std::uint64_t layer_signature(const ir::Graph& graph, std::size_t node) {
  const ir::Node& n = graph.nodes[node];
  Signature s;
  s.add(static_cast<std::int64_t>(n.op));
  const auto tensor = [&](ir::TensorId id) {
    if (id == ir::kNoTensor) {
      s.add(std::int64_t{-1});
      return;
    }
    const ir::Tensor& t = graph.tensors[id];
    s.add(static_cast<std::int64_t>(t.kind)).add(static_cast<std::int64_t>(t.type));
    s.add(static_cast<std::int64_t>(t.layout)).add(t.dims);
  };
  for (const ir::TensorId id : n.inputs) tensor(id);
  for (const ir::TensorId id : n.outputs) tensor(id);
  if (const auto* a = std::get_if<ir::ConvAttrs>(&n.attrs)) {
    s.add(a->kernel).add(a->strides).add(a->dilations).add(a->pads);
    s.add(static_cast<std::int64_t>(a->auto_pad)).add(a->group);
    s.add(static_cast<std::int64_t>(a->epilogue.residual)).add(static_cast<std::int64_t>(a->epilogue.act));
    s.add(a->epilogue.alpha).add(a->epilogue.lo).add(a->epilogue.hi);
  } else if (const auto* t = std::get_if<ir::ConvTransposeAttrs>(&n.attrs)) {
    s.add(t->kernel).add(t->strides).add(t->dilations).add(t->pads).add(t->output_padding);
    s.add(t->output_shape).add(static_cast<std::int64_t>(t->auto_pad)).add(t->group);
  }
  return s.value();
}

std::span<const PresetTable> preset_tables() {
  // Vides : à remplir depuis des mesures de référence, famille par famille.
  static constexpr std::array<PresetTable, 0> kTables{};
  return kTables;
}

Autotuner::Autotuner(VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines, DeviceProfile profile,
                     std::filesystem::path cache_file, TunerOptions options)
    : context_(context),
      memory_(memory),
      pipelines_(pipelines),
      profile_(std::move(profile)),
      cache_file_(std::move(cache_file)),
      options_(options),
      period_ns_(timestamp_period(context)) {
  load_cache();
}

void Autotuner::load_cache() {
  if (cache_file_.empty()) return;
  std::ifstream in(cache_file_);
  std::string line;
  while (std::getline(in, line)) {
    const std::size_t a = line.find('\t');
    const std::size_t b = a == std::string::npos ? a : line.find('\t', a + 1);
    if (b == std::string::npos) continue;  // ligne abîmée : ignorée, elle se remesurera
    KernelChoice choice{.variant = line.substr(a + 1, b - a - 1)};
    std::stringstream tunings(line.substr(b + 1));
    std::string item;
    bool valid = true;
    while (std::getline(tunings, item, ',')) {
      std::uint32_t v = 0;
      const auto [ptr, ec] = std::from_chars(item.data(), item.data() + item.size(), v);
      valid = valid && ec == std::errc{} && ptr == item.data() + item.size();
      choice.tuning.push_back(v);
    }
    if (valid) cache_[line.substr(0, a)] = std::move(choice);
  }
}

std::expected<void, Refusal> Autotuner::save() const {
  if (cache_file_.empty()) return {};
  std::error_code error;
  std::filesystem::create_directories(cache_file_.parent_path(), error);
  const std::filesystem::path temporary = cache_file_.string() + ".tmp";
  {
    std::ofstream out(temporary, std::ios::trunc);
    for (const auto& [key, choice] : cache_) out << key << '\t' << choice.variant << '\t' << tuning_text(choice) << '\n';
    if (!out) return std::unexpected(Refusal{RefusalCode::CacheIo, std::format("écriture de {}", temporary.string())});
  }
  std::filesystem::rename(temporary, cache_file_, error);
  if (error) return std::unexpected(Refusal{RefusalCode::CacheIo, std::format("{} : {}", cache_file_.string(), error.message())});
  return {};
}

std::expected<KernelChoice, Refusal> Autotuner::choose(const ir::Graph& graph, std::size_t node,
                                                       std::span<const KernelChoice> candidates,
                                                       std::string_view scope, std::stop_token stop) {
  if (candidates.empty()) return std::unexpected(Refusal{RefusalCode::InputMismatch, "aucun candidat"});
  if (candidates.size() == 1) return candidates.front();
  const std::uint64_t signature = layer_signature(graph, node);
  const auto known = [&](const KernelChoice& c) { return std::ranges::find(candidates, c) != candidates.end(); };
  for (const PresetTable& table : preset_tables()) {
    if (table.family != profile_.family) continue;
    for (const PresetEntry& e : table.entries) {
      const KernelChoice c{.variant = std::string(e.variant), .tuning = {e.tuning0, e.tuning1}};
      if (e.signature == signature && known(c)) {
        ++cached_;
        return c;
      }
    }
  }
  const std::string key = std::format("{}|{}|{:016x}", profile_.key, scope, signature);
  if (const auto found = cache_.find(key); found != cache_.end() && known(found->second)) {
    ++cached_;
    return found->second;
  }
  std::optional<Measured> best;
  std::optional<Refusal> last_error;
  for (const KernelChoice& c : candidates) {
    if (stop.stop_requested()) return std::unexpected(Refusal{RefusalCode::Cancelled, "autotuner arrêté"});
    auto m = measure(graph, node, c);
    if (!m) {
      last_error = m.error();
      continue;
    }
    if (!best || m->median_us < best->median_us) best = std::move(*m);
  }
  if (!best) return std::unexpected(*last_error);
  ++measured_;
  cache_[key] = best->choice;
  return best->choice;
}

std::expected<Measured, Refusal> Autotuner::measure(const ir::Graph& graph, std::size_t node,
                                                    const KernelChoice& choice) {
  Lowerer lowerer(graph, profile_.target);
  if (auto ok = lowerer.node(node, choice); !ok) return std::unexpected(ok.error());
  const Program program = lowerer.take();

  // Des tampons neufs pour les tenseurs du nœud, les blobs envoyés une fois, un tampon factice.
  std::map<std::uint32_t, Buffer> tensors;
  std::vector<Buffer> blobs;
  for (const auto& bytes : program.blobs) {
    auto buffer = Buffer::create(context_, memory_, std::max<std::size_t>(bytes.size(), 4), MemoryUse::Device);
    if (!buffer) return std::unexpected(buffer.error());
    if (auto ok = upload(context_, memory_, *buffer, 0, bytes); !ok) return std::unexpected(ok.error());
    blobs.push_back(std::move(*buffer));
  }
  auto dummy = Buffer::create(context_, memory_, 256, MemoryUse::Device);
  if (!dummy) return std::unexpected(dummy.error());
  std::vector<const Pipeline*> pipelines;
  std::uint32_t buffers = 0;
  for (const Dispatch& d : program.dispatches) {
    for (const Binding& b : d.bindings) {
      if (b.kind != BindingKind::Tensor || tensors.contains(b.index)) continue;
      const std::uint64_t bytes = (ir::storage_bytes(graph.tensors[b.index]) + 3) / 4 * 4;
      auto buffer = Buffer::create(context_, memory_, bytes, MemoryUse::Device);
      if (!buffer) return std::unexpected(buffer.error());
      tensors.emplace(b.index, std::move(*buffer));
    }
    auto pipeline = pipelines_.get({.shader = d.shader,
                                    .constants = d.constants,
                                    .buffers = static_cast<std::uint32_t>(d.bindings.size()),
                                    .push_bytes = static_cast<std::uint32_t>(d.push.size() * 4),
                                    .spirv = spirv::words(d.shader)});
    if (!pipeline) return std::unexpected(pipeline.error());
    pipelines.push_back(*pipeline);
    buffers += static_cast<std::uint32_t>(d.bindings.size());
  }
  const auto view = [&](const Binding& b) {
    if (b.kind == BindingKind::Tensor) return tensors.at(b.index).view();
    if (b.kind == BindingKind::Blob) return blobs[b.index].view();
    return dummy->view();
  };

  // Les tenseurs remplis de zéros une fois : une entrée non initialisée pourrait porter des NaN.
  {
    auto fill = Recorder::begin(context_, memory_);
    if (!fill) return std::unexpected(fill.error());
    for (const auto& [id, buffer] : tensors) {
      if (auto ok = fill->fill(buffer.view(), 0); !ok) return std::unexpected(ok.error());
    }
    auto recording = fill->finish();
    if (!recording) return std::unexpected(recording.error());
    if (auto ok = context_.submit(**recording); !ok) return std::unexpected(ok.error());
    if (auto ok = context_.wait_on_host((*recording)->last_run()); !ok) return std::unexpected(ok.error());
  }

  QueryPool queries(context_, query_functions(context_));
  if (period_ns_ > 0.0) {
    if (auto ok = queries.create(); !ok) return std::unexpected(ok.error());
  }
  const std::uint32_t repeats = std::max<std::uint32_t>(1, options_.repeats);
  auto recorder = Recorder::begin(context_, memory_,
                                  {.sets = static_cast<std::uint32_t>(program.dispatches.size()) * repeats,
                                   .buffers = buffers * repeats,
                                   .images = 1});
  if (!recorder) return std::unexpected(recorder.error());
  const auto stamp = [&](std::uint32_t index) -> Done {
    if (period_ns_ <= 0.0) return {};
    return recorder->commands([&](VkCommandBuffer cb) {
      if (index == 0) queries.fn().reset(cb, queries.handle(), 0, 2);
      queries.fn().write(cb, index == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         queries.handle(), index);
    });
  };
  if (auto ok = stamp(0); !ok) return std::unexpected(ok.error());
  std::vector<BufferView> views;
  for (std::uint32_t r = 0; r < repeats; ++r) {
    for (std::size_t k = 0; k < program.dispatches.size(); ++k) {
      const Dispatch& d = program.dispatches[k];
      views.clear();
      for (const Binding& b : d.bindings) views.push_back(view(b));
      if (auto ok = recorder->dispatch(*pipelines[k], views, {}, std::as_bytes(std::span(d.push)), d.groups); !ok) {
        return std::unexpected(ok.error());
      }
    }
  }
  if (auto ok = stamp(1); !ok) return std::unexpected(ok.error());
  auto recording = recorder->finish();
  if (!recording) return std::unexpected(recording.error());

  std::vector<double> samples;
  for (std::uint32_t pass = 0; pass < options_.warmup + std::max<std::uint32_t>(1, options_.passes); ++pass) {
    const auto start = std::chrono::steady_clock::now();
    if (auto ok = context_.submit(**recording); !ok) return std::unexpected(ok.error());
    if (auto ok = context_.wait_on_host((*recording)->last_run()); !ok) return std::unexpected(ok.error());
    const auto host = std::chrono::steady_clock::now() - start;
    double us = std::chrono::duration<double, std::micro>(host).count();
    if (period_ns_ > 0.0) {
      std::array<std::uint64_t, 2> ticks{};
      if (auto ok = check(queries.fn().results(context_.device(), queries.handle(), 0, 2, sizeof(ticks), ticks.data(),
                                               sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT),
                          "vkGetQueryPoolResults");
          !ok) {
        return std::unexpected(ok.error());
      }
      us = static_cast<double>(ticks[1] - ticks[0]) * period_ns_ / 1000.0;
    }
    if (pass >= options_.warmup) samples.push_back(us / repeats);
  }
  return Measured{.choice = choice, .median_us = median(samples), .spread_us = deviation(samples)};
}

}  // namespace ajp::gpu::vk::net
