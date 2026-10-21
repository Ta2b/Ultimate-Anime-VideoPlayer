#include "gpu/vulkan/backend/net_autotune.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <variant>

#include "ajp_vk_backend_spirv.h"
#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_artifact.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/recording.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::vk::net {
namespace {

// 2 : Conv directe à plusieurs pixels (PIX) ; 3 : taille de sous-groupe exigée ; 4 : coopmat à
// 8 sous-groupes, en wave32 sur AMD, sorties vérifiées contre le noyau direct ; 5 : Conv 3 × 3
// par Winograd.
constexpr std::string_view kCandidateSet = "c5";

// Hachage incrémental d'entiers, FNV-1a sur leurs huit octets.
class Signature {
 public:
  Signature& add(std::int64_t v) {
    std::array<std::byte, 8> b{};
    for (std::size_t k = 0; k < 8; ++k)
      b[k] = static_cast<std::byte>((static_cast<std::uint64_t>(v) >> (8 * k)) & 0xFFU);
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

// Pool de count requêtes, détruit avec l'objet.
class QueryPool {
 public:
  QueryPool(const VkContext& context, QueryFunctions fn, std::uint32_t count)
      : context_(context), fn_(fn), count_(count) {}
  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;
  QueryPool(QueryPool&&) = delete;
  QueryPool& operator=(QueryPool&&) = delete;
  ~QueryPool() {
    if (pool_ != VK_NULL_HANDLE) fn_.destroy(context_.device(), pool_, nullptr);
  }
  Done create() {
    const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                     .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                     .queryCount = count_};
    return check(fn_.create(context_.device(), &info, nullptr, &pool_), "vkCreateQueryPool");
  }
  VkQueryPool handle() const { return pool_; }
  std::uint32_t count() const { return count_; }
  const QueryFunctions& fn() const { return fn_; }

 private:
  const VkContext& context_;
  QueryFunctions fn_;
  std::uint32_t count_;
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

// Octets du motif de tuning_input : une période du type, 0 hors float16 et float.
std::size_t pattern_bytes(ir::DataType type) {
  if (type == ir::DataType::F16) return std::size_t{kPatternPeriodF16} * 2;
  return type == ir::DataType::F32 ? std::size_t{kPatternPeriodF32} * 4 : 0;
}

std::vector<float> as_floats(std::span<const std::byte> bytes, ir::DataType type) {
  std::vector<float> out;
  if (type == ir::DataType::F16) {
    out.resize(bytes.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, bytes.data() + (2 * i), 2);
      out[i] = kernels::ref::half_to_float(bits);
    }
  } else if (type == ir::DataType::F32) {
    out.resize(bytes.size() / 4);
    std::memcpy(out.data(), bytes.data(), out.size() * 4);
  }
  return out;
}

// La sortie d'un candidat concorde-t-elle avec celle de la référence ? Écart permis : 3 % de la
// plus grande valeur de la référence plus 2 % de la valeur (sommes en float16 de direct_f16p).
bool agrees(const std::vector<float>& out, const std::vector<float>& ref) {
  if (out.size() != ref.size()) return false;
  float peak = 0.0F;
  for (const float v : ref) {
    if (std::isfinite(v)) peak = std::max(peak, std::abs(v));
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (std::isnan(out[i]) != std::isnan(ref[i])) return false;
    if (std::isnan(ref[i])) continue;
    if (!(std::abs(out[i] - ref[i]) <= (0.03F * peak) + (0.02F * std::abs(ref[i])) + 1e-3F)) return false;
  }
  return true;
}

std::string tuning_text(const KernelChoice& c) {
  std::string out;
  for (const std::uint32_t t : c.tuning) out += std::format("{}{}", out.empty() ? "" : ",", t);
  return out;
}

}  // namespace

float tuning_input(ir::DataType type, std::uint64_t index) {
  if (type != ir::DataType::F16 && type != ir::DataType::F32) return 0.0F;
  // splitmix64 de l'index dans la période, 24 bits gardés : exacts en float.
  std::uint64_t z =
      (index % (type == ir::DataType::F16 ? kPatternPeriodF16 : kPatternPeriodF32)) + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  z ^= z >> 31U;
  const float v = (static_cast<float>(z >> 40U) / 8388608.0F) - 1.0F;
  return type == ir::DataType::F16 ? kernels::ref::half_to_float(kernels::ref::float_to_half(v)) : v;
}

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
    s.add(static_cast<std::int64_t>(a->epilogue.gate)).add(a->epilogue.space);
    s.add(static_cast<std::int64_t>(a->epilogue.space_mode));
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

Autotuner::Autotuner(VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines,
                     DeviceProfile profile, std::filesystem::path cache_file, TunerOptions options)
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
    // Quatrième colonne, la version du pilote ; absente des caches plus anciens : revérifiée.
    const std::size_t c = line.find('\t', b + 1);
    KernelChoice choice{.variant = line.substr(a + 1, b - a - 1)};
    std::stringstream tunings(line.substr(b + 1, c == std::string::npos ? std::string::npos : c - b - 1));
    std::string item;
    bool valid = true;
    while (std::getline(tunings, item, ',')) {
      std::uint32_t v = 0;
      const auto [ptr, ec] = std::from_chars(item.data(), item.data() + item.size(), v);
      valid = valid && ec == std::errc{} && ptr == item.data() + item.size();
      choice.tuning.push_back(v);
    }
    if (valid) {
      cache_[line.substr(0, a)] = {.choice = std::move(choice),
                                   .driver = c == std::string::npos ? std::string() : line.substr(c + 1)};
    }
  }
}

std::expected<void, Refusal> Autotuner::save() const {
  if (cache_file_.empty()) return {};
  // Par write_file : nom de travail propre à cet écrivain, puis renommage.
  std::ostringstream text;
  for (const auto& [key, entry] : cache_) {
    text << key << '\t' << entry.choice.variant << '\t' << tuning_text(entry.choice) << '\t' << entry.driver
         << '\n';
  }
  const std::string bytes = text.str();
  return write_file(cache_file_, std::as_bytes(std::span(bytes)));
}

std::expected<KernelChoice, Refusal> Autotuner::choose(const ir::Graph& graph, std::size_t node,
                                                       std::span<const KernelChoice> candidates,
                                                       std::string_view scope, const std::stop_token& stop) {
  if (candidates.empty()) return std::unexpected(Refusal{RefusalCode::InputMismatch, "aucun candidat"});
  if (candidates.size() == 1) return candidates.front();
  const std::uint64_t signature = layer_signature(graph, node);
  const auto known = [&](const KernelChoice& c) {
    return std::ranges::find(candidates, c) != candidates.end();
  };
  // La carte et le nom du pilote, pas sa version : une mise à jour du pilote ne remesure rien. La
  // version des candidats : une entrée tranchée avant l'ajout d'une variante est remesurée.
  const std::string key =
      std::format("{}|{}|{:016x}|{}", profile_.stable_key, scope, signature, kCandidateSet);
  std::optional<KernelChoice> remembered;
  if (const auto found = cache_.find(key); found != cache_.end() && known(found->second.choice)) {
    if (found->second.driver == profile_.driver_build) {
      ++cached_;
      return found->second.choice;
    }
    remembered = found->second.choice;
  }
  for (const PresetTable& table : preset_tables()) {
    if (remembered || table.family != profile_.family) continue;
    for (const PresetEntry& e : table.entries) {
      KernelChoice c{.variant = std::string(e.variant), .tuning = {e.tuning0, e.tuning1}};
      if (e.signature == signature && known(c)) {
        remembered = std::move(c);
        break;
      }
    }
  }
  // Tranché sous une autre version du pilote, ou lu dans la table : sa sortie, pas son temps, se
  // revérifie. Discordante, le nœud se remesure.
  if (remembered) {
    auto same = agrees_with_direct(graph, node, *remembered, candidates);
    if (!same) return std::unexpected(same.error());
    if (*same) {
      ++cached_;
      ++reverified_;
      cache_[key] = {.choice = *remembered, .driver = profile_.driver_build};
      return *remembered;
    }
    ++rejected_;
  }
  // Le noyau direct d'abord : sa sortie sert de référence aux autres variantes. Chaque candidat
  // est d'abord sondé (une passe d'échauffement, une passe comptée, un dispatch) et sa sortie
  // comparée ; seuls ceux à moins de keep_within fois le meilleur sondage sont mesurés en entier.
  std::vector<KernelChoice> order(candidates.begin(), candidates.end());
  std::ranges::stable_partition(order, [](const KernelChoice& c) { return c.variant == "direct"; });
  struct Probe {
    KernelChoice choice;
    double us = 0.0;
  };
  std::vector<Probe> probes;
  std::optional<Refusal> last_error;
  std::optional<std::vector<float>> reference;
  const TunerOptions probe{.warmup = 1, .passes = 1, .repeats = 1};
  for (const KernelChoice& c : order) {
    if (stop.stop_requested()) return std::unexpected(Refusal{RefusalCode::Cancelled, "autotuner arrêté"});
    auto m = measure_with(graph, node, c, probe);
    if (!m) {
      last_error = m.error();
      continue;
    }
    if (!reference) {
      reference = std::move(m->output);
    } else if (!agrees(m->output, *reference)) {
      ++rejected_;
      last_error =
          Refusal{RefusalCode::InputMismatch,
                  std::format("{} {} : sortie discordante du noyau direct", c.variant, tuning_text(c))};
      continue;
    }
    probes.push_back({.choice = c, .us = m->median_us});
  }
  if (probes.empty()) {
    return std::unexpected(last_error.value_or(Refusal{RefusalCode::InputMismatch, "aucun candidat mesuré"}));
  }
  if (probes.size() == 1) {  // un seul candidat concordant : rien à départager
    ++measured_;
    cache_[key] = {.choice = probes.front().choice, .driver = profile_.driver_build};
    return probes.front().choice;
  }
  const double fastest = std::ranges::min(probes, {}, &Probe::us).us;
  std::optional<Measured> best;
  for (const Probe& p : probes) {
    if (p.us > options_.keep_within * fastest) {
      ++pruned_;
      continue;
    }
    if (stop.stop_requested()) return std::unexpected(Refusal{RefusalCode::Cancelled, "autotuner arrêté"});
    auto m = measure(graph, node, p.choice);
    if (!m) {
      last_error = m.error();
      continue;
    }
    if (!best || m->median_us < best->median_us) best = std::move(*m);
  }
  if (!best) {
    return std::unexpected(last_error.value_or(Refusal{RefusalCode::InputMismatch, "aucun candidat mesuré"}));
  }
  ++measured_;
  cache_[key] = {.choice = best->choice, .driver = profile_.driver_build};
  return best->choice;
}

std::expected<bool, Refusal> Autotuner::agrees_with_direct(const ir::Graph& graph, std::size_t node,
                                                           const KernelChoice& choice,
                                                           std::span<const KernelChoice> candidates) {
  if (choice.variant == "direct") return true;
  const auto direct = std::ranges::find(candidates, std::string_view("direct"), &KernelChoice::variant);
  if (direct == candidates.end()) return false;
  const TunerOptions once{.warmup = 0, .passes = 1, .repeats = 1};
  auto reference = measure_with(graph, node, *direct, once);
  if (!reference) return std::unexpected(reference.error());
  auto mine = measure_with(graph, node, choice, once);
  if (!mine) return std::unexpected(mine.error());
  return agrees(mine->output, reference->output);
}

std::expected<const Buffer*, Refusal> Autotuner::pattern(ir::DataType type) {
  const std::size_t bytes = pattern_bytes(type);
  if (bytes == 0) return nullptr;
  std::optional<Buffer>& slot = type == ir::DataType::F16 ? pattern_f16_ : pattern_f32_;
  if (!slot) {
    std::vector<std::byte> host(bytes);
    if (type == ir::DataType::F16) {
      for (std::size_t i = 0; i < bytes / 2; ++i) {
        const std::uint16_t bits = kernels::ref::float_to_half(tuning_input(type, i));
        std::memcpy(host.data() + (2 * i), &bits, 2);
      }
    } else {
      for (std::size_t i = 0; i < bytes / 4; ++i) {
        const float v = tuning_input(type, i);
        std::memcpy(host.data() + (4 * i), &v, 4);
      }
    }
    auto buffer = Buffer::create(context_, memory_, bytes, MemoryUse::Device);
    if (!buffer) return std::unexpected(buffer.error());
    if (auto ok = upload(context_, memory_, *buffer, 0, host); !ok) return std::unexpected(ok.error());
    slot = std::move(*buffer);
  }
  return &*slot;
}

std::expected<Measured, Refusal> Autotuner::measure(const ir::Graph& graph, std::size_t node,
                                                    const KernelChoice& choice) {
  return measure_with(graph, node, choice, options_);
}

std::expected<Measured, Refusal> Autotuner::measure_with(const ir::Graph& graph, std::size_t node,
                                                         const KernelChoice& choice,
                                                         const TunerOptions& options) {
  Lowerer lowerer(graph, profile_.target);
  if (auto ok = lowerer.node(node, choice); !ok) return std::unexpected(ok.error());
  const Program program = lowerer.take();

  // Des tampons neufs pour les tenseurs du nœud et les blobs, ceux-ci écrits dans un tampon de
  // transit commun puis copiés par la préparation ; un tampon factice.
  std::map<std::uint32_t, Buffer> tensors;
  std::vector<Buffer> blobs;
  std::size_t staged = 0;
  for (const auto& bytes : program.blobs) {
    auto buffer =
        Buffer::create(context_, memory_, std::max<std::size_t>(bytes.size(), 4), MemoryUse::Device);
    if (!buffer) return std::unexpected(buffer.error());
    blobs.push_back(std::move(*buffer));
    staged += bytes.size();
  }
  std::optional<Buffer> staging;
  if (staged > 0) {
    auto buffer = Buffer::create(context_, memory_, staged, MemoryUse::Upload);
    if (!buffer) return std::unexpected(buffer.error());
    std::size_t offset = 0;
    for (const auto& bytes : program.blobs) {
      std::memcpy(buffer->mapped() + offset, bytes.data(), bytes.size());
      offset += bytes.size();
    }
    staging = std::move(*buffer);
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
                                    .spirv = spirv::words(d.shader),
                                    .subgroup_size = d.subgroup});
    if (!pipeline) return std::unexpected(pipeline.error());
    pipelines.push_back(*pipeline);
    buffers += static_cast<std::uint32_t>(d.bindings.size());
  }
  const auto view = [&](const Binding& b) {
    if (b.kind == BindingKind::Tensor) return tensors.at(b.index).view();
    if (b.kind == BindingKind::Blob) return blobs[b.index].view();
    return dummy->view();
  };

  // Le motif de chaque entrée, nullptr pour les tenseurs remplis de zéros : une entrée non
  // initialisée pourrait porter des NaN, et la sortie doit pouvoir se comparer d'un candidat à
  // l'autre.
  const ir::Node& n = graph.nodes[node];
  std::map<std::uint32_t, const Buffer*> sources;
  for (const auto& [id, buffer] : tensors) {
    const Buffer* source = nullptr;
    if (std::ranges::contains(n.inputs, static_cast<ir::TensorId>(id))) {
      auto found = pattern(graph.tensors[id].type);
      if (!found) return std::unexpected(found.error());
      source = *found;
    }
    sources.emplace(id, source);
  }
  // La préparation : les zéros, puis les blobs depuis le transit et le motif sur chaque entrée,
  // copié période par période.
  const auto prepare = [&](Recorder& recorder) -> Done {
    for (const auto& [id, buffer] : tensors) {
      if (sources.at(id) != nullptr) continue;
      if (auto ok = recorder.fill(buffer.view(), 0U); !ok) return ok;
    }
    return recorder.commands([&](VkCommandBuffer cb) {
      const DeviceFunctions& fn = context_.fn();
      VkDeviceSize offset = 0;
      for (std::size_t k = 0; staging && k < program.blobs.size(); ++k) {
        const VkDeviceSize size = program.blobs[k].size();
        if (size == 0) continue;
        const VkBufferCopy region{.srcOffset = offset, .dstOffset = 0, .size = size};
        fn.vkCmdCopyBuffer(cb, staging->handle(), blobs[k].handle(), 1, &region);
        offset += size;
      }
      std::vector<VkBufferCopy> regions;
      for (const auto& [id, buffer] : tensors) {
        const Buffer* source = sources.at(id);
        if (source == nullptr) continue;
        regions.clear();
        for (VkDeviceSize at = 0; at < buffer.size(); at += source->size())
          regions.push_back(
              {.srcOffset = 0, .dstOffset = at, .size = std::min(source->size(), buffer.size() - at)});
        fn.vkCmdCopyBuffer(cb, source->handle(), buffer.handle(), static_cast<std::uint32_t>(regions.size()),
                           regions.data());
      }
    });
  };

  // Avec l'horodatage, une seule soumission : la préparation, puis toutes les passes, une
  // estampille avant la première et après chacune (fin de pipeline : chaque intervalle couvre une
  // passe entière). Sans, la préparation à part, puis une passe par soumission, chronométrée sur
  // l'hôte.
  const bool stamped = period_ns_ > 0.0;
  const std::uint32_t passes = options.warmup + std::max<std::uint32_t>(1, options.passes);
  const std::uint32_t repeats = std::max<std::uint32_t>(1, options.repeats);
  const std::uint32_t recorded = stamped ? passes : 1;
  QueryPool queries(context_, query_functions(context_), passes + 1);
  if (stamped) {
    if (auto ok = queries.create(); !ok) return std::unexpected(ok.error());
  } else {
    auto setup = Recorder::begin(context_, memory_);
    if (!setup) return std::unexpected(setup.error());
    if (auto ok = prepare(*setup); !ok) return std::unexpected(ok.error());
    auto recording = setup->finish();
    if (!recording) return std::unexpected(recording.error());
    if (auto ok = context_.submit(**recording); !ok) return std::unexpected(ok.error());
    if (auto ok = context_.wait_on_host((*recording)->last_run()); !ok) return std::unexpected(ok.error());
  }
  auto recorder =
      Recorder::begin(context_, memory_,
                      {.sets = static_cast<std::uint32_t>(program.dispatches.size()) * repeats * recorded,
                       .buffers = buffers * repeats * recorded,
                       .images = 1});
  if (!recorder) return std::unexpected(recorder.error());
  const auto stamp = [&](std::uint32_t index) -> Done {
    if (!stamped) return {};
    return recorder->commands([&](VkCommandBuffer cb) {
      if (index == 0) queries.fn().reset(cb, queries.handle(), 0, queries.count());
      queries.fn().write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries.handle(), index);
    });
  };
  if (stamped) {
    if (auto ok = prepare(*recorder); !ok) return std::unexpected(ok.error());
  }
  if (auto ok = stamp(0); !ok) return std::unexpected(ok.error());
  std::vector<BufferView> views;
  for (std::uint32_t pass = 0; pass < recorded; ++pass) {
    for (std::uint32_t r = 0; r < repeats; ++r) {
      for (std::size_t k = 0; k < program.dispatches.size(); ++k) {
        const Dispatch& d = program.dispatches[k];
        views.clear();
        for (const Binding& b : d.bindings) views.push_back(view(b));
        if (auto ok =
                recorder->dispatch(*pipelines[k], views, {}, std::as_bytes(std::span(d.push)), d.groups);
            !ok) {
          return std::unexpected(ok.error());
        }
      }
    }
    if (auto ok = stamp(pass + 1); !ok) return std::unexpected(ok.error());
  }
  auto recording = recorder->finish();
  if (!recording) return std::unexpected(recording.error());

  std::vector<double> samples;
  if (stamped) {
    if (auto ok = context_.submit(**recording); !ok) return std::unexpected(ok.error());
    if (auto ok = context_.wait_on_host((*recording)->last_run()); !ok) return std::unexpected(ok.error());
    std::vector<std::uint64_t> ticks(queries.count());
    if (auto ok = check(queries.fn().results(context_.device(), queries.handle(), 0, queries.count(),
                                             ticks.size() * sizeof(std::uint64_t), ticks.data(),
                                             sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT),
                        "vkGetQueryPoolResults");
        !ok) {
      return std::unexpected(ok.error());
    }
    for (std::uint32_t pass = options.warmup; pass < passes; ++pass) {
      const double us = static_cast<double>(ticks[pass + 1] - ticks[pass]) * period_ns_ / 1000.0;
      samples.push_back(us / repeats);
    }
  } else {
    for (std::uint32_t pass = 0; pass < passes; ++pass) {
      const auto start = std::chrono::steady_clock::now();
      if (auto ok = context_.submit(**recording); !ok) return std::unexpected(ok.error());
      if (auto ok = context_.wait_on_host((*recording)->last_run()); !ok) return std::unexpected(ok.error());
      const auto host = std::chrono::steady_clock::now() - start;
      if (pass >= options.warmup)
        samples.push_back(std::chrono::duration<double, std::micro>(host).count() / repeats);
    }
  }
  Measured result{.choice = choice, .median_us = median(samples), .spread_us = deviation(samples)};
  if (const auto out = tensors.find(n.outputs[0]); out != tensors.end()) {
    std::vector<std::byte> bytes(ir::storage_bytes(graph.tensors[n.outputs[0]]));
    if (auto ok = download(context_, memory_, out->second, 0, bytes); !ok) return std::unexpected(ok.error());
    result.output = as_floats(bytes, graph.tensors[n.outputs[0]].type);
  }
  return result;
}

}  // namespace ajp::gpu::vk::net
