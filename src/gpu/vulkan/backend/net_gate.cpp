#include "gpu/vulkan/backend/net_gate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <ranges>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_engine.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/compute/buffer.h"
#include "kernels/ref/half.h"
#include "kernels/spec/rife.h"

namespace ajp::gpu::vk::net {
namespace {

// Le motif d'entrée dans le type du tenseur, ramené à [0, 1] comme une image.
std::vector<std::byte> input_bytes(const ir::Tensor& tensor) {
  const std::uint64_t bytes = ir::storage_bytes(tensor);
  std::vector<std::byte> out(bytes, std::byte{0});
  if (tensor.type == ir::DataType::F16) {
    for (std::uint64_t i = 0; i < bytes / 2; ++i) {
      const std::uint16_t h = kernels::ref::float_to_half(0.5F * (tuning_input(tensor.type, i) + 1.0F));
      std::memcpy(out.data() + (2 * i), &h, 2);
    }
  } else if (tensor.type == ir::DataType::F32) {
    for (std::uint64_t i = 0; i < bytes / 4; ++i) {
      const float v = 0.5F * (tuning_input(tensor.type, i) + 1.0F);
      std::memcpy(out.data() + (4 * i), &v, 4);
    }
  }
  return out;
}

// Le plan, la ligne et la colonne d'un élément d'une entrée [N, C, H, W].
struct Place {
  std::int64_t c = 0;
  std::int64_t y = 0;
  std::int64_t x = 0;
};

// Un dessin, en quatre sortes de zones de 128 pixels qui se suivent (B45-02), pour que le jugement ne tienne
// pas à un seul motif : (0) aplats de 32 pixels dont un sur deux est sombre, dégradé lent par-dessus, traits
// sombres d'un pixel ; (1) un long dégradé de ciel en triangle (les bandes que les sommes réduites creusent)
// ; (2) des traits obliques fins sur un fond clair ; (3) un grain fin autour d'un gris moyen. Les aplats
// sombres et les traits fins sont ce qu'une somme en float16 abîme d'abord (A45-02). Une entrée de moins de
// 128 pixels de côté n'a que la zone 0.
float drawing(Place at) {
  std::uint64_t z = (static_cast<std::uint64_t>(at.c) * 0x9E3779B97F4A7C15ULL) ^
                    (static_cast<std::uint64_t>(at.y / 32) * 0xBF58476D1CE4E5B9ULL) ^
                    (static_cast<std::uint64_t>(at.x / 32) * 0x94D049BB133111EBULL);
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z ^= z >> 27U;
  const std::int64_t zone = ((at.x / 128) + (at.y / 128)) % 4;
  if (zone == 1) {
    const float t = static_cast<float>((at.x + (2 * at.y)) % 2048) / 2048.0F;
    return 0.5F + (0.4F * (1.0F - std::abs((2.0F * t) - 1.0F)));
  }
  if (zone == 2) {
    const bool line = (at.x + at.y) % 9 == 0 || (at.x - at.y + 4096) % 13 == 0;
    return line ? 0.1F : 0.85F;
  }
  if (zone == 3) {
    std::uint64_t g = (static_cast<std::uint64_t>(at.x) * 0x9E3779B97F4A7C15ULL) ^
                      (static_cast<std::uint64_t>(at.y) * 0xBF58476D1CE4E5B9ULL) ^
                      (static_cast<std::uint64_t>(at.c) * 0x94D049BB133111EBULL);
    g = (g ^ (g >> 31U)) * 0x94D049BB133111EBULL;
    g ^= g >> 29U;
    return 0.5F + (0.06F * ((static_cast<float>(g & 0xFFU) / 255.0F) - 0.5F));
  }
  float level = static_cast<float>(z & 0xFFU) / 255.0F;
  if (((at.x / 32) + (at.y / 32)) % 2 == 0) level *= 0.15F;
  const float shade = 0.08F * static_cast<float>((at.x + (2 * at.y)) % 256) / 255.0F;
  const bool line = (at.x + (3 * at.y)) % 41 == 0 || (at.y % 53) == 7;
  return line ? 0.04F : std::min(1.0F, (0.9F * level) + shade);
}

// La valeur de l'entrée à cet élément. 7 ou 11 plans : l'entrée d'un réseau d'interpolation, deux images dont
// la seconde a glissé de deux pixels, l'instant, puis la grille du v1 (kernels/spec/rife.h) ; sans elle, les
// warps du réseau liraient n'importe où et la moindre erreur d'arrondi déplacerait toute la sortie.
float gate_value(const ir::Dims& dims, std::uint64_t index) {
  const std::int64_t w = dims[3];
  const std::int64_t h = dims[2];
  const Place at{.c = static_cast<std::int64_t>(index / static_cast<std::uint64_t>(h * w)) % dims[1],
                 .y = static_cast<std::int64_t>(index / static_cast<std::uint64_t>(w)) % h,
                 .x = static_cast<std::int64_t>(index % static_cast<std::uint64_t>(w))};
  if (dims[1] != 7 && dims[1] != 11) return drawing(at);
  if (at.c < 3) return drawing(at);
  if (at.c < 6) return drawing({.c = at.c - 3, .y = at.y, .x = at.x + 2});
  switch (at.c) {
    case 6:
      return dims[1] == 11 ? 0.5F : 0.25F;
    case 7:
      return kernels::rife_mesh(static_cast<int>(at.x), static_cast<int>(w));
    case 8:
      return kernels::rife_mesh(static_cast<int>(at.y), static_cast<int>(h));
    case 9:
      return kernels::rife_mesh_step(static_cast<int>(w));
    default:
      return kernels::rife_mesh_step(static_cast<int>(h));
  }
}

// L'entrée de la porte dans le type du tenseur, arrondie au float16 : le réseau en float16 et son jumeau en
// float lisent les mêmes valeurs. Hors [N, C, H, W] : le motif de tuning_input ramené à [0, 1].
std::vector<std::byte> gate_bytes(const ir::Tensor& tensor) {
  if (tensor.dims.size() != 4 || tensor.layout != ir::Layout::Nchw || tensor.dims[2] < 1 ||
      tensor.dims[3] < 1)
    return input_bytes(tensor);
  const std::uint64_t bytes = ir::storage_bytes(tensor);
  std::vector<std::byte> out(bytes, std::byte{0});
  const std::uint64_t size = tensor.type == ir::DataType::F16 ? 2 : 4;
  if (tensor.type != ir::DataType::F16 && tensor.type != ir::DataType::F32) return out;
  for (std::uint64_t i = 0; i < bytes / size; ++i) {
    const std::uint16_t h = kernels::ref::float_to_half(gate_value(tensor.dims, i));
    const float v = kernels::ref::half_to_float(h);
    if (size == 2) {
      std::memcpy(out.data() + (2 * i), &h, 2);
    } else {
      std::memcpy(out.data() + (4 * i), &v, 4);
    }
  }
  return out;
}

double value_at(const std::byte* data, ir::DataType type, std::uint64_t i) {
  if (type == ir::DataType::F16) {
    std::uint16_t h = 0;
    std::memcpy(&h, data + (2 * i), 2);
    return kernels::ref::half_to_float(h);
  }
  float v = 0.0F;
  std::memcpy(&v, data + (4 * i), 4);
  return v;
}

struct Outputs {
  std::vector<Buffer> buffers;
};

// Le réseau aux choix donnés, une fois, sur les entrées : ses sorties, lisibles par l'hôte.
std::expected<Outputs, Refusal> run_once(VkContext& context, MemoryAllocator& memory,
                                         PipelineCache& pipelines, const ir::Graph& graph,
                                         std::span<const KernelChoice> choices, const Target& target,
                                         std::span<const Buffer> inputs) {
  auto program = lower(graph, choices, target);
  if (!program) return std::unexpected(program.error());
  auto engine = NetEngine::create(context, memory, pipelines, graph, std::move(*program));
  if (!engine) return std::unexpected(engine.error());
  Outputs out;
  std::vector<BufferView> in_views;
  std::vector<BufferView> out_views;
  for (const Buffer& b : inputs) in_views.push_back(b.view());
  for (std::size_t k = 0; k < graph.outputs.size(); ++k) {
    auto buffer = Buffer::create(context, memory, std::max<std::uint64_t>((*engine)->output_bytes(k), 4),
                                 MemoryUse::Readback);
    if (!buffer) return std::unexpected(buffer.error());
    out_views.push_back(buffer->view());
    out.buffers.push_back(std::move(*buffer));
  }
  auto recording = (*engine)->recording(in_views, out_views);
  if (!recording) return std::unexpected(recording.error());
  if (auto ok = context.submit(**recording); !ok) return std::unexpected(ok.error());
  if (auto ok = context.wait_on_host((*recording)->last_run()); !ok) return std::unexpected(ok.error());
  return out;
}

std::expected<std::vector<Buffer>, Refusal> upload_inputs(VkContext& context, MemoryAllocator& memory,
                                                          const ir::Graph& graph, bool drawn) {
  std::vector<Buffer> inputs;
  for (const ir::TensorId id : graph.inputs) {
    const std::vector<std::byte> bytes =
        drawn ? gate_bytes(graph.tensors[id]) : input_bytes(graph.tensors[id]);
    auto buffer = Buffer::create(context, memory, std::max<std::size_t>(bytes.size(), 4), MemoryUse::Device);
    if (!buffer) return std::unexpected(buffer.error());
    if (auto ok = upload(context, memory, *buffer, 0, bytes); !ok) return std::unexpected(ok.error());
    inputs.push_back(std::move(*buffer));
  }
  return inputs;
}

// Le PSNR des sorties `got` (graphe `graph`) contre les sorties `truth` du jumeau en float, en dB, le pic
// étant la plus grande valeur absolue de la vérité. −∞ pour une valeur non finie, kGateExactDb sans écart.
std::expected<double, Refusal> fidelity_db(const ir::Graph& graph, const Outputs& got, const ir::Graph& twin,
                                           const Outputs& truth) {
  double diff = 0.0;
  double peak = 0.0;
  std::uint64_t total = 0;
  for (std::size_t k = 0; k < graph.outputs.size(); ++k) {
    const ir::Tensor& t = graph.tensors[graph.outputs[k]];
    const ir::Tensor& u = twin.tensors[twin.outputs[k]];
    if (t.type != ir::DataType::F16 && t.type != ir::DataType::F32) continue;
    const std::uint64_t count = ir::storage_bytes(t) / (t.type == ir::DataType::F16 ? 2 : 4);
    const std::byte* pa = got.buffers[k].mapped();
    const std::byte* pb = truth.buffers[k].mapped();
    if (pa == nullptr || pb == nullptr) return refuse(RefusalCode::InputMismatch, "sortie non lisible");
    for (std::uint64_t i = 0; i < count; ++i) {
      const double va = value_at(pa, t.type, i);
      const double vb = value_at(pb, u.type, i);
      if (!std::isfinite(va) || !std::isfinite(vb)) return -std::numeric_limits<double>::infinity();
      diff += (va - vb) * (va - vb);
      peak = std::max(peak, std::abs(vb));
    }
    total += count;
  }
  if (diff <= 0.0 || total == 0) return kGateExactDb;
  // Une vérité toute nulle avec un écart : rien à comparer en dB, le pire score fini (la valeur reste
  // jugeable).
  if (peak <= 0.0) return -kGateExactDb;
  return std::min(kGateExactDb, 10.0 * std::log10(peak * peak * static_cast<double>(total) / diff));
}

// La porte d'avant (écart quadratique aux noyaux de référence en float16, sur du bruit), quand le jumeau en
// float ne peut pas être joué.
std::expected<Gated, Refusal> gate_on_reference(VkContext& context, MemoryAllocator& memory,
                                                PipelineCache& pipelines, const ir::Graph& graph, Gated gated,
                                                std::vector<KernelChoice> reference, const Target& target) {
  auto drift = network_drift(context, memory, pipelines, graph, gated.choices, reference, target);
  if (!drift) return std::unexpected(drift.error());
  gated.drift = *drift;
  const bool low = std::ranges::any_of(gated.choices, low_precision);
  if (*drift <= (low ? kNetworkDriftLow : kNetworkDrift)) return gated;
  if (low) {
    std::vector<KernelChoice> wider = full_precision(gated.choices, reference);
    // Un réglage que les sommes en float ne prennent pas (refus de l'abaissement) : la référence partout.
    auto again = network_drift(context, memory, pipelines, graph, wider, reference, target);
    if (again && *again <= kNetworkDrift) {
      gated.choices = std::move(wider);
      gated.fallback = std::format(
          "précision pleine à la place de la précision réduite (écart {:.4f} puis {:.4f})", *drift, *again);
      return gated;
    }
  }
  gated.choices = std::move(reference);
  gated.fallback = std::format("noyaux de référence partout (écart {:.4f})", *drift);
  return gated;
}

}  // namespace

ir::Graph float_twin(const ir::Graph& graph) {
  ir::Graph twin = graph;
  for (ir::Tensor& t : twin.tensors) {
    if (t.type == ir::DataType::F16) t.type = ir::DataType::F32;
    if (t.value.type == ir::DataType::F16) t.value.type = ir::DataType::F32;
  }
  return twin;
}

std::vector<KernelChoice> full_precision(std::span<const KernelChoice> choices,
                                         std::span<const KernelChoice> reference) {
  std::vector<KernelChoice> wider(choices.begin(), choices.end());
  for (std::size_t k = 0; k < wider.size(); ++k) {
    if (wider[k].variant == "coopmat_acc16") {
      wider[k].variant = "coopmat";
    } else if (low_precision(wider[k])) {
      wider[k] = reference[k];
    }
  }
  return wider;
}

std::expected<std::vector<KernelChoice>, Refusal> reference_choices(const ir::Graph& graph,
                                                                    const Target& target) {
  std::vector<KernelChoice> out;
  out.reserve(graph.nodes.size());
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    auto options = candidates(graph, k, target);
    if (!options) return std::unexpected(options.error());
    const auto direct = std::ranges::find(*options, std::string("direct"), &KernelChoice::variant);
    out.push_back(direct != options->end() ? *direct : options->front());
  }
  return out;
}

std::expected<double, Refusal> network_drift(VkContext& context, MemoryAllocator& memory,
                                             PipelineCache& pipelines, const ir::Graph& graph,
                                             std::span<const KernelChoice> tuned,
                                             std::span<const KernelChoice> reference, const Target& target) {
  auto uploaded = upload_inputs(context, memory, graph, false);
  if (!uploaded) return std::unexpected(uploaded.error());
  const std::vector<Buffer>& inputs = *uploaded;
  auto a = run_once(context, memory, pipelines, graph, tuned, target, inputs);
  if (!a) return std::unexpected(a.error());
  auto b = run_once(context, memory, pipelines, graph, reference, target, inputs);
  if (!b) return std::unexpected(b.error());
  double diff = 0.0;
  double energy = 0.0;
  for (std::size_t k = 0; k < graph.outputs.size(); ++k) {
    const ir::Tensor& t = graph.tensors[graph.outputs[k]];
    if (t.type != ir::DataType::F16 && t.type != ir::DataType::F32) continue;
    const std::uint64_t count = ir::storage_bytes(t) / (t.type == ir::DataType::F16 ? 2 : 4);
    const std::byte* pa = a->buffers[k].mapped();
    const std::byte* pb = b->buffers[k].mapped();
    if (pa == nullptr || pb == nullptr) return refuse(RefusalCode::InputMismatch, "sortie non lisible");
    for (std::uint64_t i = 0; i < count; ++i) {
      const double va = value_at(pa, t.type, i);
      const double vb = value_at(pb, t.type, i);
      if (!std::isfinite(va)) return std::numeric_limits<double>::infinity();
      diff += (va - vb) * (va - vb);
      energy += vb * vb;
    }
  }
  if (energy <= 0.0) return diff > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
  return std::sqrt(diff / energy);
}

std::expected<Fidelity, Refusal> network_fidelity(VkContext& context, MemoryAllocator& memory,
                                                  PipelineCache& pipelines, const ir::Graph& graph,
                                                  std::span<const std::vector<KernelChoice>> sets,
                                                  const Target& target, const GateWatch& watch) {
  // Un arrêt demandé se voit avant chaque rejeu (C45-02) : le jumeau en float, puis un jeu de choix après
  // l'autre.
  const auto cancelled = [&]() -> std::optional<Refusal> {
    if (watch.stop.stop_requested())
      return Refusal{RefusalCode::Cancelled, "construction arrêtée pendant la porte"};
    return std::nullopt;
  };
  const auto report = [&](std::size_t done, const std::string& stage) {
    if (!watch.report) return;
    const float share = static_cast<float>(done) / static_cast<float>(sets.size() + 1);
    watch.report(watch.from + ((watch.to - watch.from) * share), stage);
  };
  if (auto stop = cancelled()) return std::unexpected(*stop);
  const ir::Graph twin = float_twin(graph);
  auto twin_reference = reference_choices(twin, target);
  if (!twin_reference) return std::unexpected(twin_reference.error());
  auto twin_inputs = upload_inputs(context, memory, twin, true);
  if (!twin_inputs) return std::unexpected(twin_inputs.error());
  auto truth = run_once(context, memory, pipelines, twin, *twin_reference, target, *twin_inputs);
  if (!truth) return std::unexpected(truth.error());
  twin_inputs->clear();
  report(1, "porte : jumeau en float");
  auto inputs = upload_inputs(context, memory, graph, true);
  if (!inputs) return std::unexpected(inputs.error());
  Fidelity out;
  for (const std::vector<KernelChoice>& choices : sets) {
    if (auto stop = cancelled()) return std::unexpected(*stop);
    auto got = run_once(context, memory, pipelines, graph, choices, target, *inputs);
    if (!got) return std::unexpected(got.error());
    auto db = fidelity_db(graph, *got, twin, *truth);
    if (!db) return std::unexpected(db.error());
    out.db.push_back(*db);
    report(out.db.size() + 1, std::format("porte : {} / {}", out.db.size(), sets.size()));
  }
  return out;
}

std::expected<Gated, Refusal> gate_network(VkContext& context, MemoryAllocator& memory,
                                           PipelineCache& pipelines, const ir::Graph& graph,
                                           std::vector<KernelChoice> tuned, const Target& target,
                                           const GateWatch& watch) {
  auto reference = reference_choices(graph, target);
  if (!reference) return std::unexpected(reference.error());
  Gated gated{.choices = std::move(tuned)};
  if (gated.choices == *reference) return gated;
  const bool low = std::ranges::any_of(gated.choices, low_precision);
  std::vector<std::vector<KernelChoice>> sets{*reference, gated.choices};
  if (low) sets.push_back(full_precision(gated.choices, *reference));
  auto fidelity = network_fidelity(context, memory, pipelines, graph, sets, target, watch);
  // Un arrêt n'est pas un jumeau injouable : il traverse la porte (C45-02).
  if (!fidelity && fidelity.error().code == RefusalCode::Cancelled) return std::unexpected(fidelity.error());
  // Le jumeau en float ne se joue pas (mémoire, opérateur sans noyau float) : la porte d'avant. Un réglage
  // que les sommes en float ne prennent pas est écarté de la liste et rejoué sans lui.
  if (!fidelity && low) {
    sets.pop_back();
    fidelity = network_fidelity(context, memory, pipelines, graph, sets, target, watch);
    if (!fidelity && fidelity.error().code == RefusalCode::Cancelled)
      return std::unexpected(fidelity.error());
  }
  if (!fidelity) {
    gated.unjudged = fidelity.error().detail;
    return gate_on_reference(context, memory, pipelines, graph, std::move(gated), std::move(*reference),
                             target);
  }
  gated.against_float = true;
  gated.floor_db = fidelity->db[0];
  // C45-04 : un jumeau ou des noyaux de référence qui sortent une valeur non finie (une constante qui déborde
  // en float16, un réseau qui diverge) ne jugent rien : -∞ pour tous les jeux passait pour « aucune perte ».
  // Refus.
  if (!std::isfinite(gated.floor_db)) {
    return refuse(RefusalCode::GraphInvalid,
                  "le calcul de référence sort une valeur non finie : le réseau ne peut pas être jugé");
  }
  gated.db = fidelity->db[1];
  const std::optional<double> wider = sets.size() == 3 ? std::optional(fidelity->db[2]) : std::nullopt;
  const GateVerdict verdict = gate_verdict(gated.floor_db, gated.db, wider);
  if (verdict == GateVerdict::Keep) return gated;
  if (verdict == GateVerdict::FullPrecision) {
    gated.choices = std::move(sets[2]);
    gated.fallback = std::format(
        "précision pleine à la place de la précision réduite ({:.1f} dB puis {:.1f} dB "
        "contre le calcul en float, {:.1f} dB aux noyaux de référence)",
        gated.db, *wider, gated.floor_db);
    return gated;
  }
  gated.choices = std::move(*reference);
  gated.fallback = std::format(
      "noyaux de référence partout ({:.1f} dB contre le calcul en float, {:.1f} dB "
      "aux noyaux de référence)",
      gated.db, gated.floor_db);
  return gated;
}

namespace {

// Ce que les relevés de l'autotuner disent des couches gardées à précision réduite.
struct ReducedSummary {
  std::size_t layers = 0;
  std::size_t timed = 0;  // celles dont les deux variantes ont été mesurées
  double chosen_us = 0.0;
  double full_us = 0.0;
  bool gain_measured() const { return timed > 0 && chosen_us > 0.0; }
  double gain_percent() const { return ((full_us / chosen_us) - 1.0) * 100.0; }
};

ReducedSummary summarize_reduced(std::span<const KernelChoice> choices,
                                 std::span<const ReducedChoice> measured) {
  ReducedSummary sum;
  sum.layers = static_cast<std::size_t>(std::ranges::count_if(choices, low_precision));
  if (sum.layers == 0) return sum;
  // Le gain n'est chiffré que pour les couches dont l'autotuner a mesuré les deux variantes ; un relevé dont
  // le choix a été repris ensuite par la porte (autre variante au même nœud) ne compte pas.
  for (const ReducedChoice& r : measured) {
    if (r.node >= choices.size() || choices[r.node].variant != r.variant || !r.full_precision_us) continue;
    sum.chosen_us += r.chosen_us;
    sum.full_us += *r.full_precision_us;
    ++sum.timed;
  }
  return sum;
}

}  // namespace

PrecisionNotice precision_notice(std::span<const KernelChoice> choices, const Gated& gated,
                                 std::span<const ReducedChoice> measured) {
  const ReducedSummary sum = summarize_reduced(choices, measured);
  PrecisionNotice notice{.layers = static_cast<std::uint32_t>(sum.layers)};
  if (sum.layers == 0) return notice;
  notice.gain_measured = sum.gain_measured();
  if (notice.gain_measured) notice.gain_percent = static_cast<float>(sum.gain_percent());
  notice.loss_measured = gated.against_float;
  if (notice.loss_measured) notice.loss_db = static_cast<float>(gated.db);
  return notice;
}

std::string reduced_precision_notice(std::string_view model, std::span<const KernelChoice> choices,
                                     const Gated& gated, std::span<const ReducedChoice> measured) {
  const ReducedSummary sum = summarize_reduced(choices, measured);
  if (sum.layers == 0) return {};
  std::string gain = "gain de vitesse non mesuré (choix repris du cache)";
  if (sum.gain_measured()) {
    gain = std::format(
        "gain mesuré {:+.0f} % de vitesse sur {} couche(s) sur {} ({:.0f} µs contre {:.0f} µs en "
        "sommes float)",
        sum.gain_percent(), sum.timed, sum.layers, sum.chosen_us, sum.full_us);
  }
  const std::string loss =
      gated.against_float
          ? std::format(
                "{:.1f} dB contre le calcul en float, {:+.2f} dB face aux noyaux de référence ({:.1f} dB)",
                gated.db, gated.db - gated.floor_db, gated.floor_db)
          : std::string("perte non mesurée (le calcul en float n'a pas pu être joué)");
  return std::format("{} : précision réduite sur {} couche(s) pour gagner en vitesse : {} ; {}", model,
                     sum.layers, loss, gain);
}

}  // namespace ajp::gpu::vk::net
