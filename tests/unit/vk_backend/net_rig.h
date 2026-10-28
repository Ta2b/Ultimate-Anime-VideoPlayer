// Banc des tests du moteur de réseaux : un appareil Vulkan créé (la RTX ici, lavapipe en CI),
// l'exécution d'un graphe gpu_ir abaissé, et sa comparaison avec l'interpréteur CPU FP32.
// Sans Vulkan, le test est sauté avec la raison ; tout autre refus le fait échouer.
#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/types.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/net_program.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"

namespace ajp::gpu::vk::test {

std::string describe(const Refusal& refusal);

template <class T>
T require_value(std::expected<T, Refusal> result) {
  if (!result) FAIL(describe(result.error()));
  return std::move(*result);
}

inline void require_ok(const std::expected<void, Refusal>& done) {
  if (!done) FAIL(describe(done.error()));
}

template <class T>
void require_refusal(const std::expected<T, Refusal>& result, RefusalCode code) {
  REQUIRE_FALSE(result.has_value());
  INFO(describe(result.error()));
  CHECK(result.error().code == code);
}

struct NetRig {
  std::unique_ptr<VkContext> context;
  std::unique_ptr<MemoryAllocator> memory;
  std::unique_ptr<PipelineCache> pipelines;
  net::Target target;
};

// Appareil du système avec les matrices coopératives activées s'il les a. Saute sans Vulkan.
NetRig make_net_rig();
// Saute sans stockage 16 bits.
void require_f16(const NetRig& rig);

// Le graphe exécuté sur l'appareil avec ces choix (ceux par défaut si vide), entrées dans
// l'ordre de graph.inputs ; sorties en float dans l'ordre de graph.outputs.
std::vector<ir::Value> run_on_device(NetRig& rig, const ir::Graph& graph,
                                     std::span<const ir::NamedValue> inputs,
                                     std::span<const net::KernelChoice> choices = {});

// L'interpréteur sur le même graphe.
std::vector<ir::Value> run_reference(const ir::Graph& graph, std::span<const ir::NamedValue> inputs);

// Écart toléré : |gpu − ref| ≤ atol + rtol × |ref| pour chaque élément ; le pire est rapporté.
struct Tolerance {
  float rtol = 1e-4F;
  float atol = 1e-5F;
};
inline constexpr Tolerance kFp32{.rtol = 1e-4F, .atol = 1e-5F};
// float16 : chaque tenseur rangé arrondit à 2^-11 près ; quelques arrondis s'accumulent.
inline constexpr Tolerance kFp16{.rtol = 8e-3F, .atol = 4e-3F};

void require_close(const ir::Value& got, const ir::Value& want, Tolerance tolerance, const std::string& what);

// Exécute et compare chaque sortie à l'interpréteur.
void check_graph(NetRig& rig, const ir::Graph& graph, std::span<const ir::NamedValue> inputs,
                 Tolerance tolerance, std::span<const net::KernelChoice> choices = {});

// Valeurs déterministes de la forme donnée, dans [lo, hi[, arrondies en float16 si type F16.
ir::Value test_values(ir::DataType type, const ir::Dims& dims, std::size_t seed, float lo = -1.0F,
                      float hi = 1.0F);

}  // namespace ajp::gpu::vk::test
