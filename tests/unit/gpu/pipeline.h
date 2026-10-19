// La chaîne complète sur le backend CPU, pas à pas et sans thread : un GpuWorker, des rings
// hôtes, des moteurs ONNX construits dans le test et un main qui rend ses slots aussitôt.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/thread.h"
#include "gpu/cpu/cpu_backend.h"
#include "gpu/program.h"
#include "gpu/worker.h"
#include "gpu_test_support.h"

namespace ajp::gpu::test {

class Pipeline {
 public:
  explicit Pipeline(std::size_t slots = 4) : slots_(slots) {
    sink.on_frame = [this](const RingFrame& frame) {
      const auto memory = rings.at(frame.generation)->pixels(frame.slot);
      pixels.emplace_back(memory.begin(), memory.end());
    };
  }

  // Ce que app enverrait par Install : le plan et les artefacts des modèles (F32 : les
  // comparaisons à la référence sont exactes).
  std::shared_ptr<const ProgramRequest> request(const core::Plan& plan) const {
    return std::make_shared<ProgramRequest>(plan, artifacts, TensorType::F32);
  }

  // Le Program chargé comme le ferait le thread gpu, pour un exécuteur sans travailleur.
  std::shared_ptr<const Program> program(const core::Plan& plan) {
    auto program = load_program(*request(plan), backend, backend.rife_input());
    REQUIRE(program);
    return *program;
  }

  // render crée le ring de cette génération à la taille du plan et l'envoie à gpu, puis app
  // envoie le Program.
  void install(const core::Plan& plan, std::uint32_t generation = 1) {
    auto& ring = rings[generation];
    if (!ring) {
      ring = std::make_unique<HostRing>(slots_, plan.output_width, plan.output_height, generation);
      worker.ring(RingGeneration{ring->slots()});
    }
    worker.install(Install{request(plan)});
    worker.pump();
  }

  void push(DecodedFrame frame) {
    REQUIRE(worker.frame(std::move(frame)));
    worker.pump();
    if (auto_release) release();
  }

  // main rend chaque slot reçu, jusqu'à ce que plus rien n'arrive.
  void release() {
    while (sink.unread > 0) {
      for (const SlotFree& free : sink.free_all()) worker.slot_free(free);
      worker.pump();
    }
  }

  bool auto_release = true;
  cpu::CpuBackend backend{BackendInit{}};
  TempDir dir{"pipeline"};
  std::vector<ModelArtifact> artifacts{
      {.id = "up2",
       .artifact = build_artifact(backend, write_model(dir.path(), "up2", upscale_x2_model()), "up2")},
      {.id = "rife",
       .artifact = build_artifact(backend, write_model(dir.path(), "rife", rife_blend_model()), "rife")},
      {.id = "motion",
       .artifact = build_artifact(backend, write_model(dir.path(), "motion", motion_model()), "motion")}};
  std::map<std::uint32_t, std::unique_ptr<HostRing>> rings;
  CollectingSink sink;
  std::vector<std::vector<std::uint16_t>> pixels;  // le slot copié à chaque réception
  std::vector<Refusal> refusals;
  GpuWorker worker{backend, backend.rife_input(), sink,
                   ThreadReport([this](const Refusal& refusal) { refusals.push_back(refusal); })};

 private:
  std::size_t slots_;
};

}  // namespace ajp::gpu::test
