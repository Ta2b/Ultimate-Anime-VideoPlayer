// Banc de l'étage gpu (upscale_stage.h) sans GPU : un backend factice dont les noyaux et les
// moteurs enregistrent leurs appels au lieu de calculer, les mesures de chaque paire dictées par
// le test, des images hôte vides, et une boîte de main qui garde les RingFrame reçus. Ce qui se
// teste : le nombre de sorties par image, leurs pts et leurs sources, coupes et dessins tenus,
// le facteur changé en direct, le seek, le protocole des slots et des sémaphores.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "app/play/pipeline.h"
#include "app/play/upscale_stage.h"
#include "app/play/video_stage.h"
#include "core/frame.h"
#include "core/ring.h"
#include "gpu/backend.h"
#include "gpu/kernel_ops.h"

namespace ajp::app::play::test {

// Un moteur nommé par ses entrées et sorties ; l'étage ne fait que l'enfiler.
class FakeEngine final : public gpu::Engine {
 public:
  FakeEngine(std::string name, std::vector<std::string> inputs, std::vector<std::string> outputs)
      : name_(std::move(name)), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}
  const std::string& name() const { return name_; }
  std::span<const std::string> input_names() const override { return inputs_; }
  std::span<const std::string> output_names() const override { return outputs_; }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

// Une écriture du ring, telle que le backend l'a reçue.
struct RingWritten {
  std::uint32_t slot = 0;
  std::uint64_t wait_value = 0;
  std::uint64_t signal_value = 0;
};

// Le backend et ses noyaux : chaque appel va dans le journal (« ingest 3 », « engine span »,
// « rife_input »…), les mesures d'une paire viennent du test par l'identité des images ingérées
// dans les tenseurs mesurés, tout le reste réussit sans rien calculer.
class FakeBackend final : public gpu::Backend, public gpu::KernelOps {
 public:
  // Mesures de la paire (left, right) ; sans entrée, une paire qui bouge (ni coupe ni dessin tenu).
  void score(SourceId left, SourceId right, double scene_score, std::uint32_t worst_block);
  // scores_start et scores_take offerts ; faux : refusés comme par un backend qui ne les a pas.
  void offer_ahead(bool on) { ahead_ = on; }
  // Le nombre de voies de calcul que disent les capacités ; 1 par défaut.
  void offer_lanes(int lanes) { lanes_ = lanes; }
  // Cette image est refusée à l'ingestion (VendorCallFailed).
  void refuse_ingest(SourceId id) { refused_ = id; }
  // Le mouvement maximal que rendent flow_max_now et flow_max_take, en pixels ; 0 par défaut.
  void motion_max(float pixels) {
    const std::scoped_lock lock(mutex_);
    motion_max_ = pixels;
  }

  std::vector<std::string> log() const;
  std::size_t count(std::string_view entry) const;  // entrées du journal égales à entry
  // Le même journal, chaque entrée précédée de la voie où elle a été enfilée (« measure ingest 3 »,
  // « upscale engine span », « interpolate rife_input »).
  std::vector<std::string> laned() const;
  // Le tenseur écrit par chaque appel de rife_crop ou rife_input, dans l'ordre.
  std::vector<std::uintptr_t> targets(std::string_view entry) const;
  std::vector<RingWritten> written() const;

  // gpu::Backend
  gpu::BackendCaps capabilities() const override {
    return {.name = "factice", .device_name = "aucun", .lanes = lanes_};
  }
  gpu::DeviceMatch match(const DeviceUuid& /*render_device*/) const override {
    return gpu::DeviceMatch::Same;
  }
  std::expected<gpu::Artifact, Refusal> build(const gpu::ModelSpec& model, const gpu::ShapeSet& /*shapes*/,
                                              gpu::Progress& /*progress*/) override;
  std::expected<std::unique_ptr<gpu::Engine>, Refusal> load(const gpu::Artifact& artifact) override;
  std::expected<gpu::Tensor, Refusal> import(const gpu::ExternalImage& /*image*/) override;
  std::expected<std::unique_ptr<gpu::TensorBuffer>, Refusal> alloc_tensor(gpu::TensorType type,
                                                                          const gpu::Shape& shape) override;
  gpu::KernelOps* kernels() override { return this; }
  void use_lane(gpu::Lane lane) override {
    const std::scoped_lock lock(mutex_);
    lane_ = lane;
  }
  std::expected<void, Refusal> enqueue(gpu::Engine& engine, std::span<const gpu::Tensor> inputs,
                                       std::span<gpu::Tensor> outputs) override;
  std::expected<gpu::Fence, Refusal> signal() override { return gpu::Fence{.handle = 1, .value = ++fence_}; }
  std::expected<void, Refusal> wait(const gpu::Fence& /*fence*/) override { return {}; }

  // gpu::KernelOps
  std::expected<void, Refusal> ingest(const DecodedFrame& frame, const gpu::IngestSpec& spec,
                                      const gpu::Tensor& rgb) override;
  std::expected<void, Refusal> resample(const gpu::Tensor& /*in*/, kernels::Filter /*filter*/,
                                        const gpu::Tensor& /*out*/) override {
    return note("resample");
  }
  std::expected<void, Refusal> rife_pad(const DecodedFrame& /*frame*/,
                                        const gpu::Tensor& /*padded*/) override {
    return note("rife_pad");
  }
  std::expected<void, Refusal> rife_crop(const gpu::Tensor& /*padded*/, const gpu::Tensor& out) override {
    return note("rife_crop", out.handle);
  }
  std::expected<void, Refusal> warp_blend(const gpu::Tensor& /*a*/, const gpu::Tensor& /*b*/,
                                          const gpu::Motion& /*motion*/, float /*t*/,
                                          const gpu::Tensor& /*out*/) override {
    return note("warp_blend");
  }
  std::expected<void, Refusal> write_ring(const gpu::Tensor& rgb, const RingSlotDesc& slot,
                                          const gpu::RingWrite& sync) override;
  std::expected<void, Refusal> measure(const DecodedFrame& /*a*/, const DecodedFrame& /*b*/) override {
    return note("measure");
  }
  std::optional<gpu::PairScores> previous_scores() override { return std::nullopt; }
  std::expected<void, Refusal> rife_input(const gpu::Tensor& /*a*/, const gpu::Tensor& /*b*/, float /*t*/,
                                          const gpu::Tensor& input) override {
    return note("rife_input", input.handle);
  }
  std::expected<void, Refusal> rife_time(const gpu::Tensor& /*input*/, float /*t*/) override {
    return note("rife_time");
  }
  std::expected<gpu::PairScores, Refusal> scores_now(const gpu::Tensor& a, const gpu::Tensor& b) override;
  std::expected<std::uint64_t, Refusal> scores_start(const gpu::Tensor& a, const gpu::Tensor& b) override;
  std::expected<gpu::PairScores, Refusal> scores_take(std::uint64_t ticket) override;
  std::expected<float, Refusal> flow_max_now(const gpu::Motion& /*motion*/) override {
    const std::scoped_lock lock(mutex_);
    push("flow_max_now");
    return motion_max_;
  }
  std::expected<std::uint64_t, Refusal> flow_max_start(const gpu::Motion& /*motion*/) override {
    const std::scoped_lock lock(mutex_);
    if (!ahead_) return std::unexpected(Refusal{RefusalCode::FormatUnsupported, "flow_max_start : absent"});
    push("flow_max_start");
    flow_tickets_[next_ticket_] = motion_max_;
    return next_ticket_++;
  }
  std::expected<float, Refusal> flow_max_take(std::uint64_t ticket) override {
    const std::scoped_lock lock(mutex_);
    const auto found = flow_tickets_.find(ticket);
    if (found == flow_tickets_.end())
      return std::unexpected(Refusal{RefusalCode::SequenceOrder, "flow_max_take : ticket inconnu"});
    const float value = found->second;
    flow_tickets_.erase(found);
    push("flow_max_take");
    return value;
  }
  std::size_t pending_flow_tickets() const {
    const std::scoped_lock lock(mutex_);
    return flow_tickets_.size();
  }
  std::expected<void, Refusal> warp_blend_poses(const gpu::Tensor& /*a*/, const gpu::Tensor& /*b*/,
                                                const gpu::Motion& /*motion*/,
                                                const gpu::Motion* /*previous*/, float /*t*/,
                                                const gpu::Tensor& /*out*/) override {
    return note("warp_blend_poses");
  }
  std::expected<void, Refusal> gmfss_frames(const gpu::Tensor& /*a*/, const gpu::Tensor& /*b*/,
                                            const gpu::Tensor& /*frames*/) override {
    return note("gmfss_frames");
  }
  std::expected<void, Refusal> soft_splat(const gpu::Tensor& /*source*/, const gpu::Tensor& /*motion*/,
                                          float /*s*/, const gpu::Tensor& /*out*/) override {
    return note("soft_splat");
  }
  std::expected<void, Refusal> soft_splat_all(std::span<const SplatJob> /*jobs*/) override {
    return note("soft_splat_all");
  }
  std::expected<void, Refusal> gmfss_compose(const gpu::Tensor& /*high_a*/, const gpu::Tensor& /*high_b*/,
                                             const gpu::Tensor& /*low_a*/, const gpu::Tensor& /*low_b*/,
                                             const gpu::Tensor& /*fusion*/, const kernels::Window& /*window*/,
                                             float /*t*/, const gpu::Tensor& /*out*/) override {
    return note("gmfss_compose");
  }
  std::expected<void, Refusal> fill(const gpu::Tensor& /*tensor*/, float /*value*/) override {
    return note("fill");
  }

 private:
  using PairKey = std::pair<std::uint32_t, std::uint32_t>;
  std::expected<void, Refusal> note(std::string entry, std::uintptr_t target = 0);
  void push(std::string entry);  // sous le verrou
  gpu::PairScores scores_of(const gpu::Tensor& a, const gpu::Tensor& b);

  mutable std::mutex mutex_;
  std::vector<std::string> log_;
  std::vector<std::string> laned_;
  std::map<std::string, std::vector<std::uintptr_t>, std::less<>> targets_;
  int lanes_ = 1;
  gpu::Lane lane_ = gpu::Lane::Upscale;
  std::vector<RingWritten> written_;
  std::map<std::uintptr_t, SourceId> ingested_;  // tenseur → dernière image ingérée dedans
  std::map<PairKey, gpu::PairScores> scores_;
  std::map<std::uint64_t, gpu::PairScores> tickets_;
  std::map<std::uint64_t, float> flow_tickets_;
  float motion_max_ = 0.0F;
  std::uintptr_t next_handle_ = 1;
  std::uint64_t next_ticket_ = 1;
  std::uint64_t fence_ = 0;
  bool ahead_ = true;
  std::optional<SourceId> refused_;
};

// La boîte de main : les RingFrame dans l'ordre, et l'attente qu'il en arrive n.
class FrameRecord final : public RingSink {
 public:
  bool try_push(const RingFrame& frame) override;
  // Vrai dès que n images sont arrivées ; faux après le délai.
  bool wait_for(std::size_t n, std::chrono::milliseconds timeout) const;
  std::vector<RingFrame> frames() const;

 private:
  mutable std::mutex mutex_;
  mutable std::condition_variable arrived_;
  std::vector<RingFrame> frames_;
};

// Une image hôte vide de la taille donnée : le backend factice n'en lit pas les pixels.
DecodedFrame host_frame(std::uint32_t id, int width, int height, std::int64_t pts_us,
                        std::uint32_t epoch = 0);

// L'étage et ce qui l'entoure : le backend, la boîte de main, les refus rapportés, les crédits
// rendus au thread source.
struct StageRig {
  FakeBackend backend;
  FrameRecord sink;
  std::vector<Refusal> refusals;  // sous le verrou de refusals_mutex
  std::mutex refusals_mutex;
  std::atomic<std::uint32_t> credits{0};
  std::unique_ptr<VideoStage> stage;

  VideoStageInit init(const MasterClock* clock = nullptr);
  static std::vector<RingSlotDesc> slots(std::uint32_t count, int width, int height);
  // Pousse l'image, en réessayant tant que l'étage est plein ; faux après le délai.
  bool push(DecodedFrame frame, std::chrono::milliseconds timeout = std::chrono::seconds(2));
  // Attend n images dans la boîte, en rendant chaque slot reçu dès son arrivée si release.
  bool collect(std::size_t n, bool release = true,
               std::chrono::milliseconds timeout = std::chrono::seconds(5));
  // Rend les slots des images reçues et pas encore rendues ; combien.
  std::size_t release_all();
  std::vector<Refusal> reported();

 private:
  std::size_t released_ = 0;  // images de la boîte dont le slot est rendu
};

}  // namespace ajp::app::play::test
