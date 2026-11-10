// Backend TensorRT (gpu/backend.h) : moteurs construits par trtexec (prepare), exécutés sur les
// flux CUDA non bloquants du contexte primaire de la RTX du rendu, un par voie de calcul
// (gpu/lanes.h), et les noyaux de la chaîne que la tranche verticale demande : ingestion des
// images NVDEC et écriture du ring importé de Vulkan.
//
// Chargé à l'exécution (ajp_backend_tensorrt.so) : le lecteur démarre sans CUDA ni TensorRT.
// Les noyaux et l'enveloppe TensorRT sont provisoires (slice_kernels.h, trt_engine.h) : ils
// laissent la place au socle CUDA (pc/cuda-compute) et à l'enveloppe TensorRT (pc/tensorrt).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/cuda/backend/gpu_timing.h"
#include "gpu/cuda/backend/host_upload.h"
#include "gpu/cuda/backend/ring_slot.h"
#include "gpu/kernel_ops.h"
#include "gpu/lanes.h"
#include "prepare/builder.h"

namespace nvinfer1 {
class IRuntime;
}

namespace ajp::gpu::cuda {

class TrtLog;
class SharedActivations;
class SpanFp8Engine;
class RifeFusedEngine;
class DrbaFusedEngine;

class TensorRtBackend final : public Backend, public KernelOps {
 public:
  // trtexec : le constructeur de moteurs livré avec le backend.
  static std::expected<std::unique_ptr<TensorRtBackend>, Refusal> create(const BackendInit& init,
                                                                         std::filesystem::path trtexec);
  ~TensorRtBackend() override;
  TensorRtBackend(const TensorRtBackend&) = delete;
  TensorRtBackend& operator=(const TensorRtBackend&) = delete;

  // Backend
  BackendCaps capabilities() const override;
  DeviceMatch match(const DeviceUuid& render_device) const override;
  std::expected<Artifact, Refusal> build(const ModelSpec& model, const ShapeSet& shapes,
                                         Progress& progress) override;
  std::expected<std::unique_ptr<Engine>, Refusal> load(const Artifact& artifact) override;
  std::expected<Tensor, Refusal> import(const ExternalImage& image) override;
  std::expected<std::unique_ptr<TensorBuffer>, Refusal> alloc_tensor(TensorType type,
                                                                     const Shape& shape) override;
  KernelOps* kernels() override { return this; }
  std::expected<void, Refusal> enqueue(Engine& engine, std::span<const Tensor> inputs,
                                       std::span<Tensor> outputs) override;
  std::expected<Fence, Refusal> signal() override;
  std::expected<void, Refusal> wait(const Fence& fence) override;
  void use_lane(Lane lane) override;  // tensorrt_lanes.cpp
  void drain() override;              // tensorrt_lanes.cpp
  // Dérivation du paquet « .span8 » (tensorrt_span_derive.cpp).
  std::expected<SpanFp8Derivation, Refusal> derive_span_fp8(const SpanFp8Request& request,
                                                            Progress& progress) override;
  // Un SpanFp8Engine écrit l'emplacement depuis l'épilogue de to_pixel (tensorrt_span_fp8.cpp) ; tout autre
  // moteur prend le chemin commun (enqueue puis write_ring).
  std::expected<void, Refusal> enqueue_ring(Engine& engine, std::span<const Tensor> inputs,
                                            std::span<Tensor> outputs, const RingTarget& target) override;

  // KernelOps : ingest et write_ring ; les autres noyaux de la chaîne sont refusés ici.
  std::expected<void, Refusal> ingest(const DecodedFrame& frame, const IngestSpec& spec,
                                      const Tensor& rgb) override;
  std::expected<void, Refusal> resample(const Tensor& in, kernels::Filter filter, const Tensor& out) override;
  std::expected<void, Refusal> rife_pad(const DecodedFrame& frame, const Tensor& padded) override;
  std::expected<void, Refusal> rife_crop(const Tensor& padded, const Tensor& out) override;
  std::expected<void, Refusal> warp_blend(const Tensor& a, const Tensor& b, const Motion& motion, float t,
                                          const Tensor& out) override;
  std::expected<void, Refusal> write_ring(const Tensor& rgb, const RingSlotDesc& slot,
                                          const RingWrite& sync) override;
  std::expected<void, Refusal> measure(const DecodedFrame& a, const DecodedFrame& b) override;
  std::optional<PairScores> previous_scores() override { return std::nullopt; }
  // Extensions de la tranche verticale (tensorrt_rife.cpp) : entrée de RIFE et mesures immédiates.
  std::expected<void, Refusal> rife_input(const Tensor& a, const Tensor& b, float t,
                                          const Tensor& input) override;
  std::expected<void, Refusal> rife_time(const Tensor& input, float t) override;
  std::expected<void, Refusal> rife_pair(const Tensor& a, const Tensor& b, float t,
                                         const Tensor& input) override;
  std::expected<PairScores, Refusal> scores_now(const Tensor& a, const Tensor& b) override;
  std::expected<std::uint64_t, Refusal> scores_start(const Tensor& a, const Tensor& b) override;
  std::expected<PairScores, Refusal> scores_take(std::uint64_t ticket) override;
  bool debands_on_ingest() const override { return true; }
  std::optional<DeviceMemory> device_memory() override;
  std::expected<float, Refusal> flow_max_now(const Motion& motion) override;
  std::expected<std::uint64_t, Refusal> flow_max_start(const Motion& motion) override;
  std::expected<float, Refusal> flow_max_take(std::uint64_t ticket) override;
  // L'analyse d'une image pour le grade (tensorrt_stats.cpp), relue sans attente.
  std::expected<std::uint64_t, Refusal> frame_stats_start(const Tensor& rgb,
                                                          kernels::FrameTransfer transfer) override;
  std::expected<std::optional<kernels::FrameStats>, Refusal> frame_stats_take(std::uint64_t ticket) override;
  std::expected<void, Refusal> warp_blend_poses(const Tensor& a, const Tensor& b, const Motion& motion,
                                                const Motion* previous, float t, const Tensor& out) override;
  // Les mêmes, directement dans le slot (BackendCaps::ring_targets).
  std::expected<void, Refusal> warp_blend_ring(const Tensor& a, const Tensor& b, const Motion& motion,
                                               float t, const RingTarget& ring) override;
  std::expected<void, Refusal> warp_blend_poses_ring(const Tensor& a, const Tensor& b, const Motion& motion,
                                                     const Motion* previous, float t,
                                                     const RingTarget& ring) override;
  // Export par puits CUDA (tensorrt_export.cpp) : RGB → NV12 ou P010 dans l'image de l'encodeur.
  std::expected<void, Refusal> write_yuv(const Tensor& rgb, const YuvTarget& target) override;
  // Étape 11.5 (tensorrt_export.cpp) : chaque voie interrogée par cudaStreamQuery ; les signaux de l'export
  // dans une page enregistrée (cudaHostRegister), écrits par cuStreamWriteValue64 derrière les noyaux.
  std::expected<void, Refusal> health() override;
  std::vector<GpuStageSummary> gpu_profile() override;
  bool map_fences(std::span<std::byte> block) override;
  void unmap_fences(std::span<std::byte> block) override;

 private:
  // Une image décodée tenue jusqu'à la fin de son ingestion côté GPU : FFmpeg la réécrit dès
  // qu'elle revient à son pool.
  struct Held {
    std::shared_ptr<const FrameData> data;
    cudaEvent_t done = nullptr;
    std::uint64_t sequence = 0;  // l'ordre de prise : la plus petite est la plus ancienne
  };
  // Deux paires en vol sur les voies, plus l'image lue en avance.
  static constexpr auto kHeldFrames = static_cast<std::size_t>(kMaxIngestHeldFrames);
  static constexpr std::size_t kFences = 16;
  // Événements par voie, posés à tour de rôle après chaque opération (tensorrt_lanes.cpp).
  static constexpr std::size_t kLaneMarks = 64;
  // Sous cette mémoire, une seule voie et une seule zone d'activation : la RTX 4050 mobile à 6 Go
  // ne paie pas la zone de plus (une carte de 8 Go, qui en annonce un peu moins, garde les voies).
  static constexpr std::size_t kLaneMemory = std::size_t{15} << 29;  // 7,5 Gio

  TensorRtBackend() = default;
  std::expected<void, Refusal> bind() const;  // le device courant est propre à chaque thread
  // rife_input, et rife_pair sans le maillage (tensorrt_rife.cpp).
  std::expected<void, Refusal> fill_rife(const Tensor& a, const Tensor& b, float t, const Tensor& input,
                                         bool mesh, const char* op);
  // Voies (tensorrt_lanes.cpp). Avant une opération de la voie courante qui lit reads et écrit
  // writes (des adresses : tenseurs, zone d'activation, moteur, sommes) : l'événement de
  // l'opération précédente de la voie posé, les attentes sur les autres voies enfilées. Rien
  // sous une seule voie : l'ordre de la file suffit.
  std::expected<void, Refusal> order(std::initializer_list<std::uintptr_t> reads,
                                     std::initializer_list<std::uintptr_t> writes);
  std::expected<void, Refusal> order(std::span<const std::uintptr_t> reads,
                                     std::span<const std::uintptr_t> writes);
  // La voie courante suit tout le travail déjà enfilé sur les autres.
  std::expected<void, Refusal> join();
  std::expected<void, Refusal> apply(const LaneHazards::Op& op);
  std::expected<void, Refusal> record(std::size_t lane);  // l'événement en attente de lane
  std::expected<void, Refusal> create_lanes();
  void destroy_lanes();
  // Attente hôte d'un événement : interrogé quelques centaines de µs d'abord, puis attendu (le
  // contexte en attente bloquante réveille le thread 50 à 200 µs après l'événement, T14).
  static std::expected<void, Refusal> wait_event(cudaEvent_t event);
  // La zone d'activation de la voie courante : l'agrandissement a la sienne, l'interpolation et
  // la mesure l'autre ; une seule sous une voie.
  const std::shared_ptr<SharedActivations>& arena();
  // Le SPAN entier en FP8 (noyaux CUTLASS, tensorrt_span_fp8.cpp) : un paquet « .span8 » à la place de
  // l'ONNX.
  std::expected<Artifact, Refusal> build_span_fp8(const ModelSpec& model, const ShapeSet& shapes);
  std::expected<std::unique_ptr<Engine>, Refusal> load_span_fp8(const Artifact& artifact);
  // ring non nul (étape 3.4b) : corps du réseau, attente de l'emplacement, queue qui écrit aussi sa surface,
  // signal.
  std::expected<void, Refusal> enqueue_span_fp8(SpanFp8Engine& engine, std::span<const Tensor> inputs,
                                                std::span<Tensor> outputs, const RingTarget* ring = nullptr);
  // RIFE fusionné (tensorrt_rife_fused.cpp) : un manifeste « .rifefused » à la place de l'ONNX, six moteurs
  // construits par trtexec, chargés ensemble.
  std::expected<Artifact, Refusal> build_rife_fused(const ModelSpec& model, const ShapeSet& shapes,
                                                    Progress& progress);
  std::expected<std::unique_ptr<Engine>, Refusal> load_rife_fused(const Artifact& artifact);
  std::expected<void, Refusal> enqueue_rife_fused(RifeFusedEngine& engine, std::span<const Tensor> inputs,
                                                  std::span<Tensor> outputs);
  // DistilDRBA fusionné (tensorrt_drba_fused.cpp) : un manifeste « .drbafused » à la place de l'ONNX, quatre
  // moteurs construits par trtexec, chargés ensemble.
  std::expected<Artifact, Refusal> build_drba_fused(const ModelSpec& model, const ShapeSet& shapes,
                                                    Progress& progress);
  std::expected<std::unique_ptr<Engine>, Refusal> load_drba_fused(const Artifact& artifact);
  std::expected<void, Refusal> enqueue_drba_fused(DrbaFusedEngine& engine, std::span<const Tensor> inputs,
                                                  std::span<Tensor> outputs);
  // Les réglages de trtexec de cet appareil (build, build_rife_fused, build_drba_fused).
  prepare::TrtexecConfig trtexec_config() const;
  // La voie lane le temps d'une portée.
  class LaneScope {
   public:
    LaneScope(TensorRtBackend& backend, Lane lane) : backend_(backend), previous_(backend.lane_) {
      backend.use_lane(lane);
    }
    ~LaneScope() { backend_.use_lane(previous_); }
    LaneScope(const LaneScope&) = delete;
    LaneScope& operator=(const LaneScope&) = delete;

   private:
    TensorRtBackend& backend_;
    Lane previous_;
  };
  void log(BackendLogLevel level, std::string_view message) const;
  std::expected<void, Refusal> hold(const DecodedFrame& frame);
  // Après l'ingestion enfilée : l'image hôte rendue à son tampon, l'image GPU tenue (hold).
  std::expected<void, Refusal> release_source(bool host, const DecodedFrame& frame);
  std::expected<ImportedSlot*, Refusal> slot(const RingSlotDesc& desc);
  // Écriture d'un slot : protocole vérifié, voies ordonnées (rgb lu, scratch écrit s'il n'est pas
  // nul), slot importé, attente de wait_value enfilée.
  std::expected<ImportedSlot*, Refusal> begin_write(const Tensor& rgb, const RingSlotDesc& desc,
                                                    const RingWrite& sync, std::uintptr_t scratch = 0);
  // bind(), puis la jointure des voies quand desc ouvre une nouvelle génération du ring : avant order()
  // de toute opération qui finit par begin_slot (A01-02). desc nul : bind() seul.
  std::expected<void, Refusal> bind_joined(const RingSlotDesc* desc);
  std::expected<void, Refusal> bind_joined(const RingTarget* ring) {
    return bind_joined(ring != nullptr ? ring->slot : nullptr);
  }
  // La partie de begin_write propre au slot : protocole, import, attente de wait_value. Les voies
  // sont ordonnées par l'appelant.
  std::expected<ImportedSlot*, Refusal> begin_slot(const RingSlotDesc& desc, const RingWrite& sync);
  // La fin de l'écriture d'un slot : le signal de signal_value.
  std::expected<void, Refusal> finish_slot(const ImportedSlot& slot, const RingWrite& sync);
  // Où écrit un warp ou une composition : le tenseur out, ou directement le slot de ring.
  struct Target {
    const Tensor* out = nullptr;
    const RingTarget* ring = nullptr;
  };
  void release_retired();
  void push_launches();  // WDDM : envoie le lot de lancements (Windows seulement)
  std::expected<void, Refusal> reserve_deband(std::size_t bytes);
  // warp_blend et warp_blend_poses : le même noyau, l'instant réglé ou non.
  struct Poses {
    bool keep = false;
    const Motion* previous = nullptr;
  };
  std::expected<void, Refusal> warp(const Tensor& a, const Tensor& b, const Motion& motion,
                                    const Poses& poses, float t, const Target& target);
  // launch(surface) chronométré sous name : surface nulle (object 0) vers un tenseur ; vers un slot,
  // l'attente de wait_value avant et le signal après, comme write_ring.
  template <typename Launch>
  std::expected<void, Refusal> launch_into(const Target& target, std::string_view name, Launch&& launch) {
    ImportedSlot* imported = nullptr;
    if (target.ring != nullptr) {
      auto begun = begin_slot(*target.ring->slot, target.ring->sync);
      if (!begun) return std::unexpected(std::move(begun.error()));
      imported = *begun;
    }
    GpuTiming& timing = stage(name);
    timing.begin(stream_);
    std::expected<void, Refusal> launched = launch(imported != nullptr ? imported->surface() : RingSurface{});
    timing.end(stream_);
    if (!launched || imported == nullptr) return launched;
    return finish_slot(*imported, target.ring->sync);
  }
  // Objet texture d'un tenseur FP16 lu par le filtre bilinéaire (warp ;
  // tensorrt_textures.cpp), créé à sa première lecture et gardé ; 0 quand le tenseur n'y est pas
  // admissible (pitch, alignement, taille) : le noyau garde la lecture exacte.
  cudaTextureObject_t texture(const Tensor& tensor);
  // Au-delà de kTextures, les objets partent à la retraite derrière un événement par voie et sont
  // détruits quand le GPU a fini de les lire, sans vider les files ; à appeler avant texture().
  void trim_textures();
  void reap_textures(bool wait);  // détruit les retraités finis ; wait : attend ceux en vol
  void destroy_textures();        // tous, les files vidées d'abord

  int device_ = 0;
  cudaDeviceProp properties_{};
  DeviceUuid uuid_{};
  int runtime_version_ = 0;
  int driver_version_ = 0;
  std::filesystem::path cache_dir_;
  std::filesystem::path trtexec_;
  std::string trtexec_sha256_;  // relevé à create (A05-05)
  BackendLog log_;
  cudaStream_t stream_ = nullptr;  // la file de la voie courante, l'une de streams_
 public:
  // Ce que garde un tenseur alloué ici pour se libérer sans synchroniser l'appareil (étape 11.4) : tant que
  // le backend vit, la libération passe par free_after_lanes ; après, par cudaFree.
  struct TensorReleaser {
    TensorRtBackend* backend = nullptr;
  };
  // cudaFreeAsync sur la file courante, après un événement de chaque autre voie : la mémoire revient au
  // pool quand toutes les voies ont dépassé ce point, sans cudaFree (qui synchronise l'appareil).
  void free_after_lanes(void* memory);

 private:
  std::shared_ptr<TensorReleaser> releaser_ = std::make_shared<TensorReleaser>(TensorReleaser{this});
  // Pool par défaut réglé une fois : rien n'est rendu au système entre deux images, et pas de
  // réutilisation par dépendances internes (le flux d'une voie ne dépend jamais d'une autre par le pool).
  void tune_pool() const;
  std::array<cudaStream_t, kLanes> streams_{};
  std::array<std::array<cudaEvent_t, kLaneMarks>, kLanes> lane_events_{};
  std::array<std::uint64_t, kLanes> unrecorded_{};  // rang dont l'événement reste à poser ; 0 : aucun
  Lane lane_ = Lane::Upscale;
  int lanes_ = 1;
  // Voies qui ont reçu du travail, un bit par voie : à plusieurs, le profil de fin dit que ses
  // temps incluent le recouvrement.
  std::uint32_t used_lanes_ = 0;
  LaneHazards hazards_;
  std::unique_ptr<TrtLog> trt_log_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::array<std::shared_ptr<SharedActivations>, 2> arenas_;  // zones d'activation (trt_engine.h)
  std::array<Held, kHeldFrames> held_{};
  std::uint64_t held_sequence_ = 0;
  std::array<cudaEvent_t, kFences> fences_{};
  std::uint64_t fence_count_ = 0;
  // Le temps des moteurs par rôle (0.2) : agrandissement, interpolation, mesure ; gardés à part, même sur une
  // seule voie (--one-lane), pour que le profil ne mêle pas SPAN et RIFE.
  enum class Role : unsigned char { Upscale, Interpolation, Measure };
  static constexpr std::size_t kRoles = 3;
  std::array<std::unique_ptr<GpuTiming>, kRoles> role_timing_;
  GpuTiming& role(Role which) { return *role_timing_.at(static_cast<std::size_t>(which)); }
  // Profil GPU par étape (ingestion, mesures, RIFE, chaque moteur, écriture), résumé au
  // journal à la destruction du backend.
  std::map<std::string, std::unique_ptr<GpuTiming>, std::less<>> stages_;
  // Étape 5.2 : entrées de RIFE (adresse du tenseur) que le dernier moteur à les lire était RIFE fusionné.
  // Ses noyaux calculent le maillage et lisent l'instant à l'élément 0 du plan 6 : rife_time n'y écrit qu'un
  // demi-flottant. Un moteur TensorRT ou DistilDRBA fusionné qui lit la même adresse la retire.
  std::vector<std::uintptr_t> compact_rife_inputs_;
  void note_rife_reader(const Tensor& input, bool compact);
  GpuTiming& stage(std::string_view name);
  std::unique_ptr<GpuSpan> chain_span_;               // de l'ingestion d'une source à la fin de son SPAN
  std::vector<std::unique_ptr<ImportedSlot>> slots_;  // génération courante, par indice
  std::uint32_t generation_ = 0;
  // Imports d'une génération remplacée, relâchés quand retired_done_ est passé côté GPU.
  std::vector<std::unique_ptr<ImportedSlot>> retired_;
  cudaEvent_t retired_done_ = nullptr;
  // Mesures relues par l'hôte (scores_start, flow_max_start) : côté GPU, puis copiées dans une
  // mémoire hôte épinglée ; kPendingReadbacks emplacements à tour de rôle, chacun avec l'événement
  // qui suit sa copie, seul attendu par la relecture.
  struct PairSums {
    double total = 0.0;
    unsigned int worst = 0;
  };
  struct PairRead {
    PairSums* device = nullptr;
    PairSums* host = nullptr;
    cudaEvent_t done = nullptr;
    std::uint64_t ticket = 0;
    double area = 0.0;  // aire remplie de RIFE
  };
  struct FlowRead {
    unsigned int* device = nullptr;  // histogramme des déplacements (kernels::kFlowBins cases)
    unsigned int* host = nullptr;
    cudaEvent_t done = nullptr;
    std::uint64_t ticket = 0;
    std::uint64_t points = 0;  // points comptés
  };
  struct StatsRead {
    void* device = nullptr;  // kernels::FrameCounts
    void* host = nullptr;
    cudaEvent_t done = nullptr;
    std::uint64_t ticket = 0;
  };
  std::array<PairRead, kPendingReadbacks> pair_reads_{};
  std::array<FlowRead, kPendingReadbacks> flow_reads_{};
  std::array<StatsRead, kPendingReadbacks> stats_reads_{};
  std::uint64_t pair_tickets_ = 0;  // derniers tickets lancés, par sorte
  std::uint64_t flow_tickets_ = 0;
  std::uint64_t stats_tickets_ = 0;
  // Lignes filtrées de l'ingestion à filtre étiré (slice_kernels.h, IngestParams::rows).
  float* ingest_rows_ = nullptr;
  std::size_t ingest_rows_size_ = 0;
  // Image ingérée avant son débandage (ingest avec IngestSpec::deband), à la taille de la plus grande vue.
  void* deband_out_ = nullptr;
  std::size_t deband_out_size_ = 0;
  // Envois des images du décodage logiciel (host_upload.h), créés à la première.
  std::unique_ptr<HostUploads> uploads_;
  // write_yuv : les tables de la réduction de chroma pour une taille et un siting, en mémoire
  // d'appareil, et les lignes réduites en largeur (export_kernels.h) ; refaites au changement.
  struct YuvTables {
    int width = 0;
    int height = 0;
    kernels::Siting siting = kernels::Siting::Left;
    int* x_start = nullptr;
    float* x_weights = nullptr;
    int x_taps = 0;
    int* y_start = nullptr;
    float* y_weights = nullptr;
    int y_taps = 0;
    void* rows = nullptr;
  };
  YuvTables yuv_;
  // Blocs de signaux enregistrés (map_fences) : adresse hôte, taille, adresse vue de l'appareil.
  struct MappedFences {
    std::byte* host = nullptr;
    std::size_t bytes = 0;
    std::uintptr_t device = 0;
  };
  std::vector<MappedFences> mapped_fences_;
  // L'adresse d'appareil de done dans un bloc enregistré ; 0 : hors de tout bloc (rappel hôte).
  std::uintptr_t mapped_fence(const void* done) const;
  std::expected<void, Refusal> yuv_tables(int width, int height, kernels::Siting siting);
  void release_yuv_tables(bool behind_lanes = false);  // true en lecture : derrière les voies (11.4)
  // Objets texture par adresse et forme (texture()) ; au-delà de kTextures, tous détruits et
  // recréés à la demande (changements de taille répétés).
  struct TextureKey {
    std::uintptr_t address = 0;
    std::int64_t width = 0;
    std::int64_t rows = 0;  // plans · hauteur
    auto operator<=>(const TextureKey&) const = default;
  };
  static constexpr std::size_t kTextures = 256;
  float pair_levels_ = 219.0F;  // pas de luma 8 bits de la source ingérée : 219 limitée, 255 pleine
  std::map<TextureKey, cudaTextureObject_t> textures_;
  struct RetiredTextures {
    std::vector<cudaTextureObject_t> objects;
    std::vector<cudaEvent_t> events;  // un par voie, posé au départ à la retraite
  };
  std::vector<RetiredTextures> retired_textures_;
};

}  // namespace ajp::gpu::cuda
