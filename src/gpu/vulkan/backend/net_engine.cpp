#include "gpu/vulkan/backend/net_engine.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "ajp_vk_backend_spirv.h"
#include "gpu/ir/graph.h"

namespace ajp::gpu::vk::net {
namespace {

bool same_view(const BufferView& a, const BufferView& b) {
  return a.buffer == b.buffer && a.offset == b.offset && a.size == b.size;
}

VkDeviceSize align(VkDeviceSize value, VkDeviceSize alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::unexpected<Refusal> mismatch(std::string detail) {
  return std::unexpected(Refusal{.code = RefusalCode::InputMismatch, .detail = std::move(detail)});
}

}  // namespace

std::expected<std::unique_ptr<NetEngine>, Refusal> NetEngine::create(VkContext& context,
                                                                     MemoryAllocator& memory,
                                                                     PipelineCache& pipelines,
                                                                     ir::Graph graph, Program program,
                                                                     NetEngineOptions options) {
  if (program.dispatches.empty()) return mismatch("programme sans dispatch");
  std::unique_ptr<NetEngine> engine(
      new NetEngine(context, memory, std::move(graph), std::move(program), std::move(options)));
  if (engine->options_.profile_dispatches) {
    engine->timestamps_ = timestamp_support(context);
    engine->profiling_ = engine->timestamps_.period_ns > 0.0;
    if (!engine->profiling_ && engine->options_.log) {
      engine->options_.log(
          "profil des dispatchs : cet appareil n'a pas de requêtes d'horodatage sur la file de calcul "
          "(timestampValidBits à 0 ou timestampComputeAndGraphics faux), rien n'est mesuré");
    }
  }
  if (auto ok = engine->prepare(pipelines); !ok) return std::unexpected(ok.error());
  return engine;
}

NetEngine::~NetEngine() {
  if (!profiling_) return;
  // Les dernières exécutions, attendues ; un échec de lecture n'empêche pas de journaliser le reste.
  for (Cached& cached : recordings_) {
    if (cached.stamps != nullptr && cached.recording->submitted() > cached.collected) {
      if (auto ok = read_stamps(cached); !ok && options_.log)
        options_.log(std::format("profil des dispatchs : lecture refusée ({})", ok.error().detail));
    }
  }
  if (options_.log && !profile_runs_.empty()) options_.log(format_profile(profile_report()));
}

std::uint64_t NetEngine::input_bytes(std::size_t k) const {
  return ir::storage_bytes(graph_.tensors[graph_.inputs[k]]);
}

std::uint64_t NetEngine::output_bytes(std::size_t k) const {
  return ir::storage_bytes(graph_.tensors[graph_.outputs[k]]);
}

std::expected<void, Refusal> NetEngine::prepare(PipelineCache& pipelines) {
  for (const ir::TensorId id : graph_.inputs) input_names_.push_back(graph_.tensors[id].name);
  for (const ir::TensorId id : graph_.outputs) output_names_.push_back(graph_.tensors[id].name);

  const VkDeviceSize alignment =
      std::max<VkDeviceSize>(ir::kBufferAlignment, context_.caps().min_storage_buffer_alignment);
  if (!program_.aliases.empty() && program_.aliases.size() != graph_.tensors.size()) {
    return mismatch(std::format("{} alias pour {} tenseurs", program_.aliases.size(), graph_.tensors.size()));
  }
  auto plan = ir::plan_memory(graph_, alignment, program_.aliases);
  if (!plan) return std::unexpected(plan.error());
  plan_ = std::move(*plan);
  places_.resize(graph_.tensors.size());
  for (const ir::Allocation& a : plan_.allocations) places_[a.tensor] = a;
  if (plan_.arena_bytes > 0) {
    auto arena = Buffer::create(context_, memory_, plan_.arena_bytes, MemoryUse::Device);
    if (!arena) return std::unexpected(arena.error());
    arena_ = std::move(*arena);
  }

  // Les blobs bout à bout, chacun aligné comme une vue de tampon de stockage, envoyés une fois.
  VkDeviceSize total = 0;
  for (const auto& blob : program_.blobs) {
    blob_offsets_.push_back(total);
    total = align(total + std::max<VkDeviceSize>(blob.size(), 4), alignment);
  }
  if (total > 0) {
    std::vector<std::byte> bytes(total, std::byte{0});
    for (std::size_t k = 0; k < program_.blobs.size(); ++k) {
      std::memcpy(bytes.data() + blob_offsets_[k], program_.blobs[k].data(), program_.blobs[k].size());
    }
    auto blobs = Buffer::create(context_, memory_, total, MemoryUse::Device);
    if (!blobs) return std::unexpected(blobs.error());
    if (auto ok = upload(context_, memory_, *blobs, 0, bytes); !ok) return std::unexpected(ok.error());
    blobs_ = std::move(*blobs);
  }
  // Les poids vivent désormais sur l'appareil : l'hôte n'en garde que les tailles.
  for (const auto& blob : program_.blobs) blob_sizes_.push_back(std::max<VkDeviceSize>(blob.size(), 4));
  program_.blobs.clear();
  program_.blobs.shrink_to_fit();
  auto dummy = Buffer::create(context_, memory_, alignment, MemoryUse::Device);
  if (!dummy) return std::unexpected(dummy.error());
  const std::vector<std::byte> zeros(alignment, std::byte{0});
  if (auto ok = upload(context_, memory_, *dummy, 0, zeros); !ok) return std::unexpected(ok.error());
  dummy_ = std::move(*dummy);

  for (const Dispatch& d : program_.dispatches) {
    const std::span<const std::uint32_t> words = spirv::words(d.shader);
    if (words.empty()) {
      return std::unexpected(Refusal{RefusalCode::FeatureUnavailable,
                                     std::format("shader {} absent du moteur Vulkan", d.shader)});
    }
    auto pipeline = pipelines.get({.shader = d.shader,
                                   .constants = d.constants,
                                   .buffers = static_cast<std::uint32_t>(d.bindings.size()),
                                   .push_bytes = static_cast<std::uint32_t>(d.push.size() * 4),
                                   .spirv = words,
                                   .subgroup_size = d.subgroup});
    if (!pipeline) return std::unexpected(pipeline.error());
    pipelines_.push_back(*pipeline);
  }
  return {};
}

std::expected<BufferView, Refusal> NetEngine::resolve(const Binding& binding,
                                                      std::span<const BufferView> inputs,
                                                      std::span<const BufferView> outputs) const {
  switch (binding.kind) {
    case BindingKind::Dummy:
      if (!dummy_) return mismatch("tampon factice absent");
      return dummy_->view();
    case BindingKind::Blob: {
      if (binding.index >= blob_sizes_.size() || !blobs_)
        return mismatch(std::format("blob {} absent", binding.index));
      return blobs_->view(blob_offsets_[binding.index], blob_sizes_[binding.index]);
    }
    case BindingKind::Tensor:
      break;
  }
  const ir::TensorId id = binding.index;
  if (id >= graph_.tensors.size()) return mismatch(std::format("tenseur {} absent", id));
  // Un tenseur rangé dans un autre : une vue dans la place de sa racine.
  ir::TensorId root = id;
  VkDeviceSize offset = 0;
  for (std::size_t hops = 0; !program_.aliases.empty() && program_.aliases[root].parent != ir::kNoTensor;
       ++hops) {
    if (hops == graph_.tensors.size())
      return mismatch(std::format("{} : alias en boucle", graph_.tensors[id].name));
    offset += program_.aliases[root].offset;
    root = program_.aliases[root].parent;
  }
  auto base = root_view(root, inputs, outputs);
  if (!base || root == id) return base;
  const VkDeviceSize bytes = ir::storage_bytes(graph_.tensors[id]);
  if (offset + bytes > base->size) {
    return mismatch(std::format("{} : {} o à {} dépassent les {} o de {}", graph_.tensors[id].name, bytes,
                                offset, base->size, graph_.tensors[root].name));
  }
  return BufferView{.buffer = base->buffer, .offset = base->offset + offset, .size = bytes};
}

std::expected<BufferView, Refusal> NetEngine::root_view(ir::TensorId id, std::span<const BufferView> inputs,
                                                        std::span<const BufferView> outputs) const {
  if (const auto in = std::ranges::find(graph_.inputs, id); in != graph_.inputs.end()) {
    return inputs[static_cast<std::size_t>(in - graph_.inputs.begin())];
  }
  if (const auto out = std::ranges::find(graph_.outputs, id); out != graph_.outputs.end()) {
    return outputs[static_cast<std::size_t>(out - graph_.outputs.begin())];
  }
  const auto& place = places_[id];
  if (!place || !arena_) {
    return std::unexpected(Refusal{RefusalCode::GraphInvalid,
                                   std::format("{} : aucune place dans l'arène", graph_.tensors[id].name)});
  }
  return arena_->view(place->offset, ir::storage_bytes(graph_.tensors[id]));
}

std::expected<std::unique_ptr<Recording>, Refusal> NetEngine::record(std::span<const BufferView> inputs,
                                                                     std::span<const BufferView> outputs,
                                                                     QueryPool*& stamps) {
  std::uint32_t buffers = 0;
  for (const Dispatch& d : program_.dispatches) buffers += static_cast<std::uint32_t>(d.bindings.size());
  auto recorder = Recorder::begin(
      context_, memory_,
      {.sets = static_cast<std::uint32_t>(program_.dispatches.size()), .buffers = buffers, .images = 1});
  if (!recorder) return std::unexpected(recorder.error());
  if (profiling_) {
    const auto count = static_cast<std::uint32_t>(2 * program_.dispatches.size());
    auto pool = std::make_unique<QueryPool>(context_, query_functions(context_), count);
    if (auto ok = pool->create(); !ok) return std::unexpected(ok.error());
    stamps = stamp_pools_.emplace_back(std::move(pool)).get();
  }
  // Une estampille de fin de pipeline à l'index donné ; la réinitialisation du pool, avant la première.
  const auto stamp = [&](std::uint32_t index) -> Done {
    return recorder->commands([&](VkCommandBuffer cb) {
      if (index == 0) stamps->fn().reset(cb, stamps->handle(), 0, stamps->count());
      stamps->fn().write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, stamps->handle(), index);
    });
  };
  std::vector<BufferView> views;
  for (std::size_t k = 0; k < program_.dispatches.size(); ++k) {
    const Dispatch& d = program_.dispatches[k];
    views.clear();
    for (const Binding& b : d.bindings) {
      auto view = resolve(b, inputs, outputs);
      if (!view) return std::unexpected(view.error());
      views.push_back(*view);
    }
    const std::span<const std::byte> push = std::as_bytes(std::span(d.push));
    if (stamps != nullptr) {
      if (auto ok = stamp(static_cast<std::uint32_t>(2 * k)); !ok) return std::unexpected(ok.error());
    }
    if (auto ok = recorder->dispatch(*pipelines_[k], views, {}, push, d.groups); !ok) {
      return std::unexpected(ok.error());
    }
    if (stamps != nullptr) {
      if (auto ok = stamp(static_cast<std::uint32_t>((2 * k) + 1)); !ok) return std::unexpected(ok.error());
    }
  }
  return recorder->finish();
}

std::expected<void, Refusal> NetEngine::read_stamps(Cached& cached) {
  const Recording& recording = *cached.recording;
  if (auto ok = context_.wait_on_host(recording.last_run()); !ok) return std::unexpected(ok.error());
  const QueryPool& pool = *cached.stamps;
  std::vector<std::uint64_t> ticks(pool.count());
  if (auto ok =
          check(pool.fn().results(context_.device(), pool.handle(), 0, pool.count(),
                                  ticks.size() * sizeof(std::uint64_t), ticks.data(), sizeof(std::uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                "vkGetQueryPoolResults");
      !ok) {
    return std::unexpected(ok.error());
  }
  cached.collected = recording.submitted();
  if (profile_runs_.size() >= kMaxProfileRuns) return {};
  ProfileRun run;
  run.dispatch_ms.reserve(program_.dispatches.size());
  for (std::size_t k = 0; k < program_.dispatches.size(); ++k) {
    run.dispatch_ms.push_back(ticks_to_ns(ticks[2 * k], ticks[(2 * k) + 1], timestamps_) / 1.0e6);
  }
  run.span_ms = ticks_to_ns(ticks.front(), ticks.back(), timestamps_) / 1.0e6;
  profile_runs_.push_back(std::move(run));
  return {};
}

std::expected<void, Refusal> NetEngine::collect_profile(const Recording& recording) {
  if (!profiling_) return {};
  for (Cached& cached : recordings_) {
    if (cached.recording.get() != &recording || cached.stamps == nullptr) continue;
    if (recording.submitted() <= cached.collected) return {};
    return read_stamps(cached);
  }
  return {};
}

void NetEngine::collect_idle_profiles() {
  for (Cached& cached : recordings_) {
    if (cached.stamps != nullptr && cached.recording->submitted() > cached.collected &&
        cached.recording->idle()) {
      // Un échec de lecture perd cette exécution du rapport, pas l'exécution elle-même.
      if (auto ok = read_stamps(cached); !ok && options_.log)
        options_.log(std::format("profil des dispatchs : lecture refusée ({})", ok.error().detail));
    }
  }
}

void NetEngine::forget_destroyed() {
  retired_.collect();
  const DestroyedLog& log = context_.destroyed();
  if (log.mark() == destroyed_mark_) return;
  gone_.clear();
  const DestroyedLog::Read read = log.since(destroyed_mark_, gone_);
  destroyed_mark_ = read.mark;
  std::erase_if(recordings_, [&](Cached& cached) {
    const bool gone = !read.complete || std::ranges::any_of(cached.views, [&](const BufferView& view) {
      return std::ranges::contains(gone_, handle_word(view.buffer));
    });
    if (gone) retired_.retire(std::move(cached.recording));
    return gone;
  });
}

std::expected<Recording*, Refusal> NetEngine::recording(std::span<const BufferView> inputs,
                                                        std::span<const BufferView> outputs) {
  if (inputs.size() != graph_.inputs.size() || outputs.size() != graph_.outputs.size()) {
    return mismatch(std::format("{} entrées et {} sorties, {} et {} attendues", inputs.size(), outputs.size(),
                                graph_.inputs.size(), graph_.outputs.size()));
  }
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    if (inputs[k].buffer == VK_NULL_HANDLE || inputs[k].size < input_bytes(k)) {
      return mismatch(
          std::format("entrée {} : {} o, {} attendus", input_names_[k], inputs[k].size, input_bytes(k)));
    }
  }
  for (std::size_t k = 0; k < outputs.size(); ++k) {
    if (outputs[k].buffer == VK_NULL_HANDLE || outputs[k].size < output_bytes(k)) {
      return mismatch(
          std::format("sortie {} : {} o, {} attendus", output_names_[k], outputs[k].size, output_bytes(k)));
    }
  }
  ++uses_;
  if (profiling_) collect_idle_profiles();
  forget_destroyed();
  const auto matches = [&](const Cached& c) {
    return std::ranges::equal(c.views.begin(), c.views.begin() + static_cast<std::ptrdiff_t>(inputs.size()),
                              inputs.begin(), inputs.end(), same_view) &&
           std::ranges::equal(c.views.begin() + static_cast<std::ptrdiff_t>(inputs.size()), c.views.end(),
                              outputs.begin(), outputs.end(), same_view);
  };
  if (auto found = std::ranges::find_if(recordings_, matches); found != recordings_.end()) {
    found->used = uses_;
    return found->recording.get();
  }
  QueryPool* stamps = nullptr;
  auto made = record(inputs, outputs, stamps);
  if (!made) return std::unexpected(made.error());
  if (recordings_.size() >= kMaxRecordings) {
    const auto oldest = std::ranges::min_element(recordings_, {}, &Cached::used);
    retired_.retire(std::move(oldest->recording));
    recordings_.erase(oldest);
  }
  Cached entry{.views = {inputs.begin(), inputs.end()},
               .recording = std::move(*made),
               .used = uses_,
               .stamps = stamps};
  entry.views.insert(entry.views.end(), outputs.begin(), outputs.end());
  recordings_.push_back(std::move(entry));
  return recordings_.back().recording.get();
}

}  // namespace ajp::gpu::vk::net
