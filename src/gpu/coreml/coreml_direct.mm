#include "gpu/coreml/coreml_direct.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <mutex>
#include <optional>
#include <string_view>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gpu/coreml/coreml_compiled.h"
#include "gpu/coreml/coreml_surfaces.h"

// La spécialisation pour l'inférence (MLOptimizationHints) : SDK de macOS 14.4 et plus, comme
// ONNX Runtime la pose (HAS_COREMLOPTIMIZATIONHINT).
#if defined(__MAC_14_4) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_14_4
#define AJP_COREML_HINTS 1
#else
#define AJP_COREML_HINTS 0
#endif

namespace ajp::gpu::coreml_rt {
namespace {

using vk::net::HostBinding;

std::unexpected<Refusal> refuse_direct(RefusalCode code, std::string text) {
  return std::unexpected(Refusal{.code = code, .detail = std::format("Core ML direct : {}", text)});
}

std::string describe(NSError* error) {
  if (error == nil) return "erreur inconnue";
  const char* text = error.localizedDescription.UTF8String;
  return std::format("{} ({} {})", text != nullptr ? text : "?", error.domain.UTF8String, error.code);
}

std::string describe(NSException* exception) {
  const char* text = exception.reason.UTF8String;
  return std::format("{} : {}", exception.name.UTF8String, text != nullptr ? text : "?");
}

NSString* ns_text(std::string_view text) {
  return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
}

// Une référence Core Foundation possédée, rendue à la destruction (ARC ne les gère pas).
template <class T>
class Cf {
 public:
  Cf() = default;
  explicit Cf(T ref) : ref_(ref) {}  // prend une référence déjà retenue
  Cf(const Cf& other) : ref_(other.ref_) {
    if (ref_ != nullptr) CFRetain(ref_);
  }
  Cf(Cf&& other) noexcept : ref_(std::exchange(other.ref_, nullptr)) {}
  Cf& operator=(Cf other) noexcept {
    std::swap(ref_, other.ref_);
    return *this;
  }
  ~Cf() {
    if (ref_ != nullptr) CFRelease(ref_);
  }
  T get() const { return ref_; }

 private:
  T ref_ = nullptr;
};

}  // namespace

// Chaque IOSurface vivante par l'adresse de sa projection : son tampon de pixels, la forme du
// tenseur qu'elle porte, et les prédictions en cours qui l'ont rendue au GPU et au Neural Engine.
class Surfaces {
 public:
  struct Found {
    Cf<CVPixelBufferRef> pixels;
    std::uint64_t id = 0;
  };

  void add(const std::byte* data, Cf<IOSurfaceRef> surface, Cf<CVPixelBufferRef> pixels, TensorType type,
           const Shape& shape, std::uint64_t bytes) {
    const std::scoped_lock lock(mutex_);
    entries_.insert_or_assign(data, Entry{.surface = std::move(surface),
                                          .pixels = std::move(pixels),
                                          .type = type,
                                          .shape = shape,
                                          .bytes = bytes,
                                          .id = next_++});
  }
  // Retirée à la libération du tenseur : vrai si le CPU la tient encore verrouillée.
  bool remove(const std::byte* data) {
    const std::scoped_lock lock(mutex_);
    const auto it = entries_.find(data);
    if (it == entries_.end()) return false;
    const bool locked = it->second.lent == 0;
    entries_.erase(it);
    return locked;
  }
  // L'IOSurface qui commence à data et porte ce tenseur (au moins bytes octets).
  std::optional<Found> find(const std::byte* data, TensorType type, const Shape& shape,
                            std::uint64_t bytes) const {
    const std::scoped_lock lock(mutex_);
    const auto it = entries_.find(data);
    if (it == entries_.end()) return std::nullopt;
    const Entry& e = it->second;
    if (e.type != type || e.shape != shape || e.bytes < bytes) return std::nullopt;
    return Found{.pixels = e.pixels, .id = e.id};
  }
  // Rendue au GPU et au Neural Engine le temps d'une prédiction (le CPU la déverrouille : ce que
  // lui et Vulkan y ont écrit leur est visible), puis reprise (ce qu'ils y ont écrit revient à la
  // chaîne). Comptées : deux réseaux peuvent partager un tenseur.
  void lend(const std::byte* data) {
    const std::scoped_lock lock(mutex_);
    const auto it = entries_.find(data);
    if (it != entries_.end() && it->second.lent++ == 0) IOSurfaceUnlock(it->second.surface.get(), 0, nullptr);
  }
  void take_back(const std::byte* data) {
    const std::scoped_lock lock(mutex_);
    const auto it = entries_.find(data);
    if (it != entries_.end() && it->second.lent > 0 && --it->second.lent == 0)
      IOSurfaceLock(it->second.surface.get(), 0, nullptr);
  }

 private:
  struct Entry {
    Cf<IOSurfaceRef> surface;
    Cf<CVPixelBufferRef> pixels;
    TensorType type = TensorType::F16;
    Shape shape;
    std::uint64_t bytes = 0;
    std::uint64_t id = 0;  // unique : une IOSurface plus tard à la même adresse en est une autre
    int lent = 0;
  };

  mutable std::mutex mutex_;
  std::unordered_map<const std::byte*, Entry> entries_;
  std::uint64_t next_ = 1;
};

LentSurface::LentSurface(std::shared_ptr<Surfaces> surfaces, const std::byte* data, IOSurfaceRef surface)
    : surfaces_(std::move(surfaces)), data_(data), surface_(surface) {
  CFRetain(surface_);
  surfaces_->lend(data_);
}

LentSurface::LentSurface(LentSurface&& other) noexcept
    : surfaces_(std::move(other.surfaces_)),
      data_(other.data_),
      surface_(std::exchange(other.surface_, nullptr)) {}

LentSurface::~LentSurface() {
  if (surface_ == nullptr) return;
  surfaces_->take_back(data_);
  CFRelease(surface_);
}

std::optional<LentSurface> lend_surface(const std::shared_ptr<Surfaces>& surfaces, const HostBinding& b) {
  if (surfaces == nullptr || b.data == nullptr) return std::nullopt;
  const std::optional<Surfaces::Found> found = surfaces->find(b.data, b.type, b.shape, b.bytes);
  if (!found) return std::nullopt;
  IOSurfaceRef surface = CVPixelBufferGetIOSurface(found->pixels.get());
  if (surface == nullptr) return std::nullopt;
  return std::optional<LentSurface>(std::in_place, surfaces, b.data, surface);
}

namespace {

// La mémoire d'un tenseur de la chaîne (Runtime::allocate_host) : une IOSurface, verrouillée pour
// le CPU de sa création à sa libération, sauf quand une prédiction la rend à Core ML.
class SurfaceMemory final : public vk::net::HostMemory {
 public:
  SurfaceMemory(std::shared_ptr<Surfaces> surfaces, Cf<IOSurfaceRef> surface, std::byte* data)
      : surfaces_(std::move(surfaces)), surface_(std::move(surface)), data_(data) {}
  SurfaceMemory(const SurfaceMemory&) = delete;
  SurfaceMemory& operator=(const SurfaceMemory&) = delete;
  SurfaceMemory(SurfaceMemory&&) = delete;
  SurfaceMemory& operator=(SurfaceMemory&&) = delete;
  ~SurfaceMemory() override {
    if (surfaces_->remove(data_)) IOSurfaceUnlock(surface_.get(), 0, nullptr);
  }
  std::byte* data() const override { return data_; }

 private:
  std::shared_ptr<Surfaces> surfaces_;
  Cf<IOSurfaceRef> surface_;
  std::byte* data_;
};

// Les unités telles qu'ONNX Runtime les donne à Core ML (model.mm, Execution::LoadModel).
MLComputeUnits compute_units(Units units) {
  switch (units) {
    case Units::CpuAndGpu:
      return MLComputeUnitsCPUAndGPU;
    case Units::CpuAndNeuralEngine:
      return MLComputeUnitsCPUAndNeuralEngine;
    case Units::All:
      break;
  }
  return MLComputeUnitsAll;
}

std::optional<TensorType> tensor_type(MLMultiArrayDataType type) {
  if (type == MLMultiArrayDataTypeFloat16) return TensorType::F16;
  if (type == MLMultiArrayDataTypeFloat32) return TensorType::F32;
  return std::nullopt;
}

MLMultiArrayDataType array_type(TensorType type) {
  return type == TensorType::F32 ? MLMultiArrayDataTypeFloat32 : MLMultiArrayDataTypeFloat16;
}

std::size_t type_bytes(TensorType type) {
  return type == TensorType::F32 ? 4U : 2U;
}

// Le nom de name dans la description, tel quel ou sous son nom de programme ML, et son type.
struct Found {
  NSString* name = nil;
  TensorType type = TensorType::F16;
};

std::expected<Found, Refusal> find(NSDictionary<NSString*, MLFeatureDescription*>* descriptions,
                                   const std::string& name) {
  NSString* key = ns_text(name);
  MLFeatureDescription* description = descriptions[key];
  if (description == nil) {
    key = ns_text(ml_program_name(name));
    description = descriptions[key];
  }
  if (description == nil)
    return refuse_direct(RefusalCode::FeatureUnavailable, std::format("{} absent du modèle", name));
  if (description.type != MLFeatureTypeMultiArray || description.multiArrayConstraint == nil)
    return refuse_direct(RefusalCode::FeatureUnavailable, std::format("{} n'est pas un tableau", name));
  const std::optional<TensorType> type = tensor_type(description.multiArrayConstraint.dataType);
  if (!type)
    return refuse_direct(RefusalCode::FeatureUnavailable, std::format("{} n'est pas flottant", name));
  return Found{.name = key, .type = *type};
}

// Une entrée ou une sortie du modèle et la mémoire de la chaîne qu'elle enveloppe en dernier : par
// son IOSurface (surface, son numéro dans Surfaces) ou telle quelle.
struct Slot {
  NSString* name = nil;
  TensorType type = TensorType::F16;
  MLMultiArray* array = nil;
  std::byte* data = nullptr;
  std::uint64_t bytes = 0;
  Shape shape;
  Cf<CVPixelBufferRef> pixels;
  std::uint64_t surface = 0;
  bool plain_only = false;  // Core ML a refusé une IOSurface pour elle : la mémoire telle quelle
};

// Le tableau sur le tampon de pixels d'une IOSurface, s'il a les pas denses de strides ; sinon nil.
MLMultiArray* surface_array(CVPixelBufferRef pixels, NSArray<NSNumber*>* shape, NSArray<NSNumber*>* strides) {
  @try {
    MLMultiArray* array = [[MLMultiArray alloc] initWithPixelBuffer:pixels shape:shape];
    if (array == nil || array.dataType != MLMultiArrayDataTypeFloat16 ||
        ![array.strides isEqualToArray:strides])
      return nil;
    return array;
  } @catch (NSException* exception) {
    (void)exception;
    return nil;
  }
}

// Copie dense (NCHW) d'un tableau de Core ML aux pas quelconques : celui qu'il rend quand il
// n'écrit pas dans la sortie fournie.
void strided_copy(const std::byte* from, const std::int64_t* dims, const std::int64_t* strides,
                  std::size_t rank, std::size_t element, std::byte*& to) {
  if (rank == 1) {
    if (strides[0] == 1) {
      const std::size_t run = static_cast<std::size_t>(dims[0]) * element;
      std::memcpy(to, from, run);
      to += run;
      return;
    }
    for (std::int64_t i = 0; i < dims[0]; ++i) {
      std::memcpy(to, from + (static_cast<std::size_t>(i * strides[0]) * element), element);
      to += element;
    }
    return;
  }
  for (std::int64_t i = 0; i < dims[0]; ++i)
    strided_copy(from + (static_cast<std::size_t>(i * strides[0]) * element), dims + 1, strides + 1, rank - 1,
                 element, to);
}

class Direct final : public DirectModel {
 public:
  Direct(MLModel* model, std::vector<Slot> inputs, std::vector<Slot> outputs,
         std::shared_ptr<Surfaces> surfaces)
      : model_(model),
        inputs_(std::move(inputs)),
        outputs_(std::move(outputs)),
        surfaces_(std::move(surfaces)) {
    for (const Slot& s : inputs_) in_types_.push_back(s.type);
    for (const Slot& s : outputs_) out_types_.push_back(s.type);
  }

  std::span<const TensorType> input_types() const override { return in_types_; }
  std::span<const TensorType> output_types() const override { return out_types_; }
  bool on_surfaces() const override {
    const auto by_surface = [](const Slot& s) { return s.surface != 0; };
    return std::ranges::all_of(inputs_, by_surface) && std::ranges::all_of(outputs_, by_surface);
  }

  std::expected<void, Refusal> predict(std::span<const HostBinding> inputs,
                                       std::span<const HostBinding> outputs) override {
    if (inputs.size() != inputs_.size() || outputs.size() != outputs_.size())
      return refuse_direct(RefusalCode::InputMismatch, "nombre d'entrées ou de sorties");
    @autoreleasepool {
      @try {
        bool inputs_changed = features_ == nil;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
          auto changed = wrap(inputs_[i], inputs[i]);
          if (!changed) return std::unexpected(changed.error());
          inputs_changed = inputs_changed || *changed;
        }
        if (inputs_changed) {
          NSMutableDictionary<NSString*, id>* values =
              [NSMutableDictionary dictionaryWithCapacity:inputs_.size()];
          for (const Slot& s : inputs_) values[s.name] = [MLFeatureValue featureValueWithMultiArray:s.array];
          NSError* error = nil;
          features_ = [[MLDictionaryFeatureProvider alloc] initWithDictionary:values error:&error];
          if (features_ == nil) return refuse_direct(RefusalCode::DeviceUnavailable, describe(error));
        }
        bool outputs_changed = options_ == nil;
        for (std::size_t i = 0; i < outputs.size(); ++i) {
          auto changed = wrap(outputs_[i], outputs[i]);
          if (!changed) return std::unexpected(changed.error());
          outputs_changed = outputs_changed || *changed;
        }
        if (outputs_changed) {
          NSMutableDictionary<NSString*, id>* backings =
              [NSMutableDictionary dictionaryWithCapacity:outputs_.size()];
          for (const Slot& s : outputs_) backings[s.name] = s.array;
          options_ = [[MLPredictionOptions alloc] init];
          options_.outputBackings = backings;
        }
        NSError* error = nil;
        id<MLFeatureProvider> result = nil;
        {
          const Lent lent(*this);
          result = [model_ predictionFromFeatures:features_ options:options_ error:&error];
        }
        if (result == nil) return refuse_direct(RefusalCode::DeviceUnavailable, describe(error));
        for (const Slot& s : outputs_) {
          if (auto ok = collect(result, s); !ok) return ok;
        }
        return {};
      } @catch (NSException* exception) {
        // Des enveloppes refaites à l'appel suivant : rien ne garde l'état d'un appel interrompu.
        features_ = nil;
        options_ = nil;
        return refuse_direct(RefusalCode::DeviceUnavailable, describe(exception));
      }
    }
  }

 private:
  // Les IOSurface des liaisons rendues à Core ML le temps de la prédiction, reprises ensuite, même
  // sur une exception.
  class Lent {
   public:
    explicit Lent(Direct& direct) : direct_(direct) {
      if (direct_.surfaces_ == nullptr) return;
      for (const std::vector<Slot>* slots : {&direct_.inputs_, &direct_.outputs_}) {
        for (const Slot& s : *slots) {
          if (s.surface == 0) continue;
          direct_.surfaces_->lend(s.data);
          direct_.lent_.push_back(s.data);
        }
      }
    }
    Lent(const Lent&) = delete;
    Lent& operator=(const Lent&) = delete;
    Lent(Lent&&) = delete;
    Lent& operator=(Lent&&) = delete;
    ~Lent() {
      for (const std::byte* data : direct_.lent_) direct_.surfaces_->take_back(data);
      direct_.lent_.clear();
    }

   private:
    Direct& direct_;
  };

  // Enveloppe la mémoire de b, une fois tant que la chaîne donne la même : vrai si l'enveloppe change.
  // Par son IOSurface quand elle en est une, à la forme du tenseur.
  std::expected<bool, Refusal> wrap(Slot& slot, const HostBinding& b) {
    if (b.type != slot.type)
      return refuse_direct(RefusalCode::InputMismatch,
                           std::format("{} : type de la liaison", slot.name.UTF8String));
    std::optional<Surfaces::Found> found;
    if (surfaces_ != nullptr && !slot.plain_only) found = surfaces_->find(b.data, b.type, b.shape, b.bytes);
    if (slot.array != nil && slot.data == b.data && slot.bytes == b.bytes && slot.shape == b.shape &&
        slot.surface == (found ? found->id : 0))
      return false;
    std::size_t count = 1;
    for (const std::int64_t d : b.shape.dims) count *= static_cast<std::size_t>(d);
    if (count * type_bytes(b.type) != b.bytes)
      return refuse_direct(RefusalCode::InputMismatch,
                           std::format("{} : taille de la liaison", slot.name.UTF8String));
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:b.shape.dims.size()];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:b.shape.dims.size()];
    std::int64_t stride = 1;
    for (std::size_t i = b.shape.dims.size(); i-- > 0;) {
      [shape insertObject:@(b.shape.dims[i]) atIndex:0];
      [strides insertObject:@(stride) atIndex:0];
      stride *= b.shape.dims[i];
    }
    MLMultiArray* array = nil;
    if (found) {
      array = surface_array(found->pixels.get(), shape, strides);
      if (array == nil) {
        slot.plain_only = true;
        found.reset();
      }
    }
    if (array == nil) {
      NSError* error = nil;
      array = [[MLMultiArray alloc] initWithDataPointer:b.data
                                                  shape:shape
                                               dataType:array_type(b.type)
                                                strides:strides
                                            deallocator:^(void* /*bytes*/) {
                                            }
                                                  error:&error];
      if (array == nil) return refuse_direct(RefusalCode::DeviceUnavailable, describe(error));
    }
    slot = Slot{.name = slot.name,
                .type = slot.type,
                .array = array,
                .data = b.data,
                .bytes = b.bytes,
                .shape = b.shape,
                .pixels = found ? found->pixels : Cf<CVPixelBufferRef>{},
                .surface = found ? found->id : 0,
                .plain_only = slot.plain_only};
    return true;
  }

  // La sortie s écrite dans sa mémoire : en place (outputBackings), sinon recopiée de ce que Core ML
  // rend.
  static std::expected<void, Refusal> collect(id<MLFeatureProvider> result, const Slot& s) {
    MLMultiArray* got = [result featureValueForName:s.name].multiArrayValue;
    if (got == nil)
      return refuse_direct(RefusalCode::DeviceUnavailable, std::format("{} sans valeur", s.name.UTF8String));
    if (got.dataType != array_type(s.type) || got.shape.count != s.shape.dims.size())
      return refuse_direct(RefusalCode::InputMismatch,
                           std::format("{} : forme ou type rendus", s.name.UTF8String));
    std::vector<std::int64_t> dims(got.shape.count);
    std::vector<std::int64_t> strides(got.shape.count);
    for (NSUInteger i = 0; i < got.shape.count; ++i) {
      dims[i] = got.shape[i].longLongValue;
      strides[i] = got.strides[i].longLongValue;
      if (dims[i] != s.shape.dims[i])
        return refuse_direct(RefusalCode::InputMismatch, std::format("{} : forme rendue", s.name.UTF8String));
    }
    // Sur son IOSurface : écrite en place.
    if (s.pixels.get() != nullptr && got.pixelBuffer == s.pixels.get()) return {};
    // Le bloc ne capture que des pointeurs et des valeurs : il s'exécute avant le retour.
    std::byte* const target = s.data;
    const std::int64_t* const shape = dims.data();
    const std::int64_t* const steps = strides.data();
    const std::size_t rank = dims.size();
    const std::size_t element = type_bytes(s.type);
    __block bool read = false;
    [got getBytesWithHandler:^(const void* bytes, NSInteger /*size*/) {
      if (bytes == nullptr) return;
      read = true;
      if (bytes == target) return;  // écrite en place
      std::byte* to = target;
      strided_copy(static_cast<const std::byte*>(bytes), shape, steps, rank, element, to);
    }];
    if (!read)
      return refuse_direct(RefusalCode::DeviceUnavailable, std::format("{} illisible", s.name.UTF8String));
    return {};
  }

  MLModel* model_;
  std::vector<Slot> inputs_;
  std::vector<Slot> outputs_;
  std::vector<TensorType> in_types_;
  std::vector<TensorType> out_types_;
  MLDictionaryFeatureProvider* features_ = nil;  // les entrées enveloppées
  MLPredictionOptions* options_ = nil;           // les sorties fournies à Core ML
  std::shared_ptr<Surfaces> surfaces_;           // nullptr : jamais par une IOSurface
  std::vector<const std::byte*> lent_;           // les IOSurface rendues pendant la prédiction
};

}  // namespace

std::shared_ptr<Surfaces> make_surfaces() {
  return std::make_shared<Surfaces>();
}

std::expected<std::unique_ptr<vk::net::HostMemory>, Refusal> allocate_surface(
    const std::shared_ptr<Surfaces>& surfaces, std::uint64_t bytes, std::uint64_t alignment, TensorType type,
    const Shape& shape) {
  const auto unsupported = [](std::string text) {
    return std::unexpected(
        Refusal{.code = RefusalCode::FormatUnsupported, .detail = std::format("IOSurface : {}", text)});
  };
  if (surfaces == nullptr)
    return std::unexpected(
        Refusal{.code = RefusalCode::FeatureUnavailable, .detail = "IOSurface : sans registre"});
  if (type != TensorType::F16) return unsupported("tenseur FP32");
  const std::int64_t width = shape.dims[3];
  const std::int64_t rows = shape.dims[0] * shape.dims[1] * shape.dims[2];
  if (width <= 0 || rows <= 0) return unsupported("forme vide");
  const auto row_bytes = static_cast<std::uint64_t>(width) * 2U;
  const std::uint64_t dense = row_bytes * static_cast<std::uint64_t>(rows);
  // La page et l'alignement de l'appareil sont des puissances de deux : le plus grand des deux.
  const std::uint64_t unit = std::max(alignment, static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE)));
  const std::uint64_t size = (std::max(bytes, dense) + unit - 1) / unit * unit;
  const auto format = static_cast<std::uint32_t>(kCVPixelFormatType_OneComponent16Half);
  @autoreleasepool {
    NSDictionary* properties = @{
      (__bridge NSString*)kIOSurfaceWidth : @(width),
      (__bridge NSString*)kIOSurfaceHeight : @(rows),
      (__bridge NSString*)kIOSurfaceBytesPerElement : @2,
      (__bridge NSString*)kIOSurfaceBytesPerRow : @(row_bytes),
      (__bridge NSString*)kIOSurfaceAllocSize : @(size),
      (__bridge NSString*)kIOSurfacePixelFormat : @(format),
    };
    Cf<IOSurfaceRef> surface(IOSurfaceCreate((__bridge CFDictionaryRef)properties));
    if (surface.get() == nullptr) return unsupported(std::format("{} × {} refusée", width, rows));
    if (IOSurfaceGetBytesPerRow(surface.get()) != row_bytes)
      return unsupported(
          std::format("lignes de {} o bourrées à {} o", row_bytes, IOSurfaceGetBytesPerRow(surface.get())));
    if (IOSurfaceGetAllocSize(surface.get()) < size) return unsupported("plus petite que demandée");
    CVPixelBufferRef raw = nullptr;
    if (CVPixelBufferCreateWithIOSurface(kCFAllocatorDefault, surface.get(), nullptr, &raw) !=
            kCVReturnSuccess ||
        raw == nullptr)
      return unsupported("tampon de pixels refusé");
    Cf<CVPixelBufferRef> pixels(raw);
    if (IOSurfaceLock(surface.get(), 0, nullptr) != KERN_SUCCESS) return unsupported("verrou refusé");
    auto* data = static_cast<std::byte*>(IOSurfaceGetBaseAddress(surface.get()));
    if (data == nullptr || reinterpret_cast<std::uintptr_t>(data) % unit != 0) {
      IOSurfaceUnlock(surface.get(), 0, nullptr);
      return unsupported(std::format("adresse non alignée sur {} o", unit));
    }
    surfaces->add(data, surface, std::move(pixels), type, shape, bytes);
    return std::make_unique<SurfaceMemory>(surfaces, std::move(surface), data);
  }
}

std::expected<std::unique_ptr<DirectModel>, Refusal> open_direct(const std::filesystem::path& compiled,
                                                                 Units units,
                                                                 std::span<const std::string> inputs,
                                                                 std::span<const std::string> outputs,
                                                                 std::shared_ptr<Surfaces> surfaces) {
  @autoreleasepool {
    @try {
      MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
      config.computeUnits = compute_units(units);
      config.allowLowPrecisionAccumulationOnGPU = YES;
#if AJP_COREML_HINTS
      if (@available(macOS 14.4, *)) {
        MLOptimizationHints* hints = [[MLOptimizationHints alloc] init];
        hints.specializationStrategy = MLSpecializationStrategyFastPrediction;
        config.optimizationHints = hints;
      }
#endif
      NSURL* url = [NSURL fileURLWithPath:ns_text(compiled.string()) isDirectory:YES];
      NSError* error = nil;
      MLModel* model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
      if (model == nil)
        return refuse_direct(RefusalCode::CacheIo,
                             std::format("{} : {}", compiled.string(), describe(error)));
      const auto slots = [](NSDictionary<NSString*, MLFeatureDescription*>* descriptions,
                            std::span<const std::string> names) -> std::expected<std::vector<Slot>, Refusal> {
        std::vector<Slot> out;
        for (const std::string& name : names) {
          auto found = find(descriptions, name);
          if (!found) return std::unexpected(found.error());
          for (const Slot& s : out) {
            if ([s.name isEqualToString:found->name])
              return refuse_direct(RefusalCode::FeatureUnavailable, std::format("{} : nom en double", name));
          }
          out.push_back(Slot{.name = found->name, .type = found->type});
        }
        return out;
      };
      auto in = slots(model.modelDescription.inputDescriptionsByName, inputs);
      if (!in) return std::unexpected(in.error());
      auto out = slots(model.modelDescription.outputDescriptionsByName, outputs);
      if (!out) return std::unexpected(out.error());
      if (in->size() != model.modelDescription.inputDescriptionsByName.count)
        return refuse_direct(RefusalCode::FeatureUnavailable, "le modèle a d'autres entrées");
      return std::make_unique<Direct>(model, std::move(*in), std::move(*out), std::move(surfaces));
    } @catch (NSException* exception) {
      return refuse_direct(RefusalCode::CacheIo, describe(exception));
    }
  }
}

}  // namespace ajp::gpu::coreml_rt
