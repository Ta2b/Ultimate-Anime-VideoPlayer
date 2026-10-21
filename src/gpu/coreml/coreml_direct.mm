#include "gpu/coreml/coreml_direct.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/coreml/coreml_compiled.h"

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

// Une entrée ou une sortie du modèle et la mémoire de la chaîne qu'elle enveloppe en dernier.
struct Slot {
  NSString* name = nil;
  TensorType type = TensorType::F16;
  MLMultiArray* array = nil;
  std::byte* data = nullptr;
  std::uint64_t bytes = 0;
  Shape shape;
};

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
  Direct(MLModel* model, std::vector<Slot> inputs, std::vector<Slot> outputs)
      : model_(model), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {
    for (const Slot& s : inputs_) in_types_.push_back(s.type);
    for (const Slot& s : outputs_) out_types_.push_back(s.type);
  }

  std::span<const TensorType> input_types() const override { return in_types_; }
  std::span<const TensorType> output_types() const override { return out_types_; }

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
        id<MLFeatureProvider> result = [model_ predictionFromFeatures:features_
                                                              options:options_
                                                                error:&error];
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
  // Enveloppe la mémoire de b, une fois tant que la chaîne donne la même : vrai si l'enveloppe change.
  static std::expected<bool, Refusal> wrap(Slot& slot, const HostBinding& b) {
    if (b.type != slot.type)
      return refuse_direct(RefusalCode::InputMismatch,
                           std::format("{} : type de la liaison", slot.name.UTF8String));
    if (slot.array != nil && slot.data == b.data && slot.bytes == b.bytes && slot.shape == b.shape)
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
    NSError* error = nil;
    MLMultiArray* array = [[MLMultiArray alloc] initWithDataPointer:b.data
                                                              shape:shape
                                                           dataType:array_type(b.type)
                                                            strides:strides
                                                        deallocator:^(void* /*bytes*/) {
                                                        }
                                                              error:&error];
    if (array == nil) return refuse_direct(RefusalCode::DeviceUnavailable, describe(error));
    slot = Slot{.name = slot.name,
                .type = slot.type,
                .array = array,
                .data = b.data,
                .bytes = b.bytes,
                .shape = b.shape};
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
};

}  // namespace

std::expected<std::unique_ptr<DirectModel>, Refusal> open_direct(const std::filesystem::path& compiled,
                                                                 Units units,
                                                                 std::span<const std::string> inputs,
                                                                 std::span<const std::string> outputs) {
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
      return std::make_unique<Direct>(model, std::move(*in), std::move(*out));
    } @catch (NSException* exception) {
      return refuse_direct(RefusalCode::CacheIo, describe(exception));
    }
  }
}

}  // namespace ajp::gpu::coreml_rt
