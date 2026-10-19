// Lecture de l'interface d'un module SPIR-V (spécification SPIR-V 1.6, § 2.3 et § 3) : une
// passe sur les instructions garde les types, décorations et variables utiles, puis chaque
// variable de l'interface est classée. Rien n'est exécuté : un module tronqué ou incohérent est
// refusé, jamais deviné.
#include "gpu/vulkan/compute/spirv_reflect.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <utility>

#include "gpu/vulkan/compute/pipeline.h"

namespace ajp::gpu::vk {
namespace {

constexpr std::uint32_t kMagic = 0x07230203U;
constexpr std::size_t kHeaderWords = 5;
constexpr int kMaxNesting = 8;  // structures imbriquées dans un bloc push

// Numéros de la spécification SPIR-V.
enum Op : std::uint32_t {
  kOpEntryPoint = 15,
  kOpCapability = 17,
  kOpTypeBool = 20,
  kOpTypeInt = 21,
  kOpTypeFloat = 22,
  kOpTypeVector = 23,
  kOpTypeImage = 25,
  kOpTypeArray = 28,
  kOpTypeRuntimeArray = 29,
  kOpTypeStruct = 30,
  kOpTypePointer = 32,
  kOpConstant = 43,
  kOpVariable = 59,
  kOpDecorate = 71,
  kOpMemberDecorate = 72,
};
enum Decoration : std::uint32_t {
  kSpecId = 1,
  kBlock = 2,
  kBufferBlock = 3,
  kArrayStride = 6,
  kBinding = 33,
  kDescriptorSet = 34,
  kOffset = 35,
};
enum StorageClass : std::uint32_t {
  kUniformConstant = 0,
  kUniform = 2,
  kPushConstant = 9,
  kStorageBuffer = 12,
};
constexpr std::uint32_t kGLCompute = 5;
constexpr std::uint32_t kDimBuffer = 5;
constexpr std::uint32_t kSampledStorage = 2;  // OpTypeImage « Sampled » : image de stockage

// Capacités reconnues, et ce qu'elles demandent à l'appareil.
enum Capability : std::uint32_t {
  kCapMatrix = 0,
  kCapShader = 1,
  kCapFloat16 = 9,
  kCapInt16 = 22,
  kCapImageQuery = 50,
  kCapGroupNonUniform = 61,
  kCapGroupNonUniformQuad = 68,
  kCapStorageBuffer16BitAccess = 4433,
};

struct Variable {
  std::uint32_t id = 0;
  std::uint32_t pointer = 0;
  std::uint32_t storage = 0;
};

// Ce que la passe retient du module.
struct Module {
  std::map<std::uint32_t, std::vector<std::uint32_t>> types;  // id → [opcode, opérandes après l'id]
  std::map<std::uint32_t, std::uint32_t> constants;           // OpConstant 32 bits : id → valeur
  std::map<std::uint32_t, std::uint32_t> bindings;
  std::map<std::uint32_t, std::uint32_t> sets;
  std::map<std::uint32_t, std::uint32_t> strides;
  std::map<std::uint32_t, std::uint32_t> blocks;  // id → Block ou BufferBlock
  std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> offsets;
  std::vector<Variable> variables;
  bool compute_main = false;
};

std::unexpected<Refusal> malformed(std::string_view name, std::string_view why) {
  return refuse(RefusalCode::InputMismatch, std::format("SPIR-V de {} : {}", name, why));
}

// Chaîne littérale à partir de first : octets dans l'ordre des mots, terminée par un zéro.
std::optional<std::string> literal(std::span<const std::uint32_t> operands, std::size_t first) {
  std::string out;
  for (std::size_t w = first; w < operands.size(); ++w) {
    for (std::uint32_t byte = 0; byte < 4; ++byte) {
      const auto c = static_cast<char>((operands[w] >> (8U * byte)) & 0xFFU);
      if (c == '\0') return out;
      out += c;
    }
  }
  return std::nullopt;
}

// Une instruction : opcode, et ses opérandes (sans le mot d'en-tête).
void record(Module& m, std::uint32_t op, std::span<const std::uint32_t> a) {
  const std::size_t n = a.size();
  switch (op) {
    case kOpEntryPoint:
      if (n >= 3 && a[0] == kGLCompute && literal(a, 2) == "main") m.compute_main = true;
      break;
    case kOpTypeBool:
    case kOpTypeInt:
    case kOpTypeFloat:
    case kOpTypeVector:
    case kOpTypeImage:
    case kOpTypeArray:
    case kOpTypeRuntimeArray:
    case kOpTypeStruct:
    case kOpTypePointer:
      if (n >= 1) {
        std::vector<std::uint32_t> type{op};
        type.insert(type.end(), a.begin() + 1, a.end());
        m.types[a[0]] = std::move(type);
      }
      break;
    case kOpConstant:
      if (n >= 3) m.constants[a[1]] = a[2];
      break;
    case kOpVariable:
      if (n >= 3) m.variables.push_back({.id = a[1], .pointer = a[0], .storage = a[2]});
      break;
    case kOpDecorate:
      if (n >= 3 && a[1] == kBinding) m.bindings[a[0]] = a[2];
      if (n >= 3 && a[1] == kDescriptorSet) m.sets[a[0]] = a[2];
      if (n >= 3 && a[1] == kArrayStride) m.strides[a[0]] = a[2];
      if (n >= 2 && (a[1] == kBlock || a[1] == kBufferBlock)) m.blocks[a[0]] = a[1];
      break;
    case kOpMemberDecorate:
      if (n >= 4 && a[2] == kOffset) m.offsets[{a[0], a[1]}] = a[3];
      break;
    default:
      break;
  }
}

const std::vector<std::uint32_t>* type_of(const Module& m, std::uint32_t id) {
  const auto found = m.types.find(id);
  return found == m.types.end() ? nullptr : &found->second;
}

// Octets occupés par un type d'un bloc push ; nullopt pour ce qu'on ne sait pas mesurer.
std::optional<std::uint64_t> size_of(const Module& m, std::uint32_t id, int depth);

std::optional<std::uint64_t> struct_end(const Module& m, std::uint32_t id, int depth) {
  const std::vector<std::uint32_t>* t = type_of(m, id);
  if (t == nullptr || (*t)[0] != kOpTypeStruct || depth > kMaxNesting) return std::nullopt;
  std::uint64_t end = 0;
  for (std::uint32_t member = 0; member + 1 < t->size(); ++member) {
    const auto offset = m.offsets.find({id, member});
    const std::optional<std::uint64_t> size = size_of(m, (*t)[member + 1], depth + 1);
    if (offset == m.offsets.end() || !size) return std::nullopt;
    end = std::max(end, offset->second + *size);
  }
  return end;
}

std::optional<std::uint64_t> size_of(const Module& m, std::uint32_t id, int depth) {
  const std::vector<std::uint32_t>* t = type_of(m, id);
  if (t == nullptr || depth > kMaxNesting) return std::nullopt;
  switch ((*t)[0]) {
    case kOpTypeInt:
    case kOpTypeFloat:
      return t->size() >= 2 ? std::optional<std::uint64_t>((*t)[1] / 8U) : std::nullopt;
    case kOpTypeVector: {
      if (t->size() < 3) return std::nullopt;
      const std::optional<std::uint64_t> component = size_of(m, (*t)[1], depth + 1);
      return component ? std::optional<std::uint64_t>(*component * (*t)[2]) : std::nullopt;
    }
    case kOpTypeArray: {
      const auto length = t->size() >= 3 ? m.constants.find((*t)[2]) : m.constants.end();
      const auto stride = m.strides.find(id);
      if (length == m.constants.end() || stride == m.strides.end()) return std::nullopt;
      return std::uint64_t{length->second} * stride->second;
    }
    case kOpTypeStruct:
      return struct_end(m, id, depth + 1);
    default:
      return std::nullopt;  // booléens, matrices, pointeurs : absents des blocs du socle
  }
}

std::string join(std::span<const std::uint32_t> values) {
  std::string out;
  for (const std::uint32_t v : values) out += out.empty() ? std::format("{}", v) : std::format(", {}", v);
  return out.empty() ? "aucun" : out;
}

// [first, first + count) en ordre croissant.
bool is_run(std::span<const std::uint32_t> values, std::uint32_t first, std::uint32_t count) {
  if (values.size() != count) return false;
  for (std::uint32_t i = 0; i < count; ++i) {
    if (values[i] != first + i) return false;
  }
  return true;
}

// Classe une variable de l'interface et l'ajoute à shader. Les autres classes de stockage
// (Input, Workgroup, Function, Private…) ne regardent pas la disposition.
Done classify(const Module& m, const Variable& v, ShaderInterface& shader, bool& push_seen) {
  if (v.storage != kPushConstant && v.storage != kUniformConstant && v.storage != kUniform &&
      v.storage != kStorageBuffer) {
    return {};
  }
  const std::vector<std::uint32_t>* pointer = type_of(m, v.pointer);
  if (pointer == nullptr || (*pointer)[0] != kOpTypePointer || pointer->size() < 3) {
    return malformed(shader.name, std::format("variable %{} sans type pointeur", v.id));
  }
  const std::uint32_t pointee = (*pointer)[2];
  const std::vector<std::uint32_t>* t = type_of(m, pointee);
  if (t == nullptr) return malformed(shader.name, std::format("type %{} inconnu", pointee));
  if (v.storage == kPushConstant) {
    const std::optional<std::uint64_t> end = struct_end(m, pointee, 0);
    if (push_seen || !end || *end > std::numeric_limits<std::uint32_t>::max()) {
      return malformed(shader.name, "bloc push_constant multiple ou impossible à mesurer");
    }
    push_seen = true;
    shader.push_bytes = static_cast<std::uint32_t>(*end);
    return {};
  }
  const auto set = m.sets.find(v.id);
  const auto binding = m.bindings.find(v.id);
  if (set == m.sets.end() || binding == m.bindings.end()) {
    return malformed(shader.name, std::format("descripteur %{} sans ensemble ni binding", v.id));
  }
  if (set->second != 0) {
    return malformed(shader.name, std::format("binding {} dans l'ensemble {}, 0 seul permis", binding->second,
                                              set->second));
  }
  const std::uint32_t op = (*t)[0];
  const auto block = m.blocks.find(pointee);
  const bool buffer = op == kOpTypeStruct && (v.storage == kStorageBuffer ||
                                              (block != m.blocks.end() && block->second == kBufferBlock));
  const bool image = op == kOpTypeImage && v.storage == kUniformConstant && t->size() >= 7 &&
                     (*t)[2] != kDimBuffer && (*t)[6] == kSampledStorage;
  if (!buffer && !image) {
    return malformed(shader.name,
                     std::format("binding {} : ni tampon ni image de stockage (uniform buffer, image "
                                 "échantillonnée, échantillonneur ou tableau de descripteurs)",
                                 binding->second));
  }
  for (const auto* list : {&shader.storage_buffers, &shader.storage_images}) {
    if (std::ranges::find(*list, binding->second) != list->end()) {
      return malformed(shader.name, std::format("binding {} déclaré deux fois", binding->second));
    }
  }
  (buffer ? shader.storage_buffers : shader.storage_images).push_back(binding->second);
  return {};
}

// La capacité est-elle activée sur l'appareil ?
bool supported(std::uint32_t capability, const DeviceCaps& caps) {
  switch (capability) {
    case kCapMatrix:
    case kCapShader:
    case kCapImageQuery:  // imageSize : Vulkan 1.0, sans fonction à activer
      return true;
    case kCapFloat16:
      return caps.enabled.shader_float16;
    case kCapInt16:
      return caps.enabled.shader_int16;
    case kCapStorageBuffer16BitAccess:
      return caps.enabled.storage_buffer_16bit;
    default:
      break;
  }
  // GroupNonUniform, Vote, Arithmetic, Ballot, Shuffle, ShuffleRelative, Clustered, Quad : les bits
  // de VkSubgroupFeatureFlagBits dans le même ordre.
  if (capability >= kCapGroupNonUniform && capability <= kCapGroupNonUniformQuad) {
    const std::uint32_t bit = 1U << (capability - kCapGroupNonUniform);
    return (caps.subgroup_operations & bit) != 0;
  }
  return false;
}

}  // namespace

std::expected<ShaderInterface, Refusal> reflect(std::span<const std::uint32_t> words, std::string_view name) {
  if (words.size() < kHeaderWords || words[0] != kMagic) return malformed(name, "en-tête absent");
  Module m;
  ShaderInterface shader{.name = std::string(name)};
  for (std::size_t i = kHeaderWords; i < words.size();) {
    const std::uint32_t count = words[i] >> 16U;
    const std::uint32_t op = words[i] & 0xFFFFU;
    if (count == 0 || count > words.size() - i) {
      return malformed(name, std::format("instruction tronquée au mot {}", i));
    }
    const std::span<const std::uint32_t> operands = words.subspan(i + 1, count - 1U);
    if (op == kOpCapability && !operands.empty()) shader.capabilities.push_back(operands[0]);
    if (op == kOpDecorate && operands.size() >= 3 && operands[1] == kSpecId) {
      shader.spec_ids.push_back(operands[2]);
    }
    record(m, op, operands);
    i += count;
  }
  if (!m.compute_main) return malformed(name, "pas de point d'entrée GLCompute « main »");
  bool push_seen = false;
  for (const Variable& v : m.variables) {
    if (auto ok = classify(m, v, shader, push_seen); !ok) return std::unexpected(ok.error());
  }
  std::ranges::sort(shader.storage_buffers);
  std::ranges::sort(shader.storage_images);
  std::ranges::sort(shader.spec_ids);
  if (std::ranges::adjacent_find(shader.spec_ids) != shader.spec_ids.end()) {
    return malformed(name, "SpecId déclaré deux fois");
  }
  return shader;
}

Done check_capabilities(const ShaderInterface& shader, const DeviceCaps& caps) {
  for (const std::uint32_t capability : shader.capabilities) {
    if (!supported(capability, caps)) {
      return refuse(RefusalCode::FeatureUnavailable, std::format("{} : capacité SPIR-V {} non activée sur {}",
                                                                 shader.name, capability, caps.device_name));
    }
  }
  return {};
}

Done check_interface(const ShaderInterface& shader, const PipelineDesc& desc) {
  if (!is_run(shader.storage_buffers, 0, desc.buffers) ||
      !is_run(shader.storage_images, desc.buffers, desc.images)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : le shader lie les tampons {} et les images {} ; la description, {} "
                              "tampons puis {} images",
                              shader.name, join(shader.storage_buffers), join(shader.storage_images),
                              desc.buffers, desc.images));
  }
  if (shader.push_bytes != desc.push_bytes) {
    return refuse(RefusalCode::InputMismatch, std::format("{} : bloc push de {} o, {} o décrits", shader.name,
                                                          shader.push_bytes, desc.push_bytes));
  }
  const auto constants = static_cast<std::uint32_t>(desc.constants.size());
  if (!is_run(shader.spec_ids, 0, constants)) {
    return refuse(RefusalCode::InputMismatch,
                  std::format("{} : constantes de spécialisation {} déclarées, {} données", shader.name,
                              join(shader.spec_ids), constants));
  }
  return {};
}

}  // namespace ajp::gpu::vk
