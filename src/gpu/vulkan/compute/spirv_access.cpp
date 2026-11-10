#include "gpu/vulkan/compute/spirv_access.h"

#include <map>
#include <set>
#include <utility>
#include <vector>

namespace ajp::gpu::vk {
namespace {

constexpr std::uint32_t kMagic = 0x07230203;
constexpr std::uint32_t kOpTypeStruct = 30;
constexpr std::uint32_t kOpTypePointer = 32;
constexpr std::uint32_t kOpVariable = 59;
constexpr std::uint32_t kOpDecorate = 71;
constexpr std::uint32_t kOpMemberDecorate = 72;
constexpr std::uint32_t kNonWritable = 24;
constexpr std::uint32_t kBinding = 33;
constexpr std::uint32_t kDescriptorSet = 34;
constexpr std::uint32_t kUniform = 2;  // bloc BufferBlock des anciens SPIR-V
constexpr std::uint32_t kStorageBuffer = 12;
constexpr std::uint32_t kOpExtension = 10;
constexpr std::uint32_t kOpEntryPoint = 15;
constexpr std::uint32_t kOpExecutionMode = 16;
constexpr std::uint32_t kOpCapability = 17;
constexpr std::uint32_t kOpTypeFloat = 22;
constexpr std::uint32_t kModeDenormPreserve = 4459;
constexpr std::uint32_t kModeRoundingRte = 4462;
constexpr std::uint32_t kCapabilityDenormPreserve = 4464;
constexpr std::uint32_t kCapabilityRoundingRte = 4467;
constexpr std::uint32_t kSpirv14 = 0x00010400;  // float_controls y entre dans le cœur

constexpr std::uint32_t head(std::uint32_t op, std::uint32_t count) {
  return (count << 16U) | op;
}

// Ce qu'il faut savoir du module pour y glisser les modes : où finissent les capacités et les points
// d'entrée, lesquels, et s'il range du float 16 bits.
struct ModuleMap {
  std::size_t after_capabilities = 0;
  std::size_t after_entry_points = 0;
  std::vector<std::uint32_t> entry_points;
  bool float16 = false;
  bool readable = false;
};

ModuleMap map_module(std::span<const std::uint32_t> words) {
  ModuleMap map;
  if (words.size() < 5 || words[0] != kMagic) return map;
  for (std::size_t at = 5; at < words.size();) {
    const std::uint32_t count = words[at] >> 16U;
    const std::uint32_t op = words[at] & 0xFFFFU;
    if (count == 0 || at + count > words.size()) return map;
    if (op == kOpCapability) map.after_capabilities = at + count;
    if (op == kOpEntryPoint && count >= 3) {
      map.entry_points.push_back(words[at + 2]);
      map.after_entry_points = at + count;
    }
    if (op == kOpTypeFloat && count >= 3 && words[at + 2] == 16) map.float16 = true;
    at += count;
  }
  map.readable = map.after_capabilities != 0 && map.after_entry_points > map.after_capabilities;
  return map;
}

}  // namespace

std::uint64_t spirv_readonly_bindings(std::span<const std::uint32_t> words) {
  if (words.size() < 5 || words[0] != kMagic) return 0;
  std::map<std::uint32_t, std::uint32_t> binding;
  std::map<std::uint32_t, std::uint32_t> set;
  std::set<std::uint32_t> non_writable;                            // variables
  std::map<std::uint32_t, std::set<std::uint32_t>> ro;             // struct → membres NonWritable
  std::map<std::uint32_t, std::size_t> members;                    // struct → nombre de membres
  std::map<std::uint32_t, std::uint32_t> pointee;                  // pointeur → type
  std::vector<std::pair<std::uint32_t, std::uint32_t>> variables;  // (type pointeur, variable)
  for (std::size_t at = 5; at < words.size();) {
    const std::uint32_t count = words[at] >> 16U;
    const std::uint32_t op = words[at] & 0xFFFFU;
    if (count == 0 || at + count > words.size()) return 0;
    const std::span<const std::uint32_t> w = words.subspan(at, count);
    if (op == kOpDecorate && count >= 3) {
      if (w[2] == kNonWritable) non_writable.insert(w[1]);
      if (w[2] == kBinding && count >= 4) binding[w[1]] = w[3];
      if (w[2] == kDescriptorSet && count >= 4) set[w[1]] = w[3];
    } else if (op == kOpMemberDecorate && count >= 4 && w[3] == kNonWritable) {
      ro[w[1]].insert(w[2]);
    } else if (op == kOpTypeStruct && count >= 2) {
      members[w[1]] = count - 2;
    } else if (op == kOpTypePointer && count >= 4 && (w[2] == kStorageBuffer || w[2] == kUniform)) {
      pointee[w[1]] = w[3];
    } else if (op == kOpVariable && count >= 4 && (w[3] == kStorageBuffer || w[3] == kUniform)) {
      variables.emplace_back(w[1], w[2]);
    }
    at += count;
  }
  std::uint64_t bound = 0;
  std::uint64_t written = 0;
  for (const auto& [pointer, variable] : variables) {
    const auto b = binding.find(variable);
    if (b == binding.end() || b->second >= 64) continue;
    if (const auto s = set.find(variable); s != set.end() && s->second != 0) continue;
    const std::uint64_t bit = std::uint64_t{1} << b->second;
    bound |= bit;
    bool readonly = non_writable.contains(variable);
    if (!readonly) {
      const auto type = pointee.find(pointer);
      if (type != pointee.end()) {
        const auto n = members.find(type->second);
        const auto marked = ro.find(type->second);
        readonly =
            n != members.end() && n->second > 0 && marked != ro.end() && marked->second.size() == n->second;
      }
    }
    if (!readonly) written |= bit;
  }
  return bound & ~written;
}

std::vector<std::uint32_t> spirv_with_float16_controls(std::span<const std::uint32_t> words, bool rte,
                                                       bool denorm) {
  if (!rte && !denorm) return {};
  const ModuleMap map = map_module(words);
  if (!map.readable || !map.float16) return {};
  std::vector<std::uint32_t> out;
  out.reserve(words.size() + 32);
  const auto copy = [&](std::size_t from, std::size_t to) {
    out.insert(out.end(), words.begin() + static_cast<std::ptrdiff_t>(from),
               words.begin() + static_cast<std::ptrdiff_t>(to));
  };
  copy(0, map.after_capabilities);
  if (rte) out.insert(out.end(), {head(kOpCapability, 2), kCapabilityRoundingRte});
  if (denorm) out.insert(out.end(), {head(kOpCapability, 2), kCapabilityDenormPreserve});
  if (words[1] < kSpirv14) {
    // « SPV_KHR_float_controls » et son zéro final, par mots de quatre octets, petit-boutiste.
    static constexpr char kName[] = "SPV_KHR_float_controls\0";  // 24 octets avec les deux zéros
    out.push_back(head(kOpExtension, 1 + sizeof(kName) / 4));
    for (std::size_t i = 0; i + 4 <= sizeof(kName); i += 4) {
      std::uint32_t word = 0;
      for (std::size_t b = 0; b < 4; ++b)
        word |= static_cast<std::uint32_t>(static_cast<unsigned char>(kName[i + b])) << (8U * b);
      out.push_back(word);
    }
  }
  copy(map.after_capabilities, map.after_entry_points);
  for (const std::uint32_t entry : map.entry_points) {
    if (rte) out.insert(out.end(), {head(kOpExecutionMode, 4), entry, kModeRoundingRte, 16U});
    if (denorm) out.insert(out.end(), {head(kOpExecutionMode, 4), entry, kModeDenormPreserve, 16U});
  }
  copy(map.after_entry_points, words.size());
  return out;
}

}  // namespace ajp::gpu::vk
