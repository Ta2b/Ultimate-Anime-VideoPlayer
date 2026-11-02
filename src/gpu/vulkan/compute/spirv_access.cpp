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

}  // namespace ajp::gpu::vk
