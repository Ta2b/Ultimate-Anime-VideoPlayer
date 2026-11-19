#include "spirv_shared.h"

#include <vector>

namespace kirari::gpu::vk::test {
namespace {

constexpr std::uint32_t kMagic = 0x07230203;
constexpr std::uint64_t kMask = 0xFFFFFFFFULL;  // les constantes évaluées sont des uint 32 bits
constexpr int kMaxDepth = 64;

// Instructions (spirv.h, version 1.6).
constexpr std::uint32_t kOpTypeBool = 20;
constexpr std::uint32_t kOpTypeInt = 21;
constexpr std::uint32_t kOpTypeFloat = 22;
constexpr std::uint32_t kOpTypeVector = 23;
constexpr std::uint32_t kOpTypeArray = 28;
constexpr std::uint32_t kOpTypeStruct = 30;
constexpr std::uint32_t kOpTypePointer = 32;
constexpr std::uint32_t kOpConstantTrue = 41;
constexpr std::uint32_t kOpConstantFalse = 42;
constexpr std::uint32_t kOpConstant = 43;
constexpr std::uint32_t kOpConstantComposite = 44;
constexpr std::uint32_t kOpSpecConstantTrue = 48;
constexpr std::uint32_t kOpSpecConstantFalse = 49;
constexpr std::uint32_t kOpSpecConstant = 50;
constexpr std::uint32_t kOpSpecConstantComposite = 51;
constexpr std::uint32_t kOpSpecConstantOp = 52;
constexpr std::uint32_t kOpVariable = 59;
constexpr std::uint32_t kOpDecorate = 71;
constexpr std::uint32_t kDecorationSpecId = 1;
constexpr std::uint32_t kStorageWorkgroup = 4;
// Opérations d'OpSpecConstantOp.
constexpr std::uint32_t kCompositeExtract = 81;
constexpr std::uint32_t kIAdd = 128;
constexpr std::uint32_t kISub = 130;
constexpr std::uint32_t kIMul = 132;
constexpr std::uint32_t kUDiv = 134;
constexpr std::uint32_t kUMod = 137;
constexpr std::uint32_t kLogicalEqual = 164;
constexpr std::uint32_t kLogicalNotEqual = 165;
constexpr std::uint32_t kLogicalOr = 166;
constexpr std::uint32_t kLogicalAnd = 167;
constexpr std::uint32_t kLogicalNot = 168;
constexpr std::uint32_t kSelect = 169;
constexpr std::uint32_t kIEqual = 170;
constexpr std::uint32_t kINotEqual = 171;
constexpr std::uint32_t kUGreaterThan = 172;
constexpr std::uint32_t kUGreaterThanEqual = 174;
constexpr std::uint32_t kULessThan = 176;
constexpr std::uint32_t kULessThanEqual = 178;
constexpr std::uint32_t kShiftRightLogical = 194;
constexpr std::uint32_t kShiftLeftLogical = 196;
constexpr std::uint32_t kBitwiseOr = 197;
constexpr std::uint32_t kBitwiseAnd = 199;

struct Module {
  std::map<std::uint32_t, std::vector<std::uint32_t>> defs;  // identifiant → son instruction
  std::map<std::uint32_t, std::uint32_t> spec_ids;           // constante → SpecId
  std::vector<std::uint32_t> shared;                         // types pointeur des variables Workgroup
};

std::uint32_t opcode(const std::vector<std::uint32_t>& w) {
  return w[0] & 0xFFFFU;
}

// L'identifiant que définit l'instruction : en tête pour un type, après le type de résultat sinon.
std::optional<std::uint32_t> result_id(std::uint32_t op, std::span<const std::uint32_t> w) {
  if (op >= kOpTypeBool && op <= kOpTypePointer && w.size() >= 2) return w[1];
  if (op >= kOpConstantTrue && op <= kOpSpecConstantOp && w.size() >= 3) return w[2];
  return std::nullopt;
}

std::optional<Module> parse(std::span<const std::uint32_t> words) {
  if (words.size() < 5 || words[0] != kMagic) return std::nullopt;
  Module m;
  for (std::size_t at = 5; at < words.size();) {
    const std::uint32_t count = words[at] >> 16U;
    if (count == 0 || at + count > words.size()) return std::nullopt;
    const std::span<const std::uint32_t> w = words.subspan(at, count);
    const std::uint32_t op = w[0] & 0xFFFFU;
    if (op == kOpDecorate && count >= 4 && w[2] == kDecorationSpecId) m.spec_ids[w[1]] = w[3];
    if (op == kOpVariable && count >= 4 && w[3] == kStorageWorkgroup) m.shared.push_back(w[1]);
    if (const auto id = result_id(op, w)) m.defs[*id] = std::vector<std::uint32_t>(w.begin(), w.end());
    at += count;
  }
  return m;
}

class Evaluator {
 public:
  Evaluator(const Module& m, const std::map<std::uint32_t, std::uint32_t>& constants)
      : m_(m), constants_(constants) {}

  std::optional<std::uint64_t> value(std::uint32_t id, int depth = 0) const {
    const auto found = m_.defs.find(id);
    if (depth > kMaxDepth || found == m_.defs.end()) return std::nullopt;
    const std::vector<std::uint32_t>& w = found->second;
    switch (opcode(w)) {
      case kOpConstant:
        return w.size() >= 4 ? std::optional<std::uint64_t>(w[3]) : std::nullopt;
      case kOpConstantTrue:
        return 1;
      case kOpConstantFalse:
        return 0;
      case kOpSpecConstant:
        return w.size() >= 4 ? std::optional<std::uint64_t>(spec(id, w[3])) : std::nullopt;
      case kOpSpecConstantTrue:
        return spec(id, 1) != 0 ? 1 : 0;
      case kOpSpecConstantFalse:
        return spec(id, 0) != 0 ? 1 : 0;
      case kOpSpecConstantOp:
        return operation(w, depth + 1);
      default:
        return std::nullopt;
    }
  }

  std::optional<std::uint64_t> type_bytes(std::uint32_t type, int depth = 0) const {
    const auto found = m_.defs.find(type);
    if (depth > kMaxDepth || found == m_.defs.end()) return std::nullopt;
    const std::vector<std::uint32_t>& w = found->second;
    const std::uint32_t op = opcode(w);
    if ((op == kOpTypeInt || op == kOpTypeFloat) && w.size() >= 3) return w[2] / 8;
    if (op == kOpTypeVector && w.size() >= 4) {
      const auto e = type_bytes(w[2], depth + 1);
      return e ? std::optional<std::uint64_t>(*e * w[3]) : std::nullopt;
    }
    if (op == kOpTypeArray && w.size() >= 4) {
      const auto e = type_bytes(w[2], depth + 1);
      const auto n = value(w[3]);
      return e && n ? std::optional<std::uint64_t>(*e * *n) : std::nullopt;
    }
    if (op == kOpTypeStruct) {
      std::uint64_t sum = 0;
      for (std::size_t k = 2; k < w.size(); ++k) {
        const auto e = type_bytes(w[k], depth + 1);
        if (!e) return std::nullopt;
        sum += *e;
      }
      return sum;
    }
    return std::nullopt;
  }

 private:
  std::uint64_t spec(std::uint32_t id, std::uint64_t fallback) const {
    const auto sid = m_.spec_ids.find(id);
    if (sid == m_.spec_ids.end()) return fallback;
    const auto given = constants_.find(sid->second);
    return given == constants_.end() ? fallback : given->second;
  }

  // Le constituant index d'une constante composée (gl_WorkGroupSize).
  std::optional<std::uint64_t> extract(std::uint32_t composite, std::uint32_t index, int depth) const {
    const auto found = m_.defs.find(composite);
    if (found == m_.defs.end()) return std::nullopt;
    const std::vector<std::uint32_t>& c = found->second;
    const std::uint32_t op = opcode(c);
    if ((op != kOpConstantComposite && op != kOpSpecConstantComposite) || 3 + index >= c.size())
      return std::nullopt;
    return value(c[3 + index], depth);
  }

  std::optional<std::uint64_t> operation(const std::vector<std::uint32_t>& w, int depth) const {
    if (w.size() < 5) return std::nullopt;
    const std::uint32_t op = w[3];
    if (op == kCompositeExtract) return w.size() >= 6 ? extract(w[4], w[5], depth) : std::nullopt;
    const auto a = value(w[4], depth);
    if (!a) return std::nullopt;
    if (op == kLogicalNot) return *a == 0 ? 1 : 0;
    if (w.size() < 6) return std::nullopt;
    const auto b = value(w[5], depth);
    if (!b) return std::nullopt;
    if (op == kSelect) {
      if (w.size() < 7) return std::nullopt;
      return *a != 0 ? b : value(w[6], depth);
    }
    return binary(op, *a, *b);
  }

  static std::optional<std::uint64_t> binary(std::uint32_t op, std::uint64_t a, std::uint64_t b) {
    switch (op) {
      case kIAdd:
        return (a + b) & kMask;
      case kISub:
        return (a - b) & kMask;
      case kIMul:
        return (a * b) & kMask;
      case kUDiv:
        return b == 0 ? std::nullopt : std::optional<std::uint64_t>(a / b);
      case kUMod:
        return b == 0 ? std::nullopt : std::optional<std::uint64_t>(a % b);
      case kIEqual:
      case kLogicalEqual:
        return a == b ? 1 : 0;
      case kINotEqual:
      case kLogicalNotEqual:
        return a != b ? 1 : 0;
      case kUGreaterThan:
        return a > b ? 1 : 0;
      case kUGreaterThanEqual:
        return a >= b ? 1 : 0;
      case kULessThan:
        return a < b ? 1 : 0;
      case kULessThanEqual:
        return a <= b ? 1 : 0;
      case kLogicalOr:
        return (a != 0 || b != 0) ? 1 : 0;
      case kLogicalAnd:
        return (a != 0 && b != 0) ? 1 : 0;
      case kShiftRightLogical:
        return b < 32 ? std::optional<std::uint64_t>(a >> b) : std::nullopt;
      case kShiftLeftLogical:
        return b < 32 ? std::optional<std::uint64_t>((a << b) & kMask) : std::nullopt;
      case kBitwiseOr:
        return a | b;
      case kBitwiseAnd:
        return a & b;
      default:
        return std::nullopt;
    }
  }

  const Module& m_;
  const std::map<std::uint32_t, std::uint32_t>& constants_;
};

}  // namespace

std::optional<std::uint64_t> spirv_shared_bytes(std::span<const std::uint32_t> words,
                                                const std::map<std::uint32_t, std::uint32_t>& constants) {
  const std::optional<Module> m = parse(words);
  if (!m) return std::nullopt;
  const Evaluator eval(*m, constants);
  std::uint64_t sum = 0;
  for (const std::uint32_t pointer : m->shared) {
    const auto found = m->defs.find(pointer);
    if (found == m->defs.end() || opcode(found->second) != kOpTypePointer || found->second.size() < 4)
      return std::nullopt;
    const auto bytes = eval.type_bytes(found->second[3]);
    if (!bytes) return std::nullopt;
    sum += *bytes;
  }
  return sum;
}

std::map<std::uint32_t, std::uint32_t> spec_constants(std::span<const std::uint32_t> constants) {
  std::map<std::uint32_t, std::uint32_t> out;
  for (std::size_t i = 0; i < constants.size(); ++i) out[static_cast<std::uint32_t>(i)] = constants[i];
  return out;
}

}  // namespace kirari::gpu::vk::test
