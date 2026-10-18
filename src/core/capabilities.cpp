#include "core/capabilities.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <utility>

#include "core/model_index.h"
#include "core/toml_fields.h"

namespace ajp::core {
namespace {

using fields::refuse;

constexpr std::array<std::pair<std::string_view, CapabilityKind>, 2> kKinds{
    {{"model", CapabilityKind::Model}, {"route", CapabilityKind::Route}}};

std::string_view kind_name(CapabilityKind kind) {
  return kind == CapabilityKind::Model ? "model" : "route";
}

// Texte haché d'un reçu : chaque champ préfixé de sa longueur, aucune ambiguïté possible.
std::string canonical(const Receipt& receipt) {
  const auto field = [](std::string_view key, std::string_view value) {
    return std::format("{}={}:{}\n", key, value.size(), value);
  };
  return "ajp-receipt-1\n" + field("kind", kind_name(receipt.capability.kind)) +
         field("name", receipt.capability.name) + field("driver", receipt.fingerprint.driver) +
         field("artifact_sha256", receipt.fingerprint.artifact_sha256) +
         field("plan_hash", std::format("{:016x}", receipt.fingerprint.plan_hash)) +
         field("pair_us", std::format("{}", receipt.bench.pair_us)) +
         field("host_waits", std::format("{}", receipt.bench.host_waits));
}

bool qualifies(const CapabilityFacts& facts) {
  return facts.receipt && receipt_intact(*facts.receipt) && facts.receipt->capability == facts.id &&
         facts.receipt->fingerprint == facts.current;
}

// Installée ou qualifiée : se charge, que le reçu existe ou non.
bool usable(const CapabilityFacts& facts) {
  const CapabilityState state = state_of(facts);
  return state == CapabilityState::Installed || state == CapabilityState::Qualified;
}

const CapabilityFacts* find_facts(std::span<const CapabilityFacts> facts, const CapabilityId& id) {
  const auto found = std::ranges::find(facts, id, &CapabilityFacts::id);
  return found == facts.end() ? nullptr : &*found;
}

std::expected<std::uint64_t, Refusal> counter(const toml::table& table, std::string_view key,
                                              const std::string& where) {
  auto value = fields::required<std::int64_t>(table, key, where);
  if (!value) return std::unexpected(value.error());
  if (*value < 0) return refuse(RefusalCode::FieldOutOfRange, std::format("{}.{}", where, key));
  return static_cast<std::uint64_t>(*value);
}

std::expected<Receipt, Refusal> parse_receipt(const toml::table& table, const std::string& where) {
  Receipt receipt;
  auto kind = fields::required<std::string>(table, "kind", where);
  if (!kind) return std::unexpected(kind.error());
  const auto* found = std::ranges::find(kKinds, *kind, &std::pair<std::string_view, CapabilityKind>::first);
  if (found == kKinds.end())
    return refuse(RefusalCode::FieldOutOfRange, std::format("{}.kind = « {} »", where, *kind));
  receipt.capability.kind = found->second;

  for (auto [key, target] : {std::pair{"name", &receipt.capability.name},
                             {"driver", &receipt.fingerprint.driver},
                             {"artifact_sha256", &receipt.fingerprint.artifact_sha256},
                             {"hash", &receipt.hash}}) {
    auto value = fields::required<std::string>(table, key, where);
    if (!value) return std::unexpected(value.error());
    *target = std::move(*value);
  }
  auto plan_hash = fields::required<std::string>(table, "plan_hash", where);
  if (!plan_hash) return std::unexpected(plan_hash.error());
  const char* end = plan_hash->data() + plan_hash->size();
  const auto parsed = std::from_chars(plan_hash->data(), end, receipt.fingerprint.plan_hash, 16);
  if (plan_hash->size() != 16 || parsed.ec != std::errc{} || parsed.ptr != end) {
    return refuse(RefusalCode::FieldType, where + ".plan_hash");
  }
  auto pair_us = counter(table, "pair_us", where);
  if (!pair_us) return std::unexpected(pair_us.error());
  receipt.bench.pair_us = *pair_us;
  auto host_waits = counter(table, "host_waits", where);
  if (!host_waits) return std::unexpected(host_waits.error());
  receipt.bench.host_waits = *host_waits;

  if (!receipt_intact(receipt)) return refuse(RefusalCode::HashMismatch, where + ".hash");
  return receipt;
}

}  // namespace

Receipt make_receipt(CapabilityId capability, Fingerprint fingerprint, BenchResult bench) {
  Receipt receipt{.capability = std::move(capability), .fingerprint = std::move(fingerprint), .bench = bench};
  receipt.hash = sha256_hex(canonical(receipt));
  return receipt;
}

bool receipt_intact(const Receipt& receipt) {
  return receipt.hash == sha256_hex(canonical(receipt));
}

CapabilityState state_of(const CapabilityFacts& facts) {
  // Sans le code qui l'exécute, des fichiers installés ne servent à rien.
  if (!facts.compiled) return CapabilityState::Unavailable;
  if (facts.installed) return qualifies(facts) ? CapabilityState::Qualified : CapabilityState::Installed;
  if (facts.delivered) return CapabilityState::Delivered;
  return CapabilityState::Compiled;
}

std::vector<CapabilityRow> panel_rows(std::span<const CapabilityFacts> facts) {
  std::vector<CapabilityRow> rows;
  for (const CapabilityKind kind : {CapabilityKind::Model, CapabilityKind::Route}) {
    for (const CapabilityFacts& one : facts) {
      if (one.id.kind != kind) continue;
      const CapabilityState state = state_of(one);
      rows.push_back(CapabilityRow{
          .id = one.id,
          .state = state,
          .usable = usable(one),
          .receipt_stale = one.receipt.has_value() && !qualifies(one),
          .receipt_hash = state == CapabilityState::Qualified && one.receipt ? one.receipt->hash : ""});
    }
  }
  return rows;
}

Capabilities to_resolve_capabilities(std::span<const CapabilityFacts> facts) {
  Capabilities caps;
  for (const CapabilityFacts& one : facts) {
    if (!usable(one)) continue;
    if (one.id.kind == CapabilityKind::Model) caps.models.push_back(one.id.name);
    if (one.id.kind == CapabilityKind::Route && one.id.name == kRouteTrueHdr) caps.true_hdr = true;
  }
  return caps;
}

std::vector<CapabilityId> unqualified_in(const Plan& plan, std::span<const CapabilityFacts> facts) {
  std::vector<CapabilityId> used;
  const auto use = [&](CapabilityKind kind, std::string_view name) {
    CapabilityId id{.kind = kind, .name = std::string(name)};
    if (!std::ranges::contains(used, id)) used.push_back(std::move(id));
  };
  if (plan.upscale) {
    for (const ModelPass& pass : plan.upscale->passes) use(CapabilityKind::Model, pass.model);
  }
  for (const InterpLevel& level : plan.levels) use(CapabilityKind::Model, level.model);
  if (plan.hdr.kind == HdrMode::Native) use(CapabilityKind::Route, kRouteHdrNative);
  if (plan.hdr.kind == HdrMode::TrueHdr) use(CapabilityKind::Route, kRouteTrueHdr);

  std::erase_if(used, [&](const CapabilityId& id) {
    const CapabilityFacts* one = find_facts(facts, id);
    return one != nullptr && state_of(*one) == CapabilityState::Qualified;
  });
  return used;
}

std::expected<std::vector<Receipt>, Refusal> parse_receipts(std::string_view toml_text) {
  auto root = fields::parse(toml_text, RefusalCode::FileUnreadable, "receipts.toml");
  if (!root) return std::unexpected(root.error());
  if (auto version = fields::check_version(*root, kReceiptsVersion, "receipts.toml"); !version) {
    return std::unexpected(version.error());
  }
  auto tables = fields::array(*root, "receipt", "receipts.toml");
  if (!tables) return std::unexpected(tables.error());
  std::vector<Receipt> receipts;
  for (std::size_t i = 0; i < (*tables)->size(); ++i) {
    const std::string where = std::format("receipt[{}]", i);
    const toml::table* table = (*tables)->get(i)->as_table();
    if (table == nullptr) return refuse(RefusalCode::FieldType, where);
    auto receipt = parse_receipt(*table, where);
    if (!receipt) return std::unexpected(receipt.error());
    receipts.push_back(std::move(*receipt));
  }
  return receipts;
}

std::string format_receipts(std::span<const Receipt> receipts) {
  toml::table root{{"version", kReceiptsVersion}};
  toml::array list;
  for (const Receipt& receipt : receipts) {
    list.push_back(toml::table{
        {"kind", std::string(kind_name(receipt.capability.kind))},
        {"name", receipt.capability.name},
        {"driver", receipt.fingerprint.driver},
        {"artifact_sha256", receipt.fingerprint.artifact_sha256},
        {"plan_hash", std::format("{:016x}", receipt.fingerprint.plan_hash)},
        {"pair_us", static_cast<std::int64_t>(receipt.bench.pair_us)},
        {"host_waits", static_cast<std::int64_t>(receipt.bench.host_waits)},
        {"hash", receipt.hash},
    });
  }
  root.insert("receipt", std::move(list));
  return fields::format(root);
}

}  // namespace ajp::core
