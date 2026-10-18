#include "core/capabilities.h"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;

namespace {

const CapabilityId kHd{.kind = CapabilityKind::Model, .name = "2x_AnimeJaNai_HD_V3_Compact"};
const CapabilityId kRife{.kind = CapabilityKind::Model, .name = "rife-4.26"};
const CapabilityId kTrueHdr{.kind = CapabilityKind::Route, .name = std::string(kRouteTrueHdr)};

const Fingerprint kNow{.driver = "580.12", .artifact_sha256 = std::string(64, 'c'), .plan_hash = 0x1234};

CapabilityFacts installed(const CapabilityId& id) {
  return CapabilityFacts{.id = id, .compiled = true, .delivered = true, .installed = true, .current = kNow};
}

CapabilityFacts qualified(const CapabilityId& id) {
  CapabilityFacts facts = installed(id);
  facts.receipt = make_receipt(id, kNow, BenchResult{.pair_us = 35'400, .host_waits = 0});
  return facts;
}

}  // namespace

TEST_CASE("Reçu : son hachage couvre la capacité, l'empreinte et le banc", "[capabilities]") {
  const Receipt receipt = make_receipt(kHd, kNow, BenchResult{.pair_us = 12'000, .host_waits = 0});
  CHECK(receipt.hash.size() == 64);
  CHECK(receipt_intact(receipt));
  CHECK(make_receipt(kHd, kNow, BenchResult{.pair_us = 12'000}) == receipt);

  for (int field = 0; field < 4; ++field) {
    Receipt tampered = receipt;
    if (field == 0) tampered.capability.name = "autre";
    if (field == 1) tampered.fingerprint.driver = "581.00";
    if (field == 2) tampered.fingerprint.plan_hash += 1;
    if (field == 3) tampered.bench.host_waits = 1;
    INFO(field);
    CHECK_FALSE(receipt_intact(tampered));
  }
}

TEST_CASE("Capacités : compilée, livrée, installée, qualifiée", "[capabilities]") {
  CapabilityFacts facts{.id = kHd};
  CHECK(state_of(facts) == CapabilityState::Unavailable);
  facts.compiled = true;
  CHECK(state_of(facts) == CapabilityState::Compiled);
  facts.delivered = true;
  CHECK(state_of(facts) == CapabilityState::Delivered);
  facts.installed = true;
  facts.current = kNow;
  CHECK(state_of(facts) == CapabilityState::Installed);
  facts.receipt = make_receipt(kHd, kNow, {});
  CHECK(state_of(facts) == CapabilityState::Qualified);

  // Un SDK importé est installé sans avoir été livré par l'index.
  CHECK(state_of(CapabilityFacts{.id = kTrueHdr, .installed = true}) == CapabilityState::Installed);
}

TEST_CASE("Capacités : un reçu périmé ou étranger ne qualifie pas", "[capabilities]") {
  CapabilityFacts new_driver = qualified(kHd);
  new_driver.current.driver = "590.01";
  CHECK(state_of(new_driver) == CapabilityState::Installed);

  CapabilityFacts other = installed(kHd);
  other.receipt = make_receipt(kRife, kNow, {});
  CHECK(state_of(other) == CapabilityState::Installed);

  CapabilityFacts forged = qualified(kHd);
  forged.receipt->bench.pair_us = 1;
  CHECK(state_of(forged) == CapabilityState::Installed);

  CapabilityFacts removed = qualified(kHd);
  removed.installed = false;
  CHECK(state_of(removed) == CapabilityState::Delivered);
}

TEST_CASE("Panneau : modèles puis routes, reçu affiché seulement si qualifié", "[capabilities]") {
  CapabilityFacts stale = qualified(kRife);
  stale.current.artifact_sha256 = std::string(64, 'd');
  const std::vector<CapabilityFacts> facts{qualified(kTrueHdr), qualified(kHd), stale,
                                           CapabilityFacts{.id = {.name = "4x_absent"}, .delivered = true}};
  const std::vector<CapabilityRow> rows = panel_rows(facts);
  REQUIRE(rows.size() == 4);
  CHECK(rows[0].id == kHd);
  CHECK(rows[0].state == CapabilityState::Qualified);
  CHECK(rows[0].receipt_hash == facts[1].receipt->hash);
  CHECK(rows[1].id == kRife);
  CHECK(rows[1].state == CapabilityState::Installed);
  CHECK(rows[1].usable);
  CHECK(rows[1].receipt_stale);
  CHECK(rows[1].receipt_hash.empty());
  CHECK(rows[2].state == CapabilityState::Delivered);
  CHECK_FALSE(rows[2].usable);
  CHECK(rows[3].id == kTrueHdr);
}

TEST_CASE("Capacités : resolve() charge une capacité installée, qualifiée ou non", "[capabilities]") {
  const std::vector<CapabilityFacts> facts{installed(kHd), qualified(kRife), installed(kTrueHdr),
                                           CapabilityFacts{.id = {.name = "4x_absent"}, .delivered = true}};
  const Capabilities caps = to_resolve_capabilities(facts);
  CHECK(caps.models == std::vector<ModelId>{kHd.name, kRife.name});
  CHECK(caps.true_hdr);
  CHECK_FALSE(to_resolve_capabilities(std::vector{qualified(kHd)}).true_hdr);

  Profile profile{.hdr = HdrMode::TrueHdr};
  profile.rules.push_back(Rule{.max_pixels = limits::kPixels.hi,
                               .max_fps = limits::kFps.hi,
                               .steps = {Step{.kind = StepKind::Upscale, .model = kHd.name, .factor = {2, 1}},
                                         Step{.kind = StepKind::Interpolate,
                                              .model = kRife.name,
                                              .factor = {2, 1},
                                              .policy = Policy::Fast}}});
  auto plan = resolve(profile, caps, SourceInfo{.width = 1280, .height = 720, .fps = 23.976},
                      DisplayInfo{.width = 2560, .height = 1440, .refresh_hz = 144.0, .hdr = true});
  REQUIRE(plan);
  CHECK(unqualified_in(*plan, facts) == std::vector{kHd, kTrueHdr});
}

TEST_CASE("receipts.toml : écrire puis relire ; un reçu altéré est refusé", "[capabilities]") {
  const std::vector<Receipt> receipts{
      make_receipt(kHd, kNow, BenchResult{.pair_us = 35'400, .host_waits = 0}),
      make_receipt(kTrueHdr, Fingerprint{.driver = "580.12", .plan_hash = 0xFFFFFFFFFFFFFFFF},
                   {.host_waits = 3})};
  const std::string text = format_receipts(receipts);
  auto reread = parse_receipts(text);
  REQUIRE(reread);
  CHECK(*reread == receipts);

  std::string tampered = text;
  tampered.replace(tampered.find("35400"), 5, "30000");
  auto refused = parse_receipts(tampered);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::HashMismatch);

  const auto refusal_of = [](const std::string& body) {
    auto result = parse_receipts(body);
    REQUIRE_FALSE(result);
    return result.error().code;
  };
  const std::string fields = "name = \"m\"\ndriver = \"d\"\nartifact_sha256 = \"\"\nhash = \"h\"\n";
  CHECK(refusal_of("version = 2\n") == RefusalCode::FormatVersion);
  CHECK(refusal_of("version = 1\n[[receipt]]\nkind = \"engine\"\n" + fields) == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("version = 1\n[[receipt]]\nkind = \"model\"\n" + fields + "plan_hash = \"xyz\"\n") ==
        RefusalCode::FieldType);
  CHECK(refusal_of("version = 1\n[[receipt]]\nkind = \"model\"\n" + fields +
                   "plan_hash = \"0000000000000000\"\npair_us = -1\nhost_waits = 0\n") ==
        RefusalCode::FieldOutOfRange);
  CHECK(parse_receipts("version = 1\n")->empty());
}
