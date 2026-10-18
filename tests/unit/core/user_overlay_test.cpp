#include "core/user_overlay.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::core;
namespace fs = std::filesystem;

namespace {

constexpr const char* kBase = R"(
version = 5
start_profile = 1

[[profile]]
slot = 1
name = "Qualité"
hdr = "native"

[[profile.rule]]
max_pixels = 2073600
steps = [{ kind = "upscale", model = "2x_HD", factor = 2 }]

[[profile]]
slot = 2
name = "Rapide"

[[profile.rule]]
steps = [{ kind = "interpolate", model = "rife-4.26", factor = 2, policy = "fast" }]

[keys]
toggle_osd = "F1"
)";

fs::path scratch(const std::string& name) {
  const fs::path dir = fs::current_path() / "test-tmp" / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

std::string read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

void write(const fs::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

LayeredProfile layered(const std::string& user) {
  auto result = parse_layered(kBase, user);
  REQUIRE(result);
  return *result;
}

std::string edited(const std::string& user, std::vector<UserEdit> edits) {
  auto text = apply_edits(kBase, user, edits);
  REQUIRE(text);
  return *text;
}

RefusalCode edit_refusal(const std::string& user, std::vector<UserEdit> edits) {
  auto text = apply_edits(kBase, user, edits);
  REQUIRE_FALSE(text);
  return text.error().code;
}

std::vector<Rule> interpolation_x3() {
  return {Rule{.max_pixels = limits::kPixels.hi,
               .max_fps = limits::kFps.hi,
               .steps = {Step{.kind = StepKind::Interpolate,
                              .model = "rife-4.26",
                              .factor = {3, 1},
                              .policy = Policy::Quality}}}};
}

}  // namespace

TEST_CASE("Superposition : sans user.toml, la base telle quelle", "[user_overlay]") {
  const LayeredProfile base = layered("");
  auto direct = parse_profile(kBase);
  REQUIRE(direct);
  CHECK(base.profile == *direct);
  CHECK(base.keys.chord(Action::ToggleOsd) == Chord{.key = Key::F1});
  CHECK(base.keys.chord(Action::TogglePause) == Chord{.key = Key::Space});
}

TEST_CASE("Superposition : user.toml complète la base clé par clé, slot par slot", "[user_overlay]") {
  const LayeredProfile merged = layered(R"toml(
start_profile = 2
[[profile]]
slot = 2
name = "Rapide (mien)"
[[profile]]
slot = 5
name = "Nouveau"
[keys]
toggle_grade = "Ctrl+H"
)toml");
  CHECK(merged.profile.start_profile == 2);
  REQUIRE(merged.profile.profiles[1]);
  CHECK(merged.profile.profiles[1]->name == "Rapide (mien)");
  REQUIRE(merged.profile.profiles[1]->rules.size() == 1);  // les règles de la base restent
  CHECK(merged.profile.profiles[1]->rules[0].steps[0].policy == Policy::Fast);
  REQUIRE(merged.profile.profiles[4]);
  CHECK(merged.profile.profiles[4]->name == "Nouveau");
  CHECK(merged.profile.profiles[0]->hdr == HdrMode::Native);
  CHECK(merged.keys.chord(Action::ToggleGrade) == Chord{.key = Key::H, .ctrl = true});
  CHECK(merged.keys.chord(Action::ToggleOsd) == Chord{.key = Key::F1});
}

TEST_CASE("Superposition : des règles dans user.toml remplacent toutes celles de la base", "[user_overlay]") {
  const LayeredProfile merged = layered(R"(
[[profile]]
slot = 1
rule = []
)");
  REQUIRE(merged.profile.profiles[0]);
  CHECK(merged.profile.profiles[0]->rules.empty());
  CHECK(merged.profile.profiles[0]->name == "Qualité");
}

TEST_CASE("Superposition : mêmes plages et refus que profile.cpp", "[user_overlay]") {
  const auto refusal_of = [](const std::string& user) {
    auto result = parse_layered(kBase, user);
    REQUIRE_FALSE(result);
    return result.error().code;
  };
  CHECK(refusal_of("start_profile = 10") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("start_profile = 4") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("version = 4") == RefusalCode::ProfileVersion);
  CHECK(refusal_of("[[profile]]\nslot = 1\nhdr = \"vivid\"") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[[profile]]\nslot = 1\n[[profile.rule]]\nsteps = [{ kind = \"upscale\", model = \"m\", "
                   "factor = 9 }]") == RefusalCode::FieldOutOfRange);
  CHECK(refusal_of("[[profile]]\nname = \"sans slot\"") == RefusalCode::FieldMissing);
  CHECK(refusal_of("[keys]\ntoggle_osd = \"Space\"") == RefusalCode::KeyConflict);
  CHECK(refusal_of("start_profile = ") == RefusalCode::ProfileUnreadable);
}

TEST_CASE("Modifications : seules les clés qui diffèrent de la base s'écrivent", "[user_overlay]") {
  const std::string renamed = edited("", {edit::ProfileName{.slot = 2, .name = "Le mien"}});
  CHECK(renamed.find("Le mien") != std::string::npos);
  CHECK(renamed.find("rife") == std::string::npos);     // les règles de la base ne sont pas recopiées
  CHECK(renamed.find("Qualité") == std::string::npos);  // ni les autres profils
  CHECK(layered(renamed).profile.profiles[1]->name == "Le mien");

  // Revenir à la valeur de la base retire la clé, puis le profil devenu vide.
  const std::string reverted = edited(renamed, {edit::ProfileName{.slot = 2, .name = "Rapide"}});
  CHECK(reverted.find("profile") == std::string::npos);
  CHECK(layered(reverted) == layered(""));

  const std::string started = edited("", {edit::StartProfile{.slot = 2}});
  CHECK(layered(started).profile.start_profile == 2);
  CHECK(edited(started, {edit::StartProfile{.slot = 1}}).find("start_profile") == std::string::npos);

  const std::string hdr = edited("", {edit::ProfileHdr{.slot = 1, .mode = HdrMode::Off}});
  CHECK(layered(hdr).profile.profiles[0]->hdr == HdrMode::Off);
  CHECK(hdr.find("hdr = \"off\"") != std::string::npos);
}

TEST_CASE("Modifications : règles, raccourcis et remise à zéro d'un profil", "[user_overlay]") {
  const std::string rules = edited("", {edit::ProfileRules{.slot = 2, .rules = interpolation_x3()}});
  const LayeredProfile with_rules = layered(rules);
  CHECK(with_rules.profile.profiles[1]->rules == interpolation_x3());
  CHECK(with_rules.profile.profiles[1]->name == "Rapide");

  const std::string keys =
      edited("", {edit::Shortcut{.action = Action::TogglePause, .chord = std::nullopt},
                  edit::Shortcut{.action = Action::ToggleOsd, .chord = Chord{.key = Key::Space}}});
  const LayeredProfile with_keys = layered(keys);
  CHECK_FALSE(with_keys.keys.chord(Action::TogglePause));
  CHECK(with_keys.keys.action_for(Chord{.key = Key::Space}) == Action::ToggleOsd);
  CHECK(edited(keys, {edit::Shortcut{.action = Action::TogglePause, .chord = Chord{.key = Key::Space}},
                      edit::Shortcut{.action = Action::ToggleOsd, .chord = Chord{.key = Key::F1}}})
            .find("keys") == std::string::npos);

  const std::string several =
      edited("", {edit::ProfileName{.slot = 1, .name = "A"}, edit::ProfileName{.slot = 2, .name = "B"}});
  const std::string reset = edited(several, {edit::ResetProfile{.slot = 1}});
  CHECK(layered(reset).profile.profiles[0]->name == "Qualité");
  CHECK(layered(reset).profile.profiles[1]->name == "B");
}

TEST_CASE("Modifications : un résultat refusé n'est pas écrit", "[user_overlay]") {
  CHECK(edit_refusal("", {edit::Shortcut{.action = Action::ToggleOsd, .chord = Chord{.key = Key::Space}}}) ==
        RefusalCode::KeyConflict);
  CHECK(edit_refusal("", {edit::StartProfile{.slot = 7}}) == RefusalCode::FieldOutOfRange);
  CHECK(edit_refusal("", {edit::StartProfile{.slot = 0}}) == RefusalCode::FieldOutOfRange);
  CHECK(edit_refusal("", {edit::ProfileName{.slot = 10, .name = "x"}}) == RefusalCode::FieldOutOfRange);
  CHECK(edit_refusal("[[profile]\n", {}) == RefusalCode::ProfileUnreadable);
}

TEST_CASE("Modifications : les clés inconnues de user.toml restent", "[user_overlay]") {
  const std::string user = "future = 1\n[ui]\ntheme = \"sombre\"\n[[profile]]\nslot = 2\nnote = \"perso\"\n";
  const std::string text = edited(user, {edit::ProfileName{.slot = 2, .name = "Le mien"}});
  CHECK(text.find("future = 1") != std::string::npos);
  CHECK(text.find("theme = \"sombre\"") != std::string::npos);
  CHECK(text.find("note = \"perso\"") != std::string::npos);
  CHECK(text.find("Le mien") != std::string::npos);
}

TEST_CASE("user.toml : écriture atomique, base jamais réécrite", "[user_overlay]") {
  const fs::path dir = scratch("overlay-save");
  const fs::path base = dir / "profile.toml";
  const fs::path user = dir / "user.toml";
  write(base, kBase);

  auto saved = save_edits(base, user, std::vector<UserEdit>{edit::ProfileName{.slot = 1, .name = "Mien"}});
  REQUIRE(saved);
  CHECK(saved->profile.profiles[0]->name == "Mien");
  CHECK(read(base) == kBase);
  auto loaded = load_layered(base, user);
  REQUIRE(loaded);
  CHECK(*loaded == *saved);
  CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 2);

  const std::string before = read(user);
  auto refused = save_edits(base, user, std::vector<UserEdit>{edit::StartProfile{.slot = 9}});
  REQUIRE_FALSE(refused);
  CHECK(read(user) == before);
  CHECK(read(base) == kBase);

  auto without_user = load_layered(base, dir / "absent.toml");
  REQUIRE(without_user);
  CHECK(*without_user == layered(""));
  auto without_base = load_layered(dir / "absent.toml", user);
  REQUIRE_FALSE(without_base);
  CHECK(without_base.error().code == RefusalCode::ProfileUnreadable);
}

TEST_CASE("Fichiers texte : lecture entière et remplacement atomique", "[user_overlay]") {
  const fs::path dir = scratch("text-files");
  const fs::path path = dir / "neuf" / "fichier.toml";
  REQUIRE(write_text_atomically(path, "un"));
  REQUIRE(write_text_atomically(path, "deux\n"));
  auto text = read_text_file(path);
  REQUIRE(text);
  CHECK(*text == "deux\n");
  CHECK(std::distance(fs::directory_iterator(path.parent_path()), fs::directory_iterator{}) == 1);

  auto missing = read_text_file(dir / "absent");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::FileUnreadable);
  // Un dossier à la place du fichier : le renommage échoue, rien n'est laissé derrière.
  fs::create_directories(dir / "dossier");
  auto blocked = write_text_atomically(dir / "dossier", "x");
  REQUIRE_FALSE(blocked);
  CHECK(blocked.error().code == RefusalCode::FileUnwritable);
  CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 2);
}
