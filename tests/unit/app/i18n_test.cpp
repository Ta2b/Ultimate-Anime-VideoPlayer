#include "app/play/i18n.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/visible_error.h"
#include "french_ui.h"

namespace ajp::app::play {
namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<std::filesystem::path> interface_sources() {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(AJP_APP_SOURCE_DIR)) {
    const auto extension = entry.path().extension();
    if (entry.is_regular_file() && (extension == ".cpp" || extension == ".h")) files.push_back(entry.path());
  }
  std::ranges::sort(files);
  return files;
}

// Le texte d'un littéral C++ tel qu'il s'affiche (échappements simples : \" \\ \n).
std::string unescape(std::string_view text) {
  std::string out;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\\' && i + 1 < text.size()) {
      ++i;
      out += text[i] == 'n' ? '\n' : text[i];
    } else {
      out += text[i];
    }
  }
  return out;
}

int placeholders(std::string_view text) {
  return static_cast<int>(std::ranges::count(text, '{') - 2 * (text.find("{{") != std::string_view::npos));
}

// Les trous d'un format avec leur spécification, dans l'ordre : « {:.0f} » contre « {} » fait lever
// std::vformat à l'exécution (B43-10).
std::vector<std::string> format_specs(std::string_view text) {
  std::vector<std::string> specs;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '{' && i + 1 < text.size() && text[i + 1] == '{') {
      ++i;
    } else if (text[i] == '{') {
      const std::size_t close = text.find('}', i);
      if (close == std::string_view::npos) break;
      specs.emplace_back(text.substr(i, close - i + 1));
      i = close;
    }
  }
  return specs;
}

}  // namespace

TEST_CASE("i18n: la table française n'a ni clé en double ni texte vide", "[i18n]") {
  std::set<std::string_view> seen;
  for (const i18n::Entry& entry : i18n::french_entries()) {
    CHECK_FALSE(entry.english.empty());
    CHECK_FALSE(entry.french.empty());
    CHECK(seen.insert(entry.english).second);
    CHECK(placeholders(entry.english) == placeholders(entry.french));
    CHECK(format_specs(entry.english) == format_specs(entry.french));
  }
}

TEST_CASE("i18n: l'anglais est la langue par défaut et le français vient de la table", "[i18n]") {
  CHECK(i18n::language() == i18n::Language::English);
  CHECK(std::string_view(i18n::tr("Subtitles (j)")) == "Subtitles (j)");
  i18n::set_language(i18n::Language::French);
  CHECK(std::string_view(i18n::tr("Subtitles (j)")) == "Sous-titres (j)");
  CHECK(i18n::tr_format("Chapter {}/{}", 2, 5) == "Chapitre 2/5");
  CHECK(std::string_view(i18n::tr("not in the table")) == "not in the table");
  i18n::set_language(i18n::Language::English);
  CHECK(i18n::tr_format("Chapter {}/{}", 2, 5) == "Chapter 2/5");
  CHECK(i18n::language_from_code("fr") == i18n::Language::French);
  CHECK(i18n::language_from_code("xx") == i18n::Language::English);
}

TEST_CASE("i18n: chaque texte passé à tr() ou tr_format() a son français", "[i18n]") {
  std::set<std::string_view> known;
  for (const i18n::Entry& entry : i18n::french_entries()) known.insert(entry.english);
  // Les formes ui_tr(("…")) (double parenthèse) et tr_view("…") comptent aussi (B43-08).
  const std::regex call(R"re((?:tr(?:_format)?|tr_view)\(\s*\(?\s*((?:"(?:\\.|[^"\\])*"\s*)+))re");
  const std::regex piece(R"re("((?:\\.|[^"\\])*)")re");
  std::set<std::string> missing;
  for (const auto& file : interface_sources()) {
    const std::string text = read_file(file);
    for (std::sregex_iterator it(text.begin(), text.end(), call), end; it != end; ++it) {
      const std::string joined = (*it)[1].str();
      std::string key;
      for (std::sregex_iterator part(joined.begin(), joined.end(), piece), last; part != last; ++part)
        key += unescape((*part)[1].str());
      if (!known.contains(key)) missing.insert(file.filename().string() + " : " + key);
    }
  }
  for (const std::string& line : missing) UNSCOPED_INFO(line);
  CHECK(missing.empty());
}

// L'inverse (A43-01, A43-02) : chaque entrée de la table a son texte anglais quelque part dans le code de
// l'interface, hors des tables elles-mêmes. Une entrée morte garde un texte que personne ne voit, et qu'un
// retour de code ressusciterait faux.
TEST_CASE("i18n: chaque entrée de la table sert encore", "[i18n]") {
  const std::regex run(R"re(((?:"(?:\\.|[^"\\])*"\s*)+))re");
  const std::regex piece(R"re("((?:\\.|[^"\\])*)")re");
  std::set<std::string> literals;
  for (const auto& file : interface_sources()) {
    if (file.filename().string().starts_with("i18n_fr")) continue;
    const std::string text = read_file(file);
    for (std::sregex_iterator it(text.begin(), text.end(), run), end; it != end; ++it) {
      const std::string joined = (*it)[1].str();
      std::string key;
      for (std::sregex_iterator part(joined.begin(), joined.end(), piece), last; part != last; ++part) {
        literals.insert(unescape((*part)[1].str()));
        key += unescape((*part)[1].str());
      }
      literals.insert(key);
    }
  }
  std::vector<std::string_view> unused;
  for (const i18n::Entry& entry : i18n::french_entries()) {
    if (!literals.contains(std::string(entry.english))) unused.push_back(entry.english);
  }
  for (const std::string_view key : unused) UNSCOPED_INFO(key);
  CHECK(unused.empty());
}

// Les fichiers dont les textes visibles sont tous passés par tr() : aucune phrase française en dur ne s'y
// glisse (un littéral avec une lettre accentuée ou « « », hors journaux, commentaires et lignes « i18n-ok »).
TEST_CASE("i18n: aucune chaîne française en dur dans les fichiers traduits", "[i18n]") {
  // Les fichiers dont les littéraux français restent : textes de développeur (options de ligne de commande,
  // erreurs internes de chargement, fichier .desktop, journaux), jamais affichés tels quels dans l'interface
  // (voir visible_error.h).
  const std::set<std::string> developer{"settings_toml.cpp",
                                        "settings_format.cpp",
                                        "settings.cpp",
                                        "auto_bench.cpp",
                                        "backend_library.cpp",
                                        "chain.cpp",
                                        "chain_hdr.cpp",
                                        "chain_shots.cpp",
                                        "chain_setup.cpp",
                                        "chapters.cpp",
                                        "drba_fused_choice.cpp",
                                        "rife_fused_choice.cpp",
                                        "engine_install.cpp",
                                        "engine_vendor.cpp",
                                        "engine_versions.cpp",
                                        "engine_online.cpp",
                                        "engine_tools.cpp",
                                        "engine_build.h",
                                        "export_run.cpp",
                                        "export_sink.cpp",
                                        "gpu_tier.cpp",
                                        "home_screen.cpp",
                                        "latest_writer.cpp",
                                        "main.cpp",
                                        "problem_report.cpp",
                                        "model_choice.cpp",
                                        "motion_model.cpp",
                                        "placebo_export.cpp",
                                        "placebo_export_vulkan.cpp",
                                        "placebo_imgui.cpp",
                                        "placebo_feedback.cpp",
                                        "placebo_wayland.cpp",
                                        "ring_watcher.cpp",
                                        "display_contract.cpp",
                                        "display_follow.cpp",
                                        "stage_fallback.cpp",
                                        "sleep_notice.cpp",
                                        "after_gap.cpp",
                                        "player_machine.cpp",
                                        "gpu_profile_out.cpp",
                                        "placebo_open.cpp",
                                        "placebo_ring.cpp",
                                        "placebo_grade.cpp",
                                        "placebo_screen.cpp",
                                        "placebo_snapshot.cpp",
                                        "player_engines.cpp",
                                        "player_run.cpp",
                                        "ring_stage.cpp",
                                        "scripted_input.cpp",
                                        "session.cpp",
                                        "span_commit.cpp",
                                        "span_derive.cpp",
                                        "span_derive.h",
                                        "ui.cpp",
                                        "upscale_drba.cpp",
                                        "user_keys.cpp",
                                        "upscale_stage.cpp",
                                        "vulkan_share.cpp",
                                        "default_player.cpp",
                                        "player.cpp",
                                        "maxine_install.cpp",
                                        "relaunch_target.cpp",
                                        "sleep_watch.cpp",
                                        "system_peak.cpp",
                                        "window_vrr.cpp"};
  const std::string accents = "(?:à|â|ç|é|è|ê|ë|î|ï|ô|û|ù|ü|œ|À|É|È|Ê|«|»)";
  const std::string body = "(?:\\\\.|[^\"\\\\])*";
  const std::regex literal("\"" + body + accents + body + "\"");
  std::vector<std::string> found;
  for (const auto& file : interface_sources()) {
    const std::string name = file.filename().string();
    if (developer.contains(name) || name.starts_with("i18n_fr"))
      continue;  // les tables françaises sont le français
    std::istringstream lines(read_file(file));
    std::string line;
    int number = 0;
    while (std::getline(lines, line)) {
      ++number;
      if (line.contains("i18n-ok") || line.contains("log->")) continue;
      const auto comment = line.find("//");
      const std::string code = comment == std::string::npos ? line : line.substr(0, comment);
      if (!std::regex_search(code, literal)) continue;
      found.push_back(file.filename().string() + ":" + std::to_string(number));
    }
  }
  for (const std::string& where : found) UNSCOPED_INFO(where);
  CHECK(found.empty());
}

// Un texte traduit n'est jamais le format d'un appel ImGui à la printf (gcc -Wformat-security : « chaîne non
// littérale ») : « ImGui::TextWrapped("%s", i18n::tr(...)) », jamais « ImGui::TextWrapped(i18n::tr(...)) ».
TEST_CASE("i18n: aucun texte ImGui à la printf ne reçoit une chaîne non littérale", "[i18n]") {
  const std::regex call(
      R"re(ImGui::(?:TextWrapped|Text|TextDisabled|BulletText|SetTooltip|SetItemTooltip)\(\s*([^"\s]))re");
  std::vector<std::string> found;
  for (const auto& file : interface_sources()) {
    const std::string text = read_file(file);
    for (std::sregex_iterator it(text.begin(), text.end(), call), end; it != end; ++it) {
      const auto line = std::count(text.begin(), text.begin() + it->position(), '\n') + 1;
      found.push_back(file.filename().string() + ":" + std::to_string(line));
    }
  }
  for (const std::string& where : found) UNSCOPED_INFO(where);
  CHECK(found.empty());
}

}  // namespace ajp::app::play

TEST_CASE("Dolby Vision 5 et audio seul : dits en clair à l'ouverture (P3.26, P3.28)", "[i18n]") {
  const ajp::app::play::FrenchUi french;
  ajp::Refusal dolby{ajp::RefusalCode::FormatUnsupported, "x"};
  dolby.key = std::string(ajp::app::play::kDolbyVision5Key);
  CHECK(ajp::app::play::visible_error(dolby).contains("Dolby Vision profil 5"));
  ajp::Refusal audio{ajp::RefusalCode::TrackUnavailable, "x"};
  audio.key = std::string(ajp::app::play::kAudioOnlyKey);
  CHECK(ajp::app::play::visible_error(audio).contains("que du son"));
}

TEST_CASE("Les familles de refus ont chacune une phrase traduite (A20-09, A24-09)", "[i18n]") {
  const ajp::app::play::FrenchUi french;
  using ajp::RefusalCode;
  for (const RefusalCode code :
       {RefusalCode::HashMismatch, RefusalCode::ModelUnavailable, RefusalCode::SeekFailed,
        RefusalCode::LibassFailed, RefusalCode::CacheBusy, RefusalCode::BuilderTimeout,
        RefusalCode::LibraryUnavailable, RefusalCode::VendorCallFailed, RefusalCode::FieldOutOfRange,
        RefusalCode::EncodeFailed}) {
    const std::string text = ajp::app::play::visible_error(ajp::Refusal{code, "x"});
    INFO(text);
    CHECK_FALSE(text.contains("error "));
    CHECK_FALSE(text.contains("erreur "));
  }
  CHECK(ajp::app::play::visible_error(ajp::Refusal{RefusalCode::HdrUnavailable, "x"}).contains("HDR"));
}

TEST_CASE("Moteur GPU refusé : le pilote trop ancien et le délai dépassé ont leur conseil (B24-13)",
          "[i18n]") {
  const ajp::app::play::FrenchUi french;
  using ajp::RefusalCode;
  CHECK(ajp::app::play::visible_error(ajp::Refusal{RefusalCode::SdkRejected, "x"}).contains("pilote"));
  CHECK(ajp::app::play::visible_error(ajp::Refusal{RefusalCode::BuilderTimeout, "x"}).contains("réessayez"));
  CHECK_FALSE(
      ajp::app::play::visible_error(ajp::Refusal{RefusalCode::BuilderFailed, "x"}).contains("pilote"));
}

TEST_CASE("Erreurs visibles : chaque code de refus a une phrase, pas son seul nom anglais (A24-09)",
          "[i18n]") {
  // Avant : 56 codes sur 71 donnaient « erreur NomDuCode (détails au journal) » ; la table des familles en
  // laissait encore six (SourceResized, SourceUnknown, LookaheadMissing, SequenceOrder, InputMismatch,
  // PacketBacklog).
  int by_name = 0;
  for (int value = 0; value <= static_cast<int>(ajp::RefusalCode::EncodeFailed); ++value) {
    const auto code = static_cast<ajp::RefusalCode>(value);
    for (const bool french : {false, true}) {
      std::optional<ajp::app::play::FrenchUi> ui;  // construit sur place : FrenchUi ne se copie pas
      if (french) ui.emplace();
      const std::string shown = ajp::app::play::visible_error(ajp::Refusal{code, "texte-brut"});
      INFO(name(code) << " : " << shown);
      if (shown.starts_with("error ") || shown.starts_with("erreur ")) ++by_name;
      CHECK_FALSE(shown.contains("texte-brut"));  // le détail technique reste au journal
    }
  }
  CHECK(by_name == 0);
}
