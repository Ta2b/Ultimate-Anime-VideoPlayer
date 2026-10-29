#include "app/play/i18n.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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

}  // namespace

TEST_CASE("i18n: la table française n'a ni clé en double ni texte vide", "[i18n]") {
  std::set<std::string_view> seen;
  for (const i18n::Entry& entry : i18n::french_entries()) {
    CHECK_FALSE(entry.english.empty());
    CHECK_FALSE(entry.french.empty());
    CHECK(seen.insert(entry.english).second);
    CHECK(placeholders(entry.english) == placeholders(entry.french));
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
  const std::regex call(R"re(tr(?:_format)?\(\s*((?:"(?:\\.|[^"\\])*"\s*)+))re");
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

// Les fichiers dont les textes visibles sont tous passés par tr() : aucune phrase française en dur ne s'y
// glisse (un littéral avec une lettre accentuée ou « « », hors journaux, commentaires et lignes « i18n-ok »).
TEST_CASE("i18n: aucune chaîne française en dur dans les fichiers traduits", "[i18n]") {
  const std::set<std::string> translated{"controls_draw.cpp", "tracks.cpp", "settings_panel.cpp"};
  const std::string accents = "(?:à|â|ç|é|è|ê|ë|î|ï|ô|û|ù|ü|œ|À|É|È|Ê|«|»)";
  const std::string body = "(?:\\\\.|[^\"\\\\])*";
  const std::regex literal("\"" + body + accents + body + "\"");
  std::vector<std::string> found;
  for (const auto& file : interface_sources()) {
    if (!translated.contains(file.filename().string())) continue;
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

}  // namespace ajp::app::play
