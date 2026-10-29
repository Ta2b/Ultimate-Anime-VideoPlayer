#include "app/play/i18n.h"

#include <atomic>
#include <string>
#include <unordered_map>

namespace ajp::app::play::i18n {
namespace {

std::atomic<Language>& current_language() {
  static std::atomic<Language> current{Language::English};
  return current;
}

const std::unordered_map<std::string_view, std::string_view>& french_table() {
  static const std::unordered_map<std::string_view, std::string_view> table = [] {
    std::unordered_map<std::string_view, std::string_view> map;
    for (const Entry& entry : french_entries()) map.emplace(entry.english, entry.french);
    return map;
  }();
  return table;
}

}  // namespace

Language language() {
  return current_language().load(std::memory_order_relaxed);
}

void set_language(Language language) {
  current_language().store(language, std::memory_order_relaxed);
}

Language language_from_code(std::string_view code) {
  return code == "fr" ? Language::French : Language::English;
}

std::string_view language_code(Language language) {
  return language == Language::French ? "fr" : "en";
}

const char* tr(const char* key) {
  if (language() == Language::English) return key;
  const auto& table = french_table();
  const auto found = table.find(key);
  // Les textes français sont des littéraux : terminés par un zéro.
  return found != table.end() ? found->second.data()
                              : key;  // NOLINT(bugprone-suspicious-stringview-data-usage)
}

}  // namespace ajp::app::play::i18n

namespace ajp::app::play::i18n {

std::string_view tr_view(std::string_view key) {
  if (language() == Language::English) return key;
  const auto& table = french_table();
  const auto found = table.find(key);
  return found != table.end() ? found->second : key;
}

}  // namespace ajp::app::play::i18n
