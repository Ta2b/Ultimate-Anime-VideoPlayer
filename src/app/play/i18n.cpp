#include "app/play/i18n.h"

#include <atomic>
#include <string>
#include <unordered_map>

namespace ajp::app::play::i18n {
namespace {

std::atomic<Language> g_language{Language::English};

const std::unordered_map<std::string_view, const char*>& french_table() {
  static const std::unordered_map<std::string_view, const char*> table = [] {
    std::unordered_map<std::string_view, const char*> map;
    for (const Entry& entry : french_entries()) map.emplace(entry.english, entry.french.data());
    return map;
  }();
  return table;
}

}  // namespace

Language language() {
  return g_language.load(std::memory_order_relaxed);
}

void set_language(Language language) {
  g_language.store(language, std::memory_order_relaxed);
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
  return found != table.end() ? found->second : key;
}

}  // namespace ajp::app::play::i18n
