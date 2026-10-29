#pragma once

#include <format>
#include <span>
#include <string>
#include <string_view>

namespace ajp::app::play::i18n {

// Langue de l'interface : l'anglais par défaut, sans détection ; le français par le réglage « Language ».
enum class Language { English, French };

Language language();
void set_language(Language language);

// Code de réglage (« en », « fr ») : un code inconnu donne l'anglais.
Language language_from_code(std::string_view code);
std::string_view language_code(Language language);

// Le texte d'interface. La clé est le texte anglais (c'est lui qui s'affiche en anglais) ; le français vient
// de la table i18n_fr_*.cpp. Une clé absente de la table s'affiche telle quelle ; un test refuse toute clé
// utilisée dans src/app sans entrée française.
const char* tr(const char* key);

// Un texte à trous : la clé est le format anglais, le français le sien, mêmes {} dans le même ordre.
template <typename... Args>
std::string tr_format(const char* key, Args&&... args) {
  return std::vformat(tr(key), std::make_format_args(args...));
}

// Les tables françaises : (clé anglaise, texte français), toutes réunies par la recherche.
struct Entry {
  std::string_view english;
  std::string_view french;
};
std::span<const Entry> french_entries();

}  // namespace ajp::app::play::i18n
