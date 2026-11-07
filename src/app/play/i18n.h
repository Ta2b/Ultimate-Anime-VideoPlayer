#pragma once

#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace ajp::app::play::i18n {

// Langue de l'interface : l'anglais par défaut, sans détection ; le français par le réglage « Language ».
enum class Language { English, French };

Language language();
void set_language(Language language);
// Le séparateur décimal de la langue : « 2,5 » en français, « 2.5 » en anglais.
char decimal_point();

// Code de réglage (« en », « fr ») : un code inconnu donne l'anglais.
Language language_from_code(std::string_view code);

// Le texte d'interface. La clé est le texte anglais (c'est lui qui s'affiche en anglais) ; le français vient
// de la table i18n_fr_*.cpp. Une clé absente de la table s'affiche telle quelle ; un test refuse toute clé
// utilisée dans src/app sans entrée française.
const char* tr(const char* key);
// Le même, pour une clé qui est déjà un string_view de littéral (étiquettes de tables).
std::string_view tr_view(std::string_view key);

// Un texte à trous : la clé est le format anglais, le français le sien, mêmes {} dans le même ordre.
template <typename... Args>
std::string tr_format(const char* key, Args&&... args) {
  return std::vformat(tr(key), std::make_format_args(args...));
}

// Un texte montré à l'écran (message bref, écran de préparation) : il ne se construit que par ui_tr ou
// ui_tr_format, donc traduit, ou par Translated::data pour ce qui n'a pas de langue (nom de fichier, de
// piste, de chapitre, nombre). Un littéral, un détail de refus ou une phrase en dur ne passent pas.
class Translated {
 public:
  Translated() = default;
  // Ce qui n'a pas de langue : montré tel quel. Jamais une phrase.
  static Translated data(std::string text) { return Translated(std::move(text)); }
  const std::string& str() const { return text_; }
  bool empty() const { return text_.empty(); }
  Translated operator+(const Translated& next) const { return Translated(text_ + next.text_); }

 private:
  explicit Translated(std::string text) : text_(std::move(text)) {}
  friend Translated ui_tr(const char* key);
  template <typename... Args>
  friend Translated ui_tr_format(const char* key, Args&&... args);
  std::string text_;
};

// tr() et tr_format() pour un texte montré.
inline Translated ui_tr(const char* key) {
  return Translated(tr(key));
}
template <typename... Args>
Translated ui_tr_format(const char* key, Args&&... args) {
  return Translated(tr_format(key, args...));
}

// Les tables françaises : (clé anglaise, texte français), toutes réunies par la recherche.
struct Entry {
  std::string_view english;
  std::string_view french;
};
std::span<const Entry> french_entries();

}  // namespace ajp::app::play::i18n
