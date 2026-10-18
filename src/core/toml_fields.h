// Lecture typée des champs TOML, pour les .cpp de core seulement : toml++ reste
// une dépendance privée, aucun en-tête public ne l'inclut. Chaque refus porte le
// chemin du champ, par exemple « model[2].sha256 ».
#pragma once

#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "core/profile.h"
#include "core/types.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++/toml.hpp>

namespace ajp::core::fields {

inline std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Texte TOML lu en table, ou refus `unreadable` qui nomme la ligne fautive.
inline std::expected<toml::table, Refusal> parse(std::string_view text, RefusalCode unreadable,
                                                 std::string_view what) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed) {
    return refuse(unreadable, std::format("{} ligne {} : {}", what, parsed.error().source().begin.line,
                                          parsed.error().description()));
  }
  return std::move(parsed).table();
}

template <class T>
std::expected<std::optional<T>, Refusal> optional(const toml::table& table, std::string_view key,
                                                  std::string_view where) {
  const toml::node* node = table.get(key);
  if (node == nullptr) return std::optional<T>{};
  if constexpr (std::is_same_v<T, double>) {
    // Un réel écrit « 30 » est un entier TOML : on l'accepte comme réel.
    if (auto i = node->value_exact<std::int64_t>()) return std::optional<T>{static_cast<double>(*i)};
  }
  std::optional<T> value = node->value_exact<T>();
  if (!value) return refuse(RefusalCode::FieldType, std::format("{}.{}", where, key));
  return value;
}

template <class T>
std::expected<T, Refusal> required(const toml::table& table, std::string_view key, std::string_view where) {
  auto value = optional<T>(table, key, where);
  if (!value) return std::unexpected(value.error());
  if (!*value) return refuse(RefusalCode::FieldMissing, std::format("{}.{}", where, key));
  return **value;
}

template <class T>
std::expected<T, Refusal> in_range(T value, Range<T> range, std::string_view where) {
  if (!range.contains(value)) return refuse(RefusalCode::FieldOutOfRange, std::string(where));
  return value;
}

// Tableau facultatif : absent = vide ; refus FieldType s'il n'est pas un tableau.
inline std::expected<const toml::array*, Refusal> array(const toml::table& table, std::string_view key,
                                                        std::string_view where) {
  static const toml::array kEmpty;
  const toml::node* node = table.get(key);
  if (node == nullptr) return &kEmpty;
  if (const toml::array* found = node->as_array()) return found;
  return refuse(RefusalCode::FieldType, std::format("{}.{}", where, key));
}

// Table facultative : absente = vide ; refus FieldType si ce n'en est pas une.
inline std::expected<const toml::table*, Refusal> table(const toml::table& parent, std::string_view key,
                                                        std::string_view where) {
  static const toml::table kEmpty;
  const toml::node* node = parent.get(key);
  if (node == nullptr) return &kEmpty;
  if (const toml::table* found = node->as_table()) return found;
  return refuse(RefusalCode::FieldType, std::format("{}.{}", where, key));
}

// Table enfant à modifier, créée si absente ; une valeur d'un autre type est remplacée.
inline toml::table& child(toml::table& parent, std::string_view key) {
  if (toml::table* found = parent[key].as_table()) return *found;
  return *parent.insert_or_assign(key, toml::table{}).first->second.as_table();
}

// Texte TOML d'une table, chaînes entre guillemets doubles.
inline std::string format(const toml::table& root) {
  constexpr toml::format_flags kFlags = toml::toml_formatter::default_flags &
                                        ~toml::format_flags::allow_literal_strings &
                                        ~toml::format_flags::allow_multi_line_strings;
  std::ostringstream out;
  out << toml::toml_formatter(root, kFlags) << '\n';
  return out.str();
}

// Version du fichier : obligatoire, égale à `expected`, sinon FormatVersion.
inline std::expected<void, Refusal> check_version(const toml::table& root, std::int64_t expected,
                                                  std::string_view what) {
  auto version = required<std::int64_t>(root, "version", what);
  if (!version) return std::unexpected(version.error());
  if (*version != expected)
    return refuse(RefusalCode::FormatVersion, std::format("{} version = {}", what, *version));
  return {};
}

}  // namespace ajp::core::fields
