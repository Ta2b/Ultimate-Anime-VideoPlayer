// Écriture et lecture des champs TOML d'un reçu. Interne au module.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++/toml.hpp>

#include "core/types.h"

namespace ajp::prepare {

// Chaîne TOML de base : guillemets, barres obliques et contrôles échappés.
std::string toml_string(std::string_view text);

// Réel TOML qui se relit au même double : forme la plus courte, « .0 » si entier.
std::string toml_real(double value);

// Lit les champs d'une table ; garde la première erreur, rendue par error().
class FieldReader {
 public:
  FieldReader(const toml::table* table, std::string where);

  std::string text(std::string_view key);
  double real(std::string_view key);  // un entier TOML est accepté
  std::uint64_t count(std::string_view key);
  bool flag(std::string_view key);
  const std::optional<Refusal>& error() const { return error_; }

 private:
  const toml::node* field(std::string_view key);
  void fail(std::string_view key, std::string_view why);

  const toml::table* table_;
  std::string where_;
  std::optional<Refusal> error_;
};

// Tableau de tables facultatif (« [[engines]] ») : un enregistrement par table.
template <class Make>
auto read_tables(const toml::table& root, std::string_view key, Make make)
    -> std::expected<std::vector<decltype(make(std::declval<FieldReader&>()))>, Refusal> {
  std::vector<decltype(make(std::declval<FieldReader&>()))> records;
  const toml::node* node = root.get(key);
  if (node == nullptr) return records;
  const toml::array* array = node->as_array();
  if (array == nullptr)
    return std::unexpected(Refusal{RefusalCode::ReceiptInvalid, std::format("{} : tableau attendu", key)});
  for (std::size_t i = 0; i < array->size(); ++i) {
    FieldReader fields(array->get(i)->as_table(), std::format("{}[{}]", key, i));
    auto record = make(fields);
    if (fields.error()) return std::unexpected(*fields.error());
    records.push_back(std::move(record));
  }
  return records;
}

}  // namespace ajp::prepare
