#include "prepare/receipt_fields.h"

namespace ajp::prepare {

std::string toml_string(std::string_view text) {
  std::string out = "\"";
  for (const char c : text) {
    const auto byte = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (byte < 0x20 || byte == 0x7f) {
      out += std::format("\\u{:04X}", byte);
    } else {
      out += c;
    }
  }
  return out + "\"";
}

std::string toml_real(double value) {
  std::string text = std::format("{}", value);
  if (text.find_first_of(".en") == std::string::npos) text += ".0";  // « 22 » serait un entier TOML
  return text;
}

FieldReader::FieldReader(const toml::table* table, std::string where)
    : table_(table), where_(std::move(where)) {
  if (table_ == nullptr) error_ = Refusal{RefusalCode::ReceiptInvalid, "table [" + where_ + "] absente"};
}

const toml::node* FieldReader::field(std::string_view key) {
  if (error_) return nullptr;
  const toml::node* node = table_->get(key);
  if (node == nullptr) fail(key, "absent");
  return node;
}

void FieldReader::fail(std::string_view key, std::string_view why) {
  if (!error_) error_ = Refusal{RefusalCode::ReceiptInvalid, std::format("{}.{} : {}", where_, key, why)};
}

std::string FieldReader::text(std::string_view key) {
  const toml::node* node = field(key);
  if (node == nullptr) return {};
  auto value = node->value_exact<std::string>();
  if (!value) fail(key, "chaîne attendue");
  return value.value_or("");
}

double FieldReader::real(std::string_view key) {
  const toml::node* node = field(key);
  if (node == nullptr) return 0.0;
  if (auto whole = node->value_exact<std::int64_t>()) return static_cast<double>(*whole);
  auto value = node->value_exact<double>();
  if (!value) fail(key, "réel attendu");
  return value.value_or(0.0);
}

std::uint64_t FieldReader::count(std::string_view key) {
  const toml::node* node = field(key);
  if (node == nullptr) return 0;
  auto value = node->value_exact<std::int64_t>();
  if (!value || *value < 0) {
    fail(key, "entier positif ou nul attendu");
    return 0;
  }
  return static_cast<std::uint64_t>(*value);
}

bool FieldReader::flag(std::string_view key) {
  const toml::node* node = field(key);
  if (node == nullptr) return false;
  auto value = node->value_exact<bool>();
  if (!value) fail(key, "booléen attendu");
  return value.value_or(false);
}

}  // namespace ajp::prepare
