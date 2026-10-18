// Outils des tests de prepare : dossier temporaire, fichiers, ONNX synthétiques
// écrits au format protobuf, et chemin de l'exécutable d'aide.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::prepare::test {

namespace fs = std::filesystem;

// Dossier neuf, supprimé avec son contenu à la fin du test.
class TempDir {
 public:
  TempDir() {
    std::random_device random;
    path_ = fs::temp_directory_path() /
            ("ajp-prepare-" + std::to_string(random()) + "-" + std::to_string(random()));
    fs::create_directories(path_);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }
  const fs::path& path() const { return path_; }
  fs::path operator/(const fs::path& name) const { return path_ / name; }

 private:
  fs::path path_;
};

inline void write_file(const fs::path& file, std::string_view bytes) {
  fs::create_directories(file.parent_path());
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

inline std::string read_file(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

inline std::vector<std::string> read_lines(const fs::path& file) {
  std::ifstream in(file);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  return lines;
}

inline std::span<const std::byte> as_bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}

// Code du refus, ou nullopt si result porte une valeur : error() n'est jamais
// appelé sur une valeur.
template <class T>
std::optional<ajp::RefusalCode> refusal(const std::expected<T, ajp::Refusal>& result) {
  if (result) return std::nullopt;
  return result.error().code;
}

// Dossiers « .aji-build-* » restants sous parent.
inline int stage_count(const fs::path& parent) {
  int count = 0;
  for (const auto& entry : fs::directory_iterator(parent)) {
    if (entry.path().filename().string().starts_with(".aji-build-")) ++count;
  }
  return count;
}

// Exécutable d'aide (helper.cpp), chemin fixé par CMake.
inline fs::path helper_path() {
  return fs::path(AJP_PREPARE_HELPER);
}

// --- Écriture protobuf minimale, pour fabriquer des ONNX de test ---

inline std::string varint(std::uint64_t value) {
  std::string out;
  while (value >= 0x80) {
    out += static_cast<char>((value & 0x7fU) | 0x80U);
    value >>= 7U;
  }
  out += static_cast<char>(value);
  return out;
}

inline std::string field(std::uint32_t number, std::uint64_t value) {
  return varint(number << 3U) + varint(value);
}

inline std::string field(std::uint32_t number, std::string_view bytes) {
  return varint((number << 3U) | 2U) + varint(bytes.size()) + std::string(bytes);
}

struct FakeInput {
  std::string name = "input";
  int elem_type = 10;                            // FLOAT16
  std::vector<std::int64_t> dims{1, 3, -1, -1};  // -1 : dim_param
};

inline std::string value_info(const FakeInput& input) {
  std::string shape;
  for (const std::int64_t d : input.dims) {
    shape += field(1, d > 0 ? field(1, static_cast<std::uint64_t>(d)) : field(2, std::string_view("N")));
  }
  const std::string tensor = field(1, static_cast<std::uint64_t>(input.elem_type)) + field(2, shape);
  return field(1, input.name) + field(2, field(1, tensor));
}

// Initialiseur ; avec location, ses poids sont externes.
inline std::string initializer(std::string_view name, std::string_view location = {}) {
  std::string tensor = field(1, std::uint64_t{4}) + field(2, std::uint64_t{1}) + field(8, name);
  if (!location.empty()) {
    tensor += field(13, field(1, std::string_view("location")) + field(2, location));
    tensor += field(14, std::uint64_t{1});
  } else {
    tensor += field(9, std::string_view("\x01\x02\x03\x04"));  // raw_data jamais relu comme protobuf
  }
  return tensor;
}

struct FakeModel {
  std::vector<FakeInput> inputs{FakeInput{}};
  std::vector<std::string> ops{"Conv", "Relu"};
  std::vector<std::string> initializers;  // noms, aussi déclarés en entrée (ancien ONNX)
  std::vector<std::string> external;      // emplacements de poids externes
};

inline std::string onnx(const FakeModel& model) {
  std::string graph;
  for (const std::string& op : model.ops) graph += field(1, field(4, op));
  for (const std::string& name : model.initializers) graph += field(5, initializer(name));
  for (std::size_t i = 0; i < model.external.size(); ++i) {
    graph += field(5, initializer("w" + std::to_string(i), model.external[i]));
  }
  for (const FakeInput& input : model.inputs) graph += field(11, value_info(input));
  for (const std::string& name : model.initializers)
    graph += field(11, value_info({.name = name, .elem_type = 1, .dims = {4}}));
  return field(1, std::uint64_t{8}) + field(7, graph);
}

}  // namespace ajp::prepare::test
