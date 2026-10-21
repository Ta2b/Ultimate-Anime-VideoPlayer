// Interne à l'importeur INI : clés lues d'une section, partagées par ini_import.cpp
// (fichier, profils, bornes) et ini_import_steps.cpp (étapes). Ne s'inclut nulle part ailleurs.
#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/ini_import.h"
#include "core/profile.h"
#include "core/types.h"

namespace ajp::core::ini {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail);
std::string lower(std::string_view text);
std::string_view trim(std::string_view text);
// Chiffres seuls, sans signe ni espace.
std::optional<std::int64_t> parse_integer(std::string_view text);
// Nombre fini écrit en entier.
std::optional<double> parse_real(std::string_view text);

struct IniValue {
  std::string text;
  int line = 0;
  bool used = false;  // lue par l'importeur ; les autres deviennent des notes
};
using IniSection = std::map<std::string, IniValue, std::less<>>;

// Clés d'une section sous un préfixe (« chain_2_ », « chain_2_stage_1_ ») ; lire une clé la marque prise.
struct Keys {
  IniSection* section;
  std::string prefix;

  Keys sub(std::string_view more) const { return Keys{section, prefix + std::string(more)}; }
  std::string name(std::string_view rest) const { return prefix + std::string(rest); }

  IniValue* find(std::string_view rest) const;
  // Ligne de la clé, 0 si elle est absente.
  int line_of(std::string_view rest) const;
  std::string text(std::string_view rest, std::string_view fallback = "") const;
  // Refus qui cite la ligne, la clé et sa valeur.
  std::unexpected<Refusal> bad(RefusalCode code, std::string_view rest, std::string_view why) const;
  std::expected<bool, Refusal> flag(std::string_view rest, bool fallback) const;
  std::expected<std::int64_t, Refusal> integer(std::string_view rest, std::int64_t fallback) const;
  std::expected<double, Refusal> real(std::string_view rest, double fallback) const;
  bool has_prefix(std::string_view more) const;
  // Marque prises, sans les lire, toutes les clés sous more.
  void consume(std::string_view more) const;
  // Plus petite ligne des clés sous le préfixe, 0 s'il n'y en a pas.
  int first_line() const;
};

struct Importer {
  std::vector<ImportNote> notes;
  Policy policy = Policy::Fast;  // [global] rife_engine
  bool drawings = true;          // [global] rife_drawings
  std::string backend = "tensorrt";

  void note(int line, std::string text) { notes.push_back({line, std::move(text)}); }
};

// Chaîne historique : model_M_*, rife_*.
std::expected<std::vector<Step>, Refusal> legacy_steps(const Keys& chain, Importer& im);
// Chaîne ordonnée : pipeline_version = 1, stage_J_* ; les étapes HDR s'ajoutent à hdr.
std::expected<std::vector<Step>, Refusal> stage_steps(const Keys& chain, Importer& im,
                                                      std::vector<HdrMode>& hdr);

}  // namespace ajp::core::ini
