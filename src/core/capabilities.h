// États des capacités : chaque modèle et chaque route passe par quatre états.
//
//   compilée   le code qui l'exécute est dans ce binaire ou dans un backend construit ;
//   livrée     un paquet haché l'apporte (models.lock, index des moteurs) ;
//   installée  ses fichiers sont sur le disque, taille et SHA-256 vérifiés ;
//   qualifiée  un Receipt, produit par prepare après goldens et banc, correspond
//              à l'empreinte actuelle (pilote, fichier, Plan).
//
// Le reçu ne conditionne jamais le chargement : une capacité installée se
// charge, qualifiée ou non ; l'OSD signale seulement qu'elle ne l'est pas.
// Ces lignes alimentent le panneau des capacités et core::Capabilities de resolve().
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/plan.h"
#include "core/resolve.h"
#include "core/types.h"

namespace ajp::core {

enum class CapabilityKind : std::uint8_t { Model, Route };
enum class CapabilityState : std::uint8_t { Unavailable, Compiled, Delivered, Installed, Qualified };

// Routes connues de resolve() ; d'autres routes (moteurs, formats du ring) se nomment librement.
inline constexpr std::string_view kRouteHdrNative = "hdr-native";
inline constexpr std::string_view kRouteTrueHdr = "truehdr";

struct CapabilityId {
  CapabilityKind kind = CapabilityKind::Model;
  std::string name;  // ModelId, ou nom de route
  friend bool operator==(const CapabilityId&, const CapabilityId&) = default;
};

// Ce qu'une qualification a mesuré ; un changement de pilote, de fichier ou de Plan périme le reçu.
struct Fingerprint {
  std::string driver;           // version du pilote du GPU de rendu
  std::string artifact_sha256;  // fichier du modèle, ou paquet du moteur
  std::uint64_t plan_hash = 0;  // Plan joué pendant le banc
  friend bool operator==(const Fingerprint&, const Fingerprint&) = default;
};

// Résultat du banc, porté par le reçu. Des entiers : le hachage se refait à l'identique.
struct BenchResult {
  std::uint64_t pair_us = 0;     // temps par paire, rendu compris, en microsecondes
  std::uint64_t host_waits = 0;  // attentes côté hôte hors appel constructeur
  friend bool operator==(const BenchResult&, const BenchResult&) = default;
};

struct Receipt {
  CapabilityId capability;
  Fingerprint fingerprint;
  BenchResult bench;
  std::string hash;  // SHA-256 de tout ce qui précède, affiché par le panneau
  friend bool operator==(const Receipt&, const Receipt&) = default;
};

// Reçu haché d'une qualification réussie (goldens passés).
Receipt make_receipt(CapabilityId capability, Fingerprint fingerprint, BenchResult bench);
// Le hachage du reçu correspond à son contenu.
bool receipt_intact(const Receipt& receipt);

// Faits rassemblés par app depuis le binaire, l'index, le disque et receipts.toml.
struct CapabilityFacts {
  CapabilityId id;
  bool compiled = false;
  bool delivered = false;  // faux pour un SDK importé : installé sans être livré
  bool installed = false;
  Fingerprint current;  // empreinte actuelle, comparée au reçu
  std::optional<Receipt> receipt;
};

// L'état le plus avancé atteint. Compilée est un préalable : sans le code qui
// l'exécute, une capacité reste indisponible, même livrée ou installée.
CapabilityState state_of(const CapabilityFacts& facts);

// Une ligne du panneau des capacités.
struct CapabilityRow {
  CapabilityId id;
  CapabilityState state = CapabilityState::Unavailable;
  bool usable = false;         // installée (ou qualifiée) : se charge, qualifiée ou non
  bool receipt_stale = false;  // un reçu existe mais ne correspond plus
  std::string receipt_hash;    // vide sauf si qualifiée
};

// Modèles d'abord, puis routes, chacun dans l'ordre des faits.
std::vector<CapabilityRow> panel_rows(std::span<const CapabilityFacts> facts);

// Ce que resolve() peut utiliser maintenant : les capacités installées.
Capabilities to_resolve_capabilities(std::span<const CapabilityFacts> facts);

// Capacités d'un Plan qui ne sont pas qualifiées : l'OSD les signale.
std::vector<CapabilityId> unqualified_in(const Plan& plan, std::span<const CapabilityFacts> facts);

inline constexpr int kReceiptsVersion = 1;

// receipts.toml : [[receipt]] kind, name, driver, artifact_sha256, plan_hash, pair_us, host_waits, hash.
// Un reçu dont le hachage ne correspond pas à son contenu est refusé (HashMismatch).
std::expected<std::vector<Receipt>, Refusal> parse_receipts(std::string_view toml_text);
std::string format_receipts(std::span<const Receipt> receipts);

}  // namespace ajp::core
