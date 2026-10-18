// Vocabulaire partagé par tous les modules. Aucune dépendance hors bibliothèque standard.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ajp {

// Identifie une source d'images : la vidéo décodée, ou la sortie d'un niveau
// d'interpolation qui devient la source du niveau suivant.
struct SourceId {
  std::uint32_t value = 0;
  friend bool operator==(SourceId, SourceId) = default;
};

// Toute erreur qui traverse une frontière de module est un refus nommé,
// renvoyé par std::expected<T, Refusal> et affiché tel quel.
enum class RefusalCode : std::uint8_t {
  ProfileUnreadable,  // fichier absent ou TOML invalide
  ProfileVersion,     // version différente de 5
  FieldMissing,       // clé obligatoire absente
  FieldType,          // clé du mauvais type
  FieldOutOfRange,    // valeur hors de sa plage constexpr
  TooManySteps,       // plus de 16 étapes dans une règle
  FactorProductTooLarge,
  StepOrder,          // ordre d'étapes sans ordonnancement connu
  PacingForbidden,    // Quality impose le rythme Source
  ModelUnavailable,   // modèle absent des capacités
  HdrUnavailable,     // route HDR impossible sur ce système ou cet écran
  SourceUnknown,      // source jamais reçue, ou sortie de la fenêtre gardée
  LookaheadMissing,   // les sources suivantes nécessaires ne sont pas encore classées
  SequenceOrder,      // hors séquence : source avant le début, après la fin ou reprise, époque qui recule
  FileUnreadable,     // fichier hors profil absent, illisible ou TOML invalide
  FileUnwritable,     // écriture ou renommage impossible
  FormatVersion,      // version inconnue de settings.toml, d'un index ou d'un INI
  ImportUnsupported,  // réglage INI sans équivalent en v5
  KeyUnknown,         // nom de touche ou d'action inconnu
  KeyConflict,        // deux actions sur le même raccourci
  OptionUnknown,      // option de ligne de commande inconnue
  HashMismatch,       // taille ou SHA-256 différents de l'index
};

struct Refusal {
  RefusalCode code;
  std::string detail;  // où et pourquoi, par exemple « profiles.3.rules[1].steps[2].factor »
};

constexpr std::string_view name(RefusalCode code) {
  switch (code) {
    case RefusalCode::ProfileUnreadable:
      return "ProfileUnreadable";
    case RefusalCode::ProfileVersion:
      return "ProfileVersion";
    case RefusalCode::FieldMissing:
      return "FieldMissing";
    case RefusalCode::FieldType:
      return "FieldType";
    case RefusalCode::FieldOutOfRange:
      return "FieldOutOfRange";
    case RefusalCode::TooManySteps:
      return "TooManySteps";
    case RefusalCode::FactorProductTooLarge:
      return "FactorProductTooLarge";
    case RefusalCode::StepOrder:
      return "StepOrder";
    case RefusalCode::PacingForbidden:
      return "PacingForbidden";
    case RefusalCode::ModelUnavailable:
      return "ModelUnavailable";
    case RefusalCode::HdrUnavailable:
      return "HdrUnavailable";
    case RefusalCode::SourceUnknown:
      return "SourceUnknown";
    case RefusalCode::LookaheadMissing:
      return "LookaheadMissing";
    case RefusalCode::SequenceOrder:
      return "SequenceOrder";
    case RefusalCode::FileUnreadable:
      return "FileUnreadable";
    case RefusalCode::FileUnwritable:
      return "FileUnwritable";
    case RefusalCode::FormatVersion:
      return "FormatVersion";
    case RefusalCode::ImportUnsupported:
      return "ImportUnsupported";
    case RefusalCode::KeyUnknown:
      return "KeyUnknown";
    case RefusalCode::KeyConflict:
      return "KeyConflict";
    case RefusalCode::OptionUnknown:
      return "OptionUnknown";
    case RefusalCode::HashMismatch:
      return "HashMismatch";
  }
  return "Unknown";
}

}  // namespace ajp
