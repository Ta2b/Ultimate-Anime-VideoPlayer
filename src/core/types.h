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
  StepOrder,         // ordre d'étapes sans ordonnancement connu
  PacingForbidden,   // Quality impose le rythme Source
  ModelUnavailable,  // modèle absent des capacités
  HdrUnavailable,    // route HDR impossible sur ce système ou cet écran

  CacheIo,          // lecture, écriture ou renommage impossible dans le cache des moteurs
  CacheBusy,        // verrou de clé tenu par un autre constructeur
  Cancelled,        // construction annulée par sa demande ou par l'arrêt du thread
  BuilderFailed,    // constructeur introuvable, sorti en erreur, ou moteur vide
  BuilderCrashed,   // constructeur tué par un signal ou un plantage
  BuilderTimeout,   // construction plus longue que son délai
  ModelUnreadable,  // ONNX absent, tronqué ou malformé
  ModelMismatch,    // hachage, précision ou forme différents de ModelSpec
  ReceiptInvalid,   // reçu de qualification illisible, incomplet ou au hachage faux
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
    case RefusalCode::CacheIo:
      return "CacheIo";
    case RefusalCode::CacheBusy:
      return "CacheBusy";
    case RefusalCode::Cancelled:
      return "Cancelled";
    case RefusalCode::BuilderFailed:
      return "BuilderFailed";
    case RefusalCode::BuilderCrashed:
      return "BuilderCrashed";
    case RefusalCode::BuilderTimeout:
      return "BuilderTimeout";
    case RefusalCode::ModelUnreadable:
      return "ModelUnreadable";
    case RefusalCode::ModelMismatch:
      return "ModelMismatch";
    case RefusalCode::ReceiptInvalid:
      return "ReceiptInvalid";
  }
  return "Unknown";
}

}  // namespace ajp
