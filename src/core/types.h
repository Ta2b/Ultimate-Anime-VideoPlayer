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
  StepOrder,              // ordre d'étapes sans ordonnancement connu
  PacingForbidden,        // Quality impose le rythme Source
  ModelUnavailable,       // modèle absent des capacités
  HdrUnavailable,         // route HDR impossible sur ce système ou cet écran
  SourceUnknown,          // source jamais reçue, ou sortie de la fenêtre gardée
  LookaheadMissing,       // les sources suivantes nécessaires ne sont pas encore classées
  SequenceOrder,          // hors séquence : source avant le début, après la fin ou reprise, époque qui recule
  FileUnreadable,         // fichier hors profil absent, illisible ou TOML invalide
  FileUnwritable,         // écriture ou renommage impossible
  FormatVersion,          // version inconnue de settings.toml, d'un index ou d'un INI
  ImportUnsupported,      // réglage INI sans équivalent en v5
  KeyUnknown,             // nom de touche ou d'action inconnu
  KeyConflict,            // deux actions sur le même raccourci
  OptionUnknown,          // option de ligne de commande inconnue
  HashMismatch,           // taille ou SHA-256 différents de l'index
  MediaUnreadable,        // fichier absent ou conteneur illisible
  TrackUnavailable,       // piste absente, ou d'un autre type que demandé
  HwDecodeUnavailable,    // une API de décodage matériel ne convient pas (codec, appareil, pilote)
  DecoderUnavailable,     // aucun essai de décodage n'a abouti
  DecodeFailed,           // erreur de lecture ou de décodage en cours de flux
  FormatUnsupported,      // format de pixel ou couleur hors du vocabulaire de core/frame.h, ou refusé par un
                          // noyau
  SeekFailed,             // cible hors du fichier ou seek refusé par le conteneur
  AudioCodecUnavailable,  // aucun décodeur FFmpeg pour ce flux audio
  AudioDecodeFailed,      // paquet ou conversion audio refusés par FFmpeg
  AudioDeviceUnavailable,  // périphérique ou file SDL3 impossibles à ouvrir
  SubtitleUnreadable,      // fichier de sous-titres absent, illisible ou trop gros
  SubtitleFormat,          // script ASS ou SRT mal formé, extension inconnue
  SubtitleAtlasFull,       // les images d'un instant dépassent l'atlas borné
  SubtitleBacklog,         // trop de demandes de rendu en attente d'un atlas
  LibassFailed,            // libass n'a pas créé sa bibliothèque, son moteur ou une piste
  ThreadUnavailable,       // le système refuse de lancer un thread de plus
  CacheIo,                 // lecture, écriture ou renommage impossible dans le cache des moteurs
  CacheBusy,               // verrou de clé tenu par un autre constructeur
  Cancelled,               // construction annulée par sa demande ou par l'arrêt du thread
  BuilderFailed,           // constructeur introuvable, sorti en erreur, ou moteur vide
  BuilderCrashed,          // constructeur tué par un signal ou un plantage
  BuilderTimeout,          // construction plus longue que son délai
  ModelUnreadable,         // ONNX absent, tronqué ou malformé
  ModelMismatch,           // hachage, précision ou forme différents de ModelSpec
  ReceiptInvalid,          // reçu de qualification illisible, incomplet ou au hachage faux
  // vendor_hdr : SDK TrueHDR, chargement dynamique, contrat d'entrée
  SdkRejected,         // SDK importé refusé : fichier absent ou vide, architecture, version, ABI
  LibraryUnavailable,  // bibliothèque ou symbole introuvable au chargement
  VendorCallFailed,    // un appel d'un SDK constructeur a renvoyé une erreur
  InputMismatch,       // image ou tampon hors du contrat d'entrée (taille, format, tramage)
  IoFailed,            // lecture, copie ou écriture de fichier impossible

  OnnxMalformed,     // octets qui ne forment pas un ModelProto lisible
  OnnxExternalData,  // poids hors du fichier ONNX, refusés
  OnnxUnsupported,   // type, opset ou construction ONNX hors du sous-ensemble lu
  OperatorUnknown,   // opérateur hors du jeu de l'IR ; le détail porte son nom
  GraphInvalid,      // tenseur absent, cycle, attribut ou entrée incohérents
  ShapeInvalid,      // formes incompatibles, dimension impossible à déduire, ou taille refusée par un noyau

  // core : threads et backends chargés à l'exécution
  ThreadFailed,    // une exception a interrompu le corps d'un thread, qui s'est arrêté
  AbiMismatch,     // bibliothèque d'une autre version de l'interface (kBackendAbi)
  DeviceMismatch,  // le backend ne sait pas exploiter l'appareil de rendu (API, constructeur, UUID)

  // present : fenêtre et swapchain
  PresentFailed,  // SDL3, libplacebo ou le pilote refusent la fenêtre, la surface ou la swapchain
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
    case RefusalCode::MediaUnreadable:
      return "MediaUnreadable";
    case RefusalCode::TrackUnavailable:
      return "TrackUnavailable";
    case RefusalCode::HwDecodeUnavailable:
      return "HwDecodeUnavailable";
    case RefusalCode::DecoderUnavailable:
      return "DecoderUnavailable";
    case RefusalCode::DecodeFailed:
      return "DecodeFailed";
    case RefusalCode::FormatUnsupported:
      return "FormatUnsupported";
    case RefusalCode::SeekFailed:
      return "SeekFailed";
    case RefusalCode::AudioCodecUnavailable:
      return "AudioCodecUnavailable";
    case RefusalCode::AudioDecodeFailed:
      return "AudioDecodeFailed";
    case RefusalCode::AudioDeviceUnavailable:
      return "AudioDeviceUnavailable";
    case RefusalCode::SubtitleUnreadable:
      return "SubtitleUnreadable";
    case RefusalCode::SubtitleFormat:
      return "SubtitleFormat";
    case RefusalCode::SubtitleAtlasFull:
      return "SubtitleAtlasFull";
    case RefusalCode::SubtitleBacklog:
      return "SubtitleBacklog";
    case RefusalCode::LibassFailed:
      return "LibassFailed";
    case RefusalCode::ThreadUnavailable:
      return "ThreadUnavailable";
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
    case RefusalCode::SdkRejected:
      return "SdkRejected";
    case RefusalCode::LibraryUnavailable:
      return "LibraryUnavailable";
    case RefusalCode::VendorCallFailed:
      return "VendorCallFailed";
    case RefusalCode::InputMismatch:
      return "InputMismatch";
    case RefusalCode::IoFailed:
      return "IoFailed";
    case RefusalCode::OnnxMalformed:
      return "OnnxMalformed";
    case RefusalCode::OnnxExternalData:
      return "OnnxExternalData";
    case RefusalCode::OnnxUnsupported:
      return "OnnxUnsupported";
    case RefusalCode::OperatorUnknown:
      return "OperatorUnknown";
    case RefusalCode::GraphInvalid:
      return "GraphInvalid";
    case RefusalCode::ShapeInvalid:
      return "ShapeInvalid";
    case RefusalCode::ThreadFailed:
      return "ThreadFailed";
    case RefusalCode::AbiMismatch:
      return "AbiMismatch";
    case RefusalCode::DeviceMismatch:
      return "DeviceMismatch";
    case RefusalCode::PresentFailed:
      return "PresentFailed";
  }
  return "Unknown";
}

}  // namespace ajp
