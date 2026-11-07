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
// Les codes de refus, une ligne chacun : l'énumération et son nom (name) en viennent (P5.18).
#define AJP_REFUSAL_CODES(X)                                                                                 \
  X(ProfileUnreadable) /* fichier absent ou TOML invalide */                                                 \
  X(ProfileVersion)    /* version différente de 5 */                                                         \
  X(FieldMissing)      /* clé obligatoire absente */                                                         \
  X(FieldType)         /* clé du mauvais type */                                                             \
  X(FieldOutOfRange)   /* valeur hors de sa plage constexpr */                                               \
  X(TooManySteps)      /* plus de 16 étapes dans une règle */                                                \
  X(FactorProductTooLarge)                                                                                   \
  X(StepOrder)        /* ordre d'étapes sans ordonnancement connu */                                         \
  X(PacingForbidden)  /* Quality impose le rythme Source */                                                  \
  X(ModelUnavailable) /* modèle absent des capacités */                                                      \
  X(HdrUnavailable)   /* route HDR impossible sur ce système ou cet écran */                                 \
  X(SourceUnknown)    /* source jamais reçue, ou sortie de la fenêtre gardée */                              \
  X(LookaheadMissing) /* les sources suivantes nécessaires ne sont pas encore classées */                    \
  X(SequenceOrder)  /* hors séquence : source avant le début, après la fin ou reprise, époque qui recule */  \
  X(FileUnreadable) /* fichier hors profil absent, illisible ou TOML invalide */                             \
  X(FileUnwritable) /* écriture ou renommage impossible */                                                   \
  X(InsufficientSpace) /* pas assez de place libre pour installer un moteur (dit combien il faut et combien  \
                          il reste) */                                                                       \
  X(FormatVersion)     /* version inconnue de settings.toml, d'un index ou d'un INI */                       \
  X(ImportUnsupported) /* réglage INI sans équivalent en v5 */                                               \
  X(KeyUnknown)        /* nom de touche ou d'action inconnu */                                               \
  X(KeyConflict)       /* deux actions sur le même raccourci */                                              \
  X(OptionUnknown)     /* option de ligne de commande inconnue */                                            \
  X(HashMismatch)      /* taille ou SHA-256 différents de l'index */                                         \
  X(MediaUnreadable)   /* fichier absent ou conteneur illisible */                                           \
  X(TrackUnavailable)  /* piste absente, ou d'un autre type que demandé */                                   \
  X(HwDecodeUnavailable) /* une API de décodage matériel ne convient pas (codec, appareil, pilote) */        \
  X(DecoderUnavailable)  /* aucun essai de décodage n'a abouti */                                            \
  X(DecodeFailed)        /* erreur de lecture ou de décodage en cours de flux */                             \
  X(FormatUnsupported) /* format de pixel ou couleur hors du vocabulaire de core/frame.h, ou refusé par un  \
                          noyau */                                                                           \
  X(SeekFailed)        /* cible hors du fichier ou seek refusé par le conteneur */                           \
  X(AudioCodecUnavailable)  /* aucun décodeur FFmpeg pour ce flux audio */                                   \
  X(AudioDecodeFailed)      /* paquet ou conversion audio refusés par FFmpeg */                              \
  X(AudioDeviceUnavailable) /* périphérique ou file SDL3 impossibles à ouvrir */                             \
  X(SubtitleUnreadable)     /* fichier de sous-titres absent, illisible ou trop gros */                      \
  X(SubtitleFormat)         /* script ASS ou SRT mal formé, extension inconnue */                            \
  X(SubtitleAtlasFull)      /* les images d'un instant dépassent l'atlas borné */                            \
  X(SubtitleBacklog)        /* trop de demandes de rendu en attente d'un atlas */                            \
  X(LibassFailed)           /* libass n'a pas créé sa bibliothèque, son moteur ou une piste */               \
  X(ThreadUnavailable)      /* le système refuse de lancer un thread de plus */                              \
  X(CacheIo)                /* lecture, écriture ou renommage impossible dans le cache des moteurs */        \
  X(CacheBusy)              /* verrou de clé tenu par un autre constructeur */                               \
  X(Cancelled)              /* construction annulée par sa demande ou par l'arrêt du thread */               \
  X(BuilderFailed)          /* constructeur introuvable, sorti en erreur, ou moteur vide */                  \
  X(BuilderCrashed)         /* constructeur tué par un signal ou un plantage */                              \
  X(BuilderTimeout)         /* construction plus longue que son délai */                                     \
  X(ModelUnreadable)        /* ONNX absent, tronqué ou malformé */                                           \
  X(ModelMismatch)          /* hachage, précision ou forme différents de ModelSpec */                        \
  X(ReceiptInvalid)         /* reçu de qualification illisible, incomplet ou au hachage faux */              \
  /* Bibliothèques externes : chargement dynamique, appels, contrat d'entrée */                              \
  X(SdkRejected)        /* SDK importé refusé : fichier absent ou vide, architecture, version, ABI */        \
  X(LibraryUnavailable) /* bibliothèque ou symbole introuvable au chargement */                              \
  X(VendorCallFailed)   /* un appel d'un SDK constructeur a renvoyé une erreur */                            \
  X(InputMismatch)      /* image ou tampon hors du contrat d'entrée (taille, format, tramage) */             \
  X(IoFailed)           /* lecture, copie ou écriture de fichier impossible */                               \
  /* gpu/ir : lecture ONNX et IR de graphes */                                                               \
  X(OnnxMalformed)    /* octets qui ne forment pas un ModelProto lisible */                                  \
  X(OnnxExternalData) /* poids hors du fichier ONNX, refusés */                                              \
  X(OnnxUnsupported)  /* type, opset ou construction ONNX hors du sous-ensemble lu */                        \
  X(OperatorUnknown)  /* opérateur hors du jeu de l'IR ; le détail porte son nom */                          \
  X(GraphInvalid)     /* tenseur absent, cycle, attribut ou entrée incohérents */                            \
  X(ShapeInvalid) /* formes incompatibles, dimension impossible à déduire, ou taille refusée par un noyau */ \
  X(SourceResized) /* la source change de taille en cours de lecture : le détail porte « LxH » ; la chaîne \
                      se rouvre */                                                                           \
  /* core : threads et backends chargés à l'exécution */                                                     \
  X(ThreadFailed)   /* une exception a interrompu le corps d'un thread, qui s'est arrêté */                  \
  X(DeviceLost)     /* contexte CUDA ou appareil Vulkan perdu : relancer le lecteur */                       \
  X(AbiMismatch)    /* bibliothèque d'une autre version de l'interface (kBackendAbi) */                      \
  X(DeviceMismatch) /* le backend ne sait pas exploiter l'appareil de rendu (API, constructeur, UUID) */     \
  /* media : pistes annexes de la source */                                                                  \
  X(PacketBacklog) /* paquets annexes non retirés : réserve de la source pleine avant la première image */   \
  /* gpu/vulkan/compute : socle de calcul Vulkan */                                                          \
  X(VulkanCallFailed)   /* un appel Vulkan a renvoyé une erreur ; le détail cite l'appel et son VkResult */  \
  X(DeviceUnavailable)  /* aucun appareil, aucune file compute, ou UUID introuvable */                       \
  X(FeatureUnavailable) /* version, extension, fonction, format ou limite absents de l'appareil */           \
  X(GpuMemoryExhausted) /* mémoire de l'appareil, sous-allocateur ou pool de descripteurs à leur borne */    \
  /* export : encodage et mux */                                                                             \
  X(EncoderUnavailable) /* aucun encodeur du codec demandé ne s'ouvre (absent de FFmpeg, GPU, pilote) */     \
  X(EncodeFailed)       /* l'encodeur ou le mux a refusé une image ou un paquet */

enum class RefusalCode : std::uint8_t {
#define AJP_REFUSAL_ENUM(code) code,
  AJP_REFUSAL_CODES(AJP_REFUSAL_ENUM)
#undef AJP_REFUSAL_ENUM
};

struct Refusal {
  RefusalCode code;
  std::string detail;  // où et pourquoi, par exemple « profiles.3.rules[1].steps[2].factor »
  // Vrai quand l'étage qui refuse continue : la lecture n'a pas à s'arrêter. Faux par défaut, un
  // refus arrête l'étage qui le signale.
  bool recoverable = false;
  // La clé fautive d'un fichier de réglages, « audio.delay_ms » ; vide sinon. La reprise la lit, jamais le
  // texte de detail (P2.11).
  std::string key;
};

constexpr std::string_view name(RefusalCode code) {
  switch (code) {
#define AJP_REFUSAL_NAME(code) \
  case RefusalCode::code:      \
    return #code;
    AJP_REFUSAL_CODES(AJP_REFUSAL_NAME)
#undef AJP_REFUSAL_NAME
  }
  return "Unknown";
}

}  // namespace ajp
