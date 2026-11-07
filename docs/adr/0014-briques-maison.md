# 0014 — Les briques maison qui restent, et pourquoi

## Contexte
L'audit 0.9.1 (Tab 1, A4 ; P5.41) relève des briques écrites dans le dépôt au lieu d'une bibliothèque :
Ed25519, deux SHA-256, inflate, trois lecteurs protobuf, l'encodage des sous-titres, le lecteur `.cube`, une
i18n à deux langues, et `curl` et `tar` du système pour les téléchargements. Il demande, pour chacune, un
remplacement ou une justification écrite.

## Décision
| Brique | Sort | Raison |
|---|---|---|
| Ed25519 (`app/play/ed25519.cpp`) | Gardée | ADR [0013](0013-ed25519-maison-garde.md) : vérification seule, vecteurs RFC 8032 et Wycheproof |
| SHA-256 (`core/sha256_blocks.cpp`) | Une seule compression (P4.10) | `core::Sha256` et `prepare::Sha256` passent par `core::sha256_blocks`, SHA-NI quand le processeur l'a ; OpenSSL ne serait là que pour elle |
| inflate (`core/inflate.cpp`) | Remplacé (P1.30) | Le décodeur maison est parti : le fichier n'est plus qu'une enveloppe en flux autour de zlib (deflate brut), pour les membres de roue et de zip lus par plages |
| Protobuf (`gpu/ir/protobuf.h`) | Un seul lecteur (P1.31), gardé | `prepare/` (écriture champ par champ, `protobuf_fields.h`) et les métadonnées ONNX lisent par `gpu/ir/protobuf.h` ; un ONNX se lit ainsi sans libprotobuf, une dépendance de 30 Mo aux versions qui divergent entre plateformes |
| Encodage des sous-titres (`subs/text_encoding.cpp`) | Remplacé en partie (P3.74) | La conversion des pages de code passe par iconv ou MultiByteToWideChar ; seul le choix de l'encodage reste maison |
| `.cube` (`render/cube_lut.cpp`) | Gardé | Le format tient en une page ; aucune bibliothèque courante ne le lit sans tirer OpenColorIO |
| i18n (`app/play/i18n.h`) | Gardée | Deux langues, des littéraux vérifiés par `i18n_test.cpp` à la compilation des tests ; gettext ajouterait des `.mo` à livrer sur trois systèmes sans rien vérifier de plus |
| `curl` et `tar` du système (`app/play/engine_tools.h`) | Gardés, libcurl et libarchive étudiés | Présents sur les trois systèmes (Windows 10 1803 et plus livre `curl.exe` et `tar.exe`) ; libcurl donnerait la progression et l'annulation sans processus, au prix d'une dépendance TLS à livrer : à refaire si un système sans eux doit être servi |

## Conséquences
- Une nouvelle brique maison demande une ligne dans ce tableau ou une ADR à elle.
