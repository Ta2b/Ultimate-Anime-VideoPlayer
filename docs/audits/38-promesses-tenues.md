# Audit 38. Promesses tenues

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Tout ce que le lecteur promet doit être vrai : README, aide de la ligne de commande, textes de l'assistant et du panneau, notes de version.

## Méthode

1. Relever chaque promesse : une fonction, un défaut, une durée, un comportement.
2. La vérifier dans le code, et noter « tenue », « fausse » ou « invérifiable ».
3. Proposer la correction du texte ou du code.

## À couvrir

- README et captures d'écran.
- `--help` et options.
- Durées annoncées, comme « environ 15 min ».
- Notes de version récentes.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Où sont les promesses

| Point | Où | Zone |
|---|---|---|
| README du dépôt | `README.md` | — |
| aide de la ligne de commande | `src/app/play/options.cpp:242` | [18](18-demarrage-arret.md) |
| textes de l'assistant | `src/app/play/i18n_fr_wizard.cpp` | [43](43-textes-traductions.md) |
| textes du panneau | `src/app/play/i18n_fr_settings.cpp` | [43](43-textes-traductions.md) |
| modèle de rapport de bug | `.github/ISSUE_TEMPLATE/bug.yml` | [28](28-ci-publication.md) |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| TRT-02 | Majeur | « Annuler et regarder » joue la vidéo réduite à la hauteur de travail, sans SPAN ni RIFE. `--no-upscale` a le même défaut. |
| UX-05 | Majeur | La durée de « Optimise now » est calculée avant le choix de TensorRT. Le Ready annonce environ 5 min au lieu d'environ 20, téléchargement non compris. |
| UX-13 | Majeur | « Report a problem » promet un journal sans dossiers, mais les chemins complets hors du dossier personnel y restent (voir WIN-08). |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [28. CI, publication et signature](28-ci-publication.md)
- [43. Textes et traductions](43-textes-traductions.md)
