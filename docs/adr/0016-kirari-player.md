# 0016 — ajp devient Kirari Player, un seul dépôt

## Contexte
Le lecteur s'appelait « Ultimate Anime VideoPlayer » dans les textes et `ajp` partout ailleurs (binaire, paquets,
namespace C++, variables, dossiers). Ses releases vivaient dans un second dépôt, `Ta2b/ajp-releases`, alors que le
dépôt des sources est public. Ses données étaient rangées sur deux niveaux (`<données>/ajp/ajp/`,
`SDL_GetPrefPath("ajp", "ajp")`), ce que Tatoub juge peu lisible (novembre 2026).

## Décision
- Le lecteur s'appelle **Kirari Player** (きらり, un éclat qui scintille) ; le binaire, les paquets, le namespace,
  les cibles CMake et les variables `KIRARI_*` suivent. Le renommage est mécanique (un script, gardé hors du dépôt).
- **Un seul dossier** : `<données>/kirari/` (`SDL_GetPrefPath("", "kirari")`), `%LOCALAPPDATA%\kirari` sous
  Windows, `~/.cache/kirari` sous Linux. Au premier lancement, l'ancien dossier d'ajp est renommé en celui de
  Kirari, réglages et moteurs TensorRT compris (`src/app/play/legacy_dirs.h`) ; un dossier Kirari déjà servi
  n'est jamais écrasé.
- **Un seul dépôt** : les releases, les paquets TensorRT et le manifeste des mises à jour vivent dans le dépôt des
  sources (`KIRARI_RELEASE_REPO`, `tools/release/versions.env`). Le jeton `AJP_RELEASE_TOKEN` reste le seul à
  écrire, le jeton du workflow ne fait que lire (B51-12).
- **Relais** : les lecteurs ajp 0.11 et d'avant lisent le manifeste de `Ta2b/ajp-releases`. Chaque release finale
  y dépose aussi son manifeste signé (`release.yml`, `KIRARI_LEGACY_RELEASE_REPO`), que `resign-manifest.yml`
  re-signe avec celui du dépôt.
- **Pas de reprise des anciennes releases** : celles d'ajp restent dans `Ta2b/ajp-releases`, rien n'est recopié
  (leurs numéros et leurs préversions ne suivaient pas de règle claire, dit Tatoub). La release `models-interp-1`,
  dont `tools/release/fetch-models.sh` tire les graphes d'interpolation, est refaite dans le nouveau dépôt par
  `export-interpolation-models.yml` (case « publier »), avant la première release de Kirari.

## Ce qui ne change pas
- Les noms des secrets et variables de GitHub (`AJP_UPDATE_KEY`, `AJP_RELEASE_TOKEN`, `AJP_TS_*`…) : les
  renommer obligerait à les recréer, et une release casserait sur un oubli.
- L'`AppId` de l'installeur Windows : Kirari remplace ajp au lieu de s'installer à côté ; l'installeur retire
  l'ancien dossier et ses raccourcis, après avoir désinscrit `ajp.exe` (« Ouvrir avec », lecteur par défaut).
- Les formats sur disque : la magie `AJPSPAN8` des paquets FP8, les artefacts `.ajpvk` du moteur Vulkan, les
  métadonnées `ajp.gmfss*` des graphes GMFSS exportés.
- Les documents historiques (audits, relevés, notes de release, `CHANGELOG.md`).

## Conséquences
- `src/core`, `src/gpu`, `src/kernels` et `src/prepare` changent : nouveau paquet TensorRT, moteurs reconstruits.
- Le lecteur par défaut se choisit de nouveau après la mise à jour (le ProgId devient `Kirari.Video`).
- `Ta2b/ajp-releases` reste en ligne tant que des lecteurs 0.11 tournent ; vider `KIRARI_LEGACY_RELEASE_REPO`
  arrête le relais.
- Le logo (piste « Étincelle » : le bouton lecture et l'éclat kirari sur sa pointe) a sa source dans
  `tools/release/logo/kirari.svg` ; `node tools/release/logo/render.mjs` en refait `kirari.ico` et `kirari.png`.
  Choisi en novembre 2026, il peut encore changer.
