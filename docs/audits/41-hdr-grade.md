# Audit 41. HDR, grade et looks

**Audit par zone. Priorité : moyenne.** 29 fichiers, 3 849 lignes ; tests rattachés : 10 fichiers, 1 639 lignes.

## Mission

La sortie HDR et SDR, la compression vers le pic de l'écran, le grade animé, les LUT et les looks, la netteté.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Compression HDR selon le système et le bureau : Windows, KDE, GNOME, macOS (GPU-08).
- Métadonnées HDR10 envoyées à l'écran : statiques ou changeantes (GPU-07).
- LUT `.cube` : domaine d'entrée, bornes, fichiers hostiles.
- Thread de composition des LUT et appels `pl_gpu` depuis un second thread (GPU-11, GPU-12).
- Banding : anneau RGB10A2 avant un grade qui étire le contraste (GPU-15).
- Coût : nombre de passes plein écran, résolution à laquelle le grade tourne.
- Textes en dur (GPU-14).

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (13 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (2 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (2 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (1 inclusion)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (1 inclusion)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (1 inclusion)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (1 inclusion)

Utilisé par :

- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (5 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (3 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (3 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [28. CI, publication et signature](28-ci-publication.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-07 | Majeur | Le limiteur de flashs change les métadonnées HDR10 en cours de lecture (1 000 puis 250 nits). Le téléviseur change de courbe à chaque séquence. |
| GPU-08 | Majeur | La compression P0.16 ne joue pas sur macOS. Sous Wayland hors KDE, elle suppose un blanc à 203 nits. |
| GPU-11 | Mineur | Le thread des LUT est un `std::jthread` brut qui crée une texture `pl_gpu` en parallèle du rendu. Un `bad_alloc` mène à `std::terminate`. |
| GPU-12 | Mineur | `LookLoader` détache un thread occupé, qui peut journaliser après `spdlog::shutdown`. |
| GPU-14 | Mineur | « Vrai noir : mesure impossible (voir le journal). » est en français en dur. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/chain_hdr.cpp` | 386 |
| `src/app/play/look_catalog.cpp` | 145 |
| `src/app/play/look_catalog.h` | 45 |
| `src/app/play/placebo_grade.cpp` | 358 |
| `src/render/CMakeLists.txt` | 26 |
| `src/render/cube_lut.cpp` | 336 |
| `src/render/cube_lut.h` | 70 |
| `src/render/display_model.cpp` | 74 |
| `src/render/display_model.h` | 86 |
| `src/render/embed_shaders.cmake` | 35 |
| `src/render/emitters.cpp` | 85 |
| `src/render/emitters.h` | 35 |
| `src/render/faces.cpp` | 139 |
| `src/render/faces.h` | 76 |
| `src/render/grade.cpp` | 474 |
| `src/render/grade.h` | 78 |
| `src/render/grade_params.cpp` | 188 |
| `src/render/grade_params.h` | 193 |
| `src/render/shaders.h` | 24 |
| `src/render/shaders/anime-hdr-anime/1-local.hook` | 44 |
| `src/render/shaders/anime-hdr-anime/2-area.hook` | 59 |
| `src/render/shaders/anime-hdr-anime/3-anime.hook` | 162 |
| `src/render/shaders/anime-hdr-black.hook` | 243 |
| `src/render/shaders/anime-hdr-calibration.hook` | 42 |
| `src/render/shaders/anime-hdr-look.hook` | 147 |
| `src/render/shaders/anime-hdr-params.hook` | 177 |
| `src/render/shaders/sharpen.hook` | 36 |
| `src/render/sharpen.cpp` | 49 |
| `src/render/sharpen.h` | 37 |

## Tests rattachés

- `tests/unit/app/look_v2_test.cpp`
- `tests/unit/render/cube_lut_test.cpp`
- `tests/unit/render/display_model_test.cpp`
- `tests/unit/render/emitters_test.cpp`
- `tests/unit/render/faces_test.cpp`
- `tests/unit/render/grade_gpu_anime_test.cpp`
- `tests/unit/render/grade_gpu_support.h`
- `tests/unit/render/grade_gpu_test.cpp`
- `tests/unit/render/grade_params_test.cpp`
- `tests/unit/render/sharpen_gpu_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
