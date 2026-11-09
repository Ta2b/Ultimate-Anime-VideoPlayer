# Audit 09. Étages d'agrandissement et d'interpolation

**Audit par zone. Priorité : haute.** 30 fichiers, 3 407 lignes ; tests rattachés : 19 fichiers, 3 061 lignes.

## Mission

Les étages qui font le travail image par image : SPAN, RIFE ou DistilDRBA, ordre warp et SPAN, plan d'interpolation selon les coupes, mise à l'échelle finale, choix des modèles et des précisions.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- `UpscaleStage::outputs` (CCN 172) : chaque branche, chaque cas de coupe et de dessin tenu.
- Ordre warp et SPAN (Avant, Après, Auto) : critère de choix et qualité.
- Plan d'interpolation : seuils réglés à l'œil, coupes, fondus.
- Choix du modèle et de la précision : FP8, INT8, FP16, cohérence avec les capacités du moteur.
- Anneau : écriture, emplacements, débit.

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (20 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (16 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (9 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (7 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (4 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (4 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (4 inclusions)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (2 inclusions)
- [43. Textes et traductions](43-textes-traductions.md) (2 inclusions)
- [17. Sous-titres](17-sous-titres.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [23. Réglages : le panneau](23-reglages-panneau.md) (1 inclusion)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (15 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (4 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (3 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (2 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| SEC-21 | Mineur | La graine ONNX fait 16 octets, le corpus est jeté chaque nuit, et `scan_onnx` n'est pas fuzzé. CodeQL tourne sans construction. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/apple_model.cpp` | 33 |
| `src/app/play/apple_model.h` | 18 |
| `src/app/play/compose_key.h` | 112 |
| `src/app/play/derived_model.cpp` | 33 |
| `src/app/play/derived_model.h` | 29 |
| `src/app/play/drba_fused_choice.cpp` | 13 |
| `src/app/play/drba_fused_choice.h` | 16 |
| `src/app/play/drba_model.cpp` | 43 |
| `src/app/play/drba_model.h` | 46 |
| `src/app/play/final_scaler.h` | 15 |
| `src/app/play/model_choice.cpp` | 95 |
| `src/app/play/model_choice.h` | 78 |
| `src/app/play/motion_model.cpp` | 125 |
| `src/app/play/motion_model.h` | 28 |
| `src/app/play/precision.h` | 66 |
| `src/app/play/rife_fused_choice.cpp` | 16 |
| `src/app/play/rife_fused_choice.h` | 16 |
| `src/app/play/rife_plan.cpp` | 26 |
| `src/app/play/rife_plan.h` | 67 |
| `src/app/play/ring_stage.cpp` | 216 |
| `src/app/play/ring_stage.h` | 134 |
| `src/app/play/span_auto.cpp` | 106 |
| `src/app/play/span_auto.h` | 73 |
| `src/app/play/upscale_ahead.cpp` | 148 |
| `src/app/play/upscale_drba.cpp` | 282 |
| `src/app/play/upscale_late.cpp` | 42 |
| `src/app/play/upscale_motion.cpp` | 195 |
| `src/app/play/upscale_smooth.cpp` | 56 |
| `src/app/play/upscale_stage.cpp` | 776 |
| `src/app/play/upscale_stage.h` | 504 |

## Tests rattachés

- `tests/unit/app/apple_model_test.cpp`
- `tests/unit/app/compose_key_test.cpp`
- `tests/unit/app/doubles.cpp`
- `tests/unit/app/doubles.h`
- `tests/unit/app/drba_fused_choice_test.cpp`
- `tests/unit/app/late_fate_stage_test.cpp`
- `tests/unit/app/model_choice_test.cpp`
- `tests/unit/app/motion_model_test.cpp`
- `tests/unit/app/rife_fused_choice_test.cpp`
- `tests/unit/app/rife_plan_test.cpp`
- `tests/unit/app/ring_stage_test.cpp`
- `tests/unit/app/smooth_stage_test.cpp`
- `tests/unit/app/span_auto_test.cpp`
- `tests/unit/app/stage_rig.cpp`
- `tests/unit/app/stage_rig.h`
- `tests/unit/app/steady_stage_test.cpp`
- `tests/unit/app/upscale_rig.h`
- `tests/unit/app/upscale_stage_drba_test.cpp`
- `tests/unit/app/upscale_stage_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
