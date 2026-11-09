# Audit 43. Textes et traductions

**Audit par zone. Priorité : moyenne.** 13 fichiers, 1 735 lignes ; tests rattachés : 2 fichiers, 223 lignes.

## Mission

Toutes les chaînes visibles, en anglais et en français : traductions, longueur, vocabulaire, nombres et unités.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Traductions orphelines et textes en dur hors de `tr()` (UX-09, UX-21, GPU-14).
- Nombres : virgule ou point selon la langue (UX-10).
- Un seul nom par notion (calibration, niveaux).
- Longueur des notes et infobulles, jargon.
- Textes figés dans l'ancienne langue jusqu'à la vidéo suivante (UX-20).

## Frontières

Utilisé par :

- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (8 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (7 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (6 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (3 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (3 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (2 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (2 inclusions)
- [47. Export vidéo](47-export.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (2 inclusions)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (2 inclusions)
- [23. Réglages : le panneau](23-reglages-panneau.md) (2 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [17. Sous-titres](17-sous-titres.md) (1 inclusion)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/french_text.h` | 32 |
| `src/app/play/i18n.cpp` | 62 |
| `src/app/play/i18n.h` | 71 |
| `src/app/play/i18n_fr.cpp` | 131 |
| `src/app/play/i18n_fr.h` | 21 |
| `src/app/play/i18n_fr_image.cpp` | 148 |
| `src/app/play/i18n_fr_messages.cpp` | 205 |
| `src/app/play/i18n_fr_settings.cpp` | 319 |
| `src/app/play/i18n_fr_sound.cpp` | 187 |
| `src/app/play/i18n_fr_stats.cpp` | 73 |
| `src/app/play/i18n_fr_system.cpp` | 171 |
| `src/app/play/i18n_fr_visible.cpp` | 145 |
| `src/app/play/i18n_fr_wizard.cpp` | 170 |

## Tests rattachés

- `tests/unit/app/french_ui.h`
- `tests/unit/app/i18n_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
