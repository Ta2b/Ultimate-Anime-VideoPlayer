# Audit 10. Rendu et présentation par libplacebo

**Audit par zone. Priorité : haute.** 18 fichiers, 3 438 lignes ; tests rattachés : 4 fichiers, 343 lignes.

## Mission

L'ouverture de libplacebo, la swapchain, la présentation de chaque image, le retour de présentation, l'anneau côté rendu et son veilleur d'écriture, les captures.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Perte d'appareil ou de surface : détection, message, journal inondé (GPU-06, GPU-17).
- Retour de présentation : niveau obtenu sous Windows et sous CachyOS, repli quand il manque.
- Anneau et veilleur : `forget_all`, timelines, générations (GPU-10).
- Swapchain : recréation à la taille, mode FIFO, fenêtre réduite ou cachée.
- Glisser ou redimensionner la fenêtre sous Windows : image figée ?

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (13 inclusions)
- [12. Cadence et horloges](12-cadence-horloges.md) (6 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (5 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (4 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (2 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (2 inclusions)
- [15. Décodage, matériel, formats et couleurs](15-decodage-formats.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [17. Sous-titres](17-sous-titres.md) (1 inclusion)

Utilisé par :

- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (3 inclusions)
- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (3 inclusions)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (3 inclusions)
- [47. Export vidéo](47-export.md) (3 inclusions)
- [41. HDR, grade et looks](41-hdr-grade.md) (2 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [25. Assistant de premier lancement](25-assistant.md) (1 inclusion)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (1 inclusion)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| GPU-06 | Majeur | Une perte d'appareil ou de surface côté présentation fige l'image sans message. Le journal reçoit 250 à 1 000 lignes par seconde. |
| GPU-10 | Mineur | Après une erreur de `vkWaitSemaphores`, `forget_all` peut attendre sans fin (latent). La boucle sonde ensuite à 4 ms. |
| GPU-17 | Mineur | Un GPU bloqué sans être perdu n'est pas détecté. La fermeture peut rester bloquée dans `pl_gpu_finish`. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/app/play/placebo_feedback.cpp` | 360 |
| `src/app/play/placebo_feedback.h` | 123 |
| `src/app/play/placebo_impl.h` | 257 |
| `src/app/play/placebo_offscreen.cpp` | 55 |
| `src/app/play/placebo_open.cpp` | 656 |
| `src/app/play/placebo_present.cpp` | 101 |
| `src/app/play/placebo_present.h` | 48 |
| `src/app/play/placebo_ring.cpp` | 253 |
| `src/app/play/placebo_screen.cpp` | 485 |
| `src/app/play/placebo_screen.h` | 194 |
| `src/app/play/placebo_snapshot.cpp` | 80 |
| `src/app/play/placebo_wayland.cpp` | 208 |
| `src/app/play/placebo_wayland.h` | 67 |
| `src/app/play/present_level.h` | 121 |
| `src/app/play/ring_watcher.cpp` | 164 |
| `src/app/play/ring_watcher.h` | 77 |
| `src/app/play/vulkan_share.cpp` | 142 |
| `src/app/play/vulkan_share.h` | 47 |

## Tests rattachés

- `tests/unit/app/present_level_test.cpp`
- `tests/unit/app/ring_interop_test.cpp`
- `tests/unit/app/ring_watcher_test.cpp`
- `tests/unit/app/wayland_stamp_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
