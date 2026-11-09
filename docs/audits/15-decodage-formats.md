# Audit 15. Décodage, matériel, formats et couleurs

**Audit par zone. Priorité : haute.** 17 fichiers, 3 145 lignes ; tests rattachés : 10 fichiers, 1 817 lignes.

## Mission

Le décodage logiciel et matériel (NVDEC, D3D11VA, VA-API, Vulkan Video), les conversions de formats et de couleurs, le désentrelacement, les captures.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Formats depuis NVDEC et Vulkan Video : 4:4:4, 4:2:2, 12 bits (MED-02, MED-04).
- Conversion P0.11 : coût, allocations, avant le tri du seek (MED-03).
- Couleurs : matrices, plages, chroma, HDR10, HLG, Dolby Vision.
- Repli matériel vers logiciel en cours de flux.
- Désentrelacement NVDEC et Vulkan, IVTC (MED-13).
- Horodatages et débordements (MED-14).

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (16 inclusions)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (12 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (2 inclusions)

Utilisé par :

- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (2 inclusions)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md) (2 inclusions)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) (2 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (1 inclusion)
- [47. Export vidéo](47-export.md) (1 inclusion)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (1 inclusion)
- [14. Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| MED-02 | Majeur | La conversion P0.11 ne traite que les images hôte. Une source 4:4:4 ou 4:2:2 décodée par NVDEC ou Vulkan Video s'arrête dès la première image. |
| MED-03 | Mineur | La conversion P0.11 alloue environ 25 Mo par image 4K sans pool, en monothread, même pour les images jetées et dans la pré-analyse. Majeur en 4K si la mesure le confirme. |
| MED-04 | Mineur | Position de chroma fixée en dur et profondeur incohérente selon le 4:4:4. Gris, RGB planaire, YCgCo et ICtCp restent refusés, et seulement en cours de lecture. |
| MED-13 | Mineur | Les images NVDEC et Vulkan entrelacées passent sans bwdif si `field_order` est inconnu. Sans IVTC, un 1080i télécinéma garde la cadence 3:2. |
| MED-16 | Mineur | En VFR, le seek exact et le pas d'image utilisent la période moyenne. La cible peut sauter l'image visible, ou le pas reculer de deux images. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/media/color.cpp` | 346 |
| `src/media/color.h` | 31 |
| `src/media/cuda_driver.cpp` | 450 |
| `src/media/cuda_driver.h` | 112 |
| `src/media/deinterlace.cpp` | 116 |
| `src/media/deinterlace.h` | 37 |
| `src/media/frame_data.cpp` | 318 |
| `src/media/frame_data.h` | 35 |
| `src/media/hw_api.h` | 89 |
| `src/media/hw_device.cpp` | 357 |
| `src/media/hw_device.h` | 75 |
| `src/media/stills.cpp` | 351 |
| `src/media/stills.h` | 85 |
| `src/media/vaapi_device.cpp` | 109 |
| `src/media/video_decoder.cpp` | 509 |
| `src/media/video_decoder.h` | 71 |
| `src/media/yuv_sample.h` | 54 |

## Tests rattachés

- `tests/unit/media/clip_writer.cpp`
- `tests/unit/media/clip_writer.h`
- `tests/unit/media/clips.cpp`
- `tests/unit/media/clips.h`
- `tests/unit/media/deinterlace_test.cpp`
- `tests/unit/media/frame_data_test.cpp`
- `tests/unit/media/gpu_decode_test.cpp`
- `tests/unit/media/hw_test.cpp`
- `tests/unit/media/seek_test.cpp`
- `tests/unit/media/stills_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
