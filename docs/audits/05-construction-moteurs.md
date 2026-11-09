# Audit 05. Construction des moteurs, cache et trtexec

**Audit par zone. Priorité : très haute.** 28 fichiers, 3 104 lignes ; tests rattachés : 9 fichiers, 2 103 lignes.

## Mission

La transformation d'un modèle en moteur prêt : identité et clé de cache, verrou entre processus, lancement de `trtexec` et suivi de sa progression, écriture du cache.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Clé de cache : couvre-t-elle tout ce qui change le moteur (pilote, TensorRT, architecture, forme, précision) ?
- Verrou entre processus : délai, processus mort, deux lecteurs en même temps.
- `trtexec` : seul celui du paquet, empreinte vérifiée juste avant le lancement, délai (TRT-05, SEC-20).
- Lancement de processus : arguments, sortie lue, arrêt propre, objets Job et groupes.
- Formes statiques : une construction par géométrie de source, coût réel.
- Écriture du cache : atomique, cache corrompu détecté.
- Dossiers temporaires à nom fixe (TRT-18). Code mort : `preparer`, `receipt` (QUA-03).

## Frontières

Dépend de :

- [20. Socle commun (core)](20-socle-core.md) (18 inclusions)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (9 inclusions)
- [21. Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) (4 inclusions)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)

Utilisé par :

- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md) (5 inclusions)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md) (4 inclusions)
- [01. TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) (4 inclusions)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md) (2 inclusions)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) (2 inclusions)
- [03. Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) (2 inclusions)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) (2 inclusions)
- [19. Intégration au système](19-integration-systeme.md) (1 inclusion)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| TRT-05 | Mineur | Le délai de `trtexec` (20 min) égale la plus longue construction mesurée. Sur un portable lent, la construction échoue et recommence à chaque ouverture. |
| TRT-17 | Mineur | Défauts groupés : échec de `nvidia-smi` gardé, carte inconnue en « ptx », moteurs reconstruits à chaque pilote sans purge, doc MPEG-4 fausse, largeur en 16/9 sur ultra-large. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `src/prepare/CMakeLists.txt` | 18 |
| `src/prepare/builder.cpp` | 258 |
| `src/prepare/builder.h` | 91 |
| `src/prepare/cache.cpp` | 137 |
| `src/prepare/cache.h` | 58 |
| `src/prepare/cache_io.cpp` | 190 |
| `src/prepare/cache_io.h` | 55 |
| `src/prepare/identity.cpp` | 199 |
| `src/prepare/identity.h` | 103 |
| `src/prepare/key_lock.cpp` | 134 |
| `src/prepare/key_lock.h` | 48 |
| `src/prepare/log.h` | 16 |
| `src/prepare/preparer.cpp` | 131 |
| `src/prepare/preparer.h` | 89 |
| `src/prepare/process.h` | 79 |
| `src/prepare/process_posix.cpp` | 228 |
| `src/prepare/process_text.cpp` | 82 |
| `src/prepare/process_win.cpp` | 197 |
| `src/prepare/receipt.cpp` | 233 |
| `src/prepare/receipt.h` | 94 |
| `src/prepare/receipt_fields.cpp` | 79 |
| `src/prepare/receipt_fields.h` | 68 |
| `src/prepare/sha256.cpp` | 120 |
| `src/prepare/sha256.h` | 54 |
| `src/prepare/trtexec_command.cpp` | 171 |
| `src/prepare/trtexec_command.h` | 94 |
| `src/prepare/trtexec_progress.cpp` | 42 |
| `src/prepare/trtexec_progress.h` | 36 |

## Tests rattachés

- `tests/unit/prepare/builder_test.cpp`
- `tests/unit/prepare/cache_test.cpp`
- `tests/unit/prepare/helper.cpp`
- `tests/unit/prepare/identity_test.cpp`
- `tests/unit/prepare/preparer_test.cpp`
- `tests/unit/prepare/process_test.cpp`
- `tests/unit/prepare/receipt_test.cpp`
- `tests/unit/prepare/support.h`
- `tests/unit/prepare/trtexec_command_test.cpp`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
