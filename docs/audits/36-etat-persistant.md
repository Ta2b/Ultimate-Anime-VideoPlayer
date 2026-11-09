# Audit 36. État persistant et compatibilité entre versions

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Tout ce que le lecteur écrit sur le disque doit survivre à une coupure, à une ancienne version et à une nouvelle version.

## Méthode

1. Inventaire de chaque fichier lu ou écrit : réglages, `play.toml`, `update.toml`, `resume.tsv`, caches des plans, des pipelines et des moteurs, journaux, minidumps, fichiers temporaires.
2. Pour chacun : emplacement sous Windows et sous CachyOS, format et version, écriture atomique, lecture d'un fichier abîmé, migration, nettoyage.
3. Vérifier le passage d'une version à la suivante et le retour arrière.

## À couvrir

- Écritures sans fsync.
- Fichiers d'une ancienne version.
- Ce qui reste après une désinstallation.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Écriture et emplacements

| Point | Où | Zone |
|---|---|---|
| écriture atomique commune | `src/core/file_sync.cpp:63` | [20](20-socle-core.md) |
| noms des fichiers de réglages et de mise à jour | `src/app/main.cpp:370` | [18](18-demarrage-arret.md) |
| dossiers de données | `src/app/play/data_dirs.cpp:47` | [19](19-integration-systeme.md) |
| purge des fichiers mal placés | `src/app/play/data_dirs.cpp:101` | [19](19-integration-systeme.md) |
| reprise de position | `src/app/play/resume.cpp:79` | [13](13-session-lecture.md) |
| lecture de play.toml | `src/app/play/settings_toml.cpp:142` | [22](22-reglages-modele.md) |
| lecture des anciennes clés | `src/app/play/settings_toml.cpp:94` | [22](22-reglages-modele.md) |
| écrivain en arrière-plan | `src/app/play/latest_writer.cpp:43` | [22](22-reglages-modele.md) |
| empreinte du cache des moteurs | `src/prepare/cache.cpp:41` | [05](05-construction-moteurs.md) |

### Toutes les écritures de fichiers

Pour chaque ligne : quel fichier, écriture atomique ou non, que se passe-t-il si le processus est coupé au milieu. (53 lignes, zones de la cible seulement.)

| Où | Zone | Code |
|---|---|---|
| `src/gpu/cuda/backend/span_fp8_run.cpp:147` | [02](02-span-fp8.md) | `std::ofstream file(argv[5], std::ios::binary);` |
| `src/prepare/drba_split.cpp:523` | [04](04-onnx-modeles.md) | `if (auto written = write_atomic(dir / file, piece.model); !written)` |
| `src/prepare/drba_split.cpp:530` | [04](04-onnx-modeles.md) | `if (auto written = write_atomic(manifest_path, std::span(raw, text.size())); !written)` |
| `src/prepare/rife_split.cpp:407` | [04](04-onnx-modeles.md) | `if (auto written = write_atomic(dir / file, piece.model); !written)` |
| `src/prepare/rife_split.cpp:414` | [04](04-onnx-modeles.md) | `if (auto written = write_atomic(manifest_path, std::span(raw, text.size())); !written)` |
| `src/prepare/builder.cpp:218` | [05](05-construction-moteurs.md) | `std::ofstream log_file(plan.log_file, std::ios::trunc);` |
| `src/prepare/cache.cpp:43` | [05](05-construction-moteurs.md) | `core::write_text_atomically(entry.digest_file(), digest + '\n');  // « \n » seul, sous Win` |
| `src/prepare/cache_io.cpp:36` | [05](05-construction-moteurs.md) | `std::ofstream file(marker, std::ios::binary);` |
| `src/prepare/cache_io.cpp:101` | [05](05-construction-moteurs.md) | `std::expected<void, Refusal> write_atomic(const fs::path& destination, std::string_view by` |
| `src/prepare/cache_io.cpp:105` | [05](05-construction-moteurs.md) | `std::ofstream file(temporary, std::ios::binary \| std::ios::trunc);` |
| `src/prepare/cache_io.h:36` | [05](05-construction-moteurs.md) | `std::expected<void, Refusal> write_atomic(const std::filesystem::path& destination, std::s` |
| `src/prepare/receipt.cpp:214` | [05](05-construction-moteurs.md) | `return write_atomic(file, receipt_text(receipt));` |
| `src/app/play/engine_install.cpp:305` | [06](06-installation-moteurs.md) | `std::ofstream whole(packed, std::ios::binary \| std::ios::app);` |
| `src/app/play/engine_install.cpp:325` | [06](06-installation-moteurs.md) | `std::ofstream out(partial, std::ios::binary \| std::ios::trunc);` |
| `src/app/play/engine_install.cpp:540` | [06](06-installation-moteurs.md) | `std::ofstream note(staging / std::string(kPackageNote), std::ios::trunc);` |
| `src/app/play/auto_bench.cpp:116` | [07](07-calibration-niveaux.md) | `return core::write_text_atomically(file, format_bench(result));` |
| `src/app/play/auto_bench.cpp:203` | [07](07-calibration-niveaux.md) | `return core::write_text_atomically(file, format_reading(reading));` |
| `src/app/play/auto_bench_run.cpp:129` | [07](07-calibration-niveaux.md) | `std::ofstream(dir / "reglages" / "settings.toml")` |
| `src/app/play/bench_steps.cpp:193` | [07](07-calibration-niveaux.md) | `std::ofstream out(bench_file, std::ios::binary \| std::ios::trunc);` |
| `src/app/play/gpu_profile_out.cpp:26` | [07](07-calibration-niveaux.md) | `if (auto written = core::write_text_atomically(gpu_profile_path(*stats), gpu::gpu_profile_` |
| `src/app/play/machine_calibration.cpp:257` | [07](07-calibration-niveaux.md) | `return core::write_text_atomically(file, format_calibration(calibration));` |
| `src/app/play/span_derive.cpp:28` | [07](07-calibration-niveaux.md) | `std::ofstream file(path, std::ios::binary \| std::ios::trunc);` |
| `src/core/auto_memory.cpp:87` | [07](07-calibration-niveaux.md) | `return write_text_atomically(store, memory.dump());` |
| `src/app/play/chain.cpp:422` | [08](08-boucle-chaine.md) | `void write_record(std::ofstream& stats, const PresentRecord& record) {` |
| `src/app/play/chain.h:174` | [08](08-boucle-chaine.md) | `void write_record(std::ofstream& stats, const PresentRecord& record);` |
| `src/app/play/player_run.cpp:324` | [08](08-boucle-chaine.md) | `std::ofstream stats;` |
| `src/app/play/apple_model.cpp:29` | [09](09-etages-traitement.md) | `if (auto written = core::write_atomic(target, wanted); !written) return std::unexpected(wr` |
| `src/app/play/derived_model.cpp:29` | [09](09-etages-traitement.md) | `if (auto written = core::write_atomic(target, *derived); !written) return std::unexpected(` |
| `src/app/play/placebo_feedback.h:100` | [10](10-presentation-libplacebo.md) | `std::ofstream stamps_;` |
| `src/app/play/placebo_open.cpp:302` | [10](10-presentation-libplacebo.md) | `std::ofstream out(part, std::ios::binary \| std::ios::trunc);` |
| `src/app/play/placebo_snapshot.cpp:62` | [10](10-presentation-libplacebo.md) | `std::ofstream file(path, std::ios::binary);` |
| `src/app/play/resume.cpp:84` | [13](13-session-lecture.md) | `return core::write_text_atomically(store, text);` |
| `src/app/main.cpp:190` | [18](18-demarrage-arret.md) | `std::ofstream out(*options->report, std::ios::binary);` |
| `src/app/main.cpp:230` | [18](18-demarrage-arret.md) | `std::ofstream out(*options->report, std::ios::binary);` |
| `src/app/play/player.cpp:297` | [18](18-demarrage-arret.md) | `if (std::ofstream(relaunch_marker) << "1\n") log->info("moteurs mis à jour : le lecteur se` |
| `src/app/play/player.cpp:638` | [18](18-demarrage-arret.md) | `std::ofstream file(out, std::ios::binary);` |
| `src/app/play/default_player.cpp:296` | [19](19-integration-systeme.md) | `if (auto written = core::write_text_atomically(file, *text); !written)` |
| `src/core/file_sync.cpp:63` | [20](20-socle-core.md) | `std::expected<void, Refusal> write_atomic(const std::filesystem::path& path,` |
| `src/core/file_sync.cpp:72` | [20](20-socle-core.md) | `std::ofstream out(temp, std::ios::binary \| std::ios::trunc);` |
| `src/core/file_sync.cpp:96` | [20](20-socle-core.md) | `std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std:` |
| `src/core/file_sync.cpp:97` | [20](20-socle-core.md) | `return write_atomic(path, std::as_bytes(std::span(text)));` |
| `src/core/file_sync.h:27` | [20](20-socle-core.md) | `std::expected<void, Refusal> write_atomic(const std::filesystem::path& path,` |
| `src/core/file_sync.h:29` | [20](20-socle-core.md) | `std::expected<void, Refusal> write_text_atomically(const std::filesystem::path& path, std:` |
| `src/core/user_overlay.cpp:255` | [20](20-socle-core.md) | `if (auto written = write_text_atomically(user, *updated); !written) return std::unexpected` |
| `src/app/play/settings_toml.cpp:626` | [22](22-reglages-modele.md) | `return core::write_text_atomically(path, text);` |
| `src/core/settings.cpp:286` | [22](22-reglages-modele.md) | `return write_text_atomically(path, *text);` |
| `src/app/play/updater.cpp:198` | [26](26-mise-a-jour.md) | `return core::write_text_atomically(install / kFileList, text);` |
| `src/app/play/updater.cpp:326` | [26](26-mise-a-jour.md) | `return core::write_text_atomically(file, format_update_state(state));` |
| `src/app/play/updater.cpp:489` | [26](26-mise-a-jour.md) | `std::ofstream out(probe, std::ios::binary);` |
| `src/media/shot_cache.cpp:139` | [42](42-plans-coupes.md) | `std::ofstream stream(temporary, std::ios::binary \| std::ios::trunc);` |
| `src/gpu/ir/onnx_info_main.cpp:133` | [44](44-ir-onnx.md) | `std::ofstream file(options.dump, std::ios::binary);` |
| `src/gpu/vulkan/backend/net_build.cpp:204` | [45](45-backend-vulkan.md) | `std::ofstream out(temporary, std::ios::binary \| std::ios::trunc);` |
| `src/kernels/glsl_main.cpp:27` | [45](45-backend-vulkan.md) | `std::ofstream out(args[1], std::ios::binary \| std::ios::trunc);` |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| QUA-05 | Majeur | Six écritures restent faites à la main, sans fsync, contre P5.11. Une coupure peut laisser un moteur Vulkan ou un cache vide. |
| QUA-07 | Majeur | Sous Windows, l'échec d'écriture de `update.toml` (antivirus) est ignoré. Le lancement compte comme un plantage, et la version peut être retirée après 3 lancements. |
| WIN-02 | Majeur | Avec un profil redirigé, la migration copie plus de 1 Go sur le thread principal, avant la fenêtre. Le lecteur semble ne pas démarrer pendant des minutes. |
| SEC-16 | Mineur | `purge_misplaced` efface à chaque lancement les dossiers qui « ressemblent à des versions » dans `%APPDATA%` ou `~/.local/share`. D'autres programmes peuvent y perdre des données. |
| UX-19 | Mineur | Le dossier des moteurs est enregistré à chaque frappe. Quitter pendant la saisie garde un chemin à moitié tapé. |
| WIN-03 | Mineur | Jusqu'à environ 192 Mo de journal et 5 minidumps restent dans le profil itinérant. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [02. SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md)
- [04. Analyse et découpe des modèles ONNX](04-onnx-modeles.md)
- [05. Construction des moteurs, cache et trtexec](05-construction-moteurs.md)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md)
- [07. Calibration, niveaux et mode Auto](07-calibration-niveaux.md)
- [08. Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md)
- [09. Étages d'agrandissement et d'interpolation](09-etages-traitement.md)
- [10. Rendu et présentation par libplacebo](10-presentation-libplacebo.md)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md)
- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [19. Intégration au système](19-integration-systeme.md)
- [20. Socle commun (core)](20-socle-core.md)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md)
- [42. Analyse des plans, coupes et pré-analyse](42-plans-coupes.md)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md)
- [45. Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md)
