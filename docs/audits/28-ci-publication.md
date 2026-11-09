# Audit 28. CI, publication et signature

**Audit par zone. Priorité : haute.** 101 fichiers, 4 895 lignes ; tests rattachés : 19 fichiers, 366 lignes.

## Mission

Les workflows GitHub, la garde de release, les scripts de paquet, de signature et de publication, le fuzzing.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Garde de release : jobs exigés sur le commit étiqueté (SEC-02).
- Paquet TensorRT Linux : garde, empreinte, `trtexec` présent (SEC-03, TRT-04, TRT-14).
- `--clobber`, signature qui échoue ouverte, secrets pendant la construction (SEC-04, SEC-17, SEC-18).
- Contrôles de qualité : réduisent-ils la dette ou la figent-ils (QUA-01, QUA-06, QUA-08) ?
- Fuzzing : graines, corpus, cibles (SEC-21).

## Frontières

Dépend de :

- [24. Interface : rendu, contrôles et messages](24-interface-rendu.md) (2 inclusions)
- [17. Sous-titres](17-sous-titres.md) (2 inclusions)
- [22. Réglages : modèle, application et persistance](22-reglages-modele.md) (2 inclusions)
- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) (1 inclusion)
- [18. Démarrage et arrêt](18-demarrage-arret.md) (1 inclusion)
- [06. Installation des moteurs et téléchargements](06-installation-moteurs.md) (1 inclusion)
- [26. Mise à jour automatique et signature](26-mise-a-jour.md) (1 inclusion)
- [44. Représentation intermédiaire et import ONNX](44-ir-onnx.md) (1 inclusion)
- [41. HDR, grade et looks](41-hdr-grade.md) (1 inclusion)
- [13. Session de lecture : seek, épisodes et reprise](13-session-lecture.md) (1 inclusion)
- [20. Socle commun (core)](20-socle-core.md) (1 inclusion)

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| B2 | Bloquant | Aucun exécutable n'est signé, sur aucune plateforme, et macOS n'est pas notarisé. Les scripts existent mais restent inactifs. |
| B3 | Bloquant | Le chemin principal n'a jamais tourné avant la livraison : aucun test GPU en CI, installation TensorRT Windows non faite, mise à jour Windows jamais testée, aucune mesure AMD. |
| QUA-01 | Majeur | La base de complexité fige les 110 fonctions du jour, et `--update` la réécrit sans revue. Les `.mm` échappent au contrôle. |
| QUA-10 | Majeur | La PR #567 (203 sous-commits, 333 fichiers) a été fusionnée le jour même, sans autre relecteur. 359 renvois « Px.y » restent dans 191 fichiers. |
| SEC-02 | Majeur | La garde accepte un `ci.yml` vert qui n'a lancé que `rules` (push sur `release/**`, squash merge). Une release peut partir sans ASan, TSan, Windows, macOS ni CUDA. |
| SEC-03 | Majeur | Le module TensorRT Linux se publie depuis toute branche, sans garde, et le premier publié sert à toutes les releases de même empreinte. L'empreinte ignore `linux-tensorrt.sh`, le `CMakeLists.txt` racine et la version de CUDA (voir TRT-14). |
| SEC-04 | Majeur | `--clobber` remplace un zip TensorRT ou un modèle déjà publié par un autre SHA-256. L'installation échoue en `HashMismatch` chez tous les clients de cette version. |
| TRT-04 | Majeur | Un paquet Linux peut partir sans `trtexec`, sans échec. Aucun moteur ne se construit chez aucun client, et le message conseille de réinstaller. |
| QUA-02 | Mineur | Le lint des utilitaires ne voit que les noms : dix copies de `refuse` et cinq mises en minuscules restent, dont une écrite après la règle. Il invalide P5.13. |
| QUA-06 | Mineur | Le plafond de `path::string()` vaut le compte du jour (219), et quatre enveloppes UTF-8 restent. `write_atomic` peut lever malgré son `expected`. |
| QUA-08 | Mineur | Le contrôle du graphe ne lit que `ajp_add_module`. Les backends et l'exécutable, qui incluent `prepare/`, lui échappent. |
| SEC-17 | Mineur | Le jeton de release et le secret Azure sont présents pendant la construction de tout le code tiers. Majeur dès que la signature est active. |
| SEC-18 | Mineur | Sans secret, la signature sort en succès et macOS signe ad hoc. Une release finale peut partir non signée sans échec. |
| SEC-24 | Mineur | Sans re-signature en 120 jours, tout le parc affiche un échec. Un `expires` lointain est accepté sans borne. |
| SEC-26 | Mineur | L'installeur Inno Setup ignore les mises à jour : version affichée figée, fichiers ajoutés laissés à la désinstallation. `updates/previous` n'est jamais purgé. |
| TRT-14 | Mineur | L'empreinte ignore le `CMakeLists.txt` racine, `third_party/`, `linux-tensorrt.sh`, le Dockerfile et le kit CUDA. Un ancien binaire peut être servi (voir SEC-03). |

## Fichiers

| Fichier | Lignes |
|---|---|
| `.github/CODEOWNERS` | 0 |
| `.github/ISSUE_TEMPLATE/bug.yml` | 64 |
| `.github/ISSUE_TEMPLATE/config.yml` | 5 |
| `.github/dependabot.yml` | 8 |
| `.github/workflows/ci.yml` | 291 |
| `.github/workflows/cloud.yml` | 58 |
| `.github/workflows/codeql.yml` | 39 |
| `.github/workflows/deps-watch.yml` | 51 |
| `.github/workflows/diagnose-coreml.yml` | 45 |
| `.github/workflows/export-interpolation-models.yml` | 62 |
| `.github/workflows/fuzz.yml` | 85 |
| `.github/workflows/release-guard.yml` | 40 |
| `.github/workflows/release-linux.yml` | 155 |
| `.github/workflows/release-macos.yml` | 136 |
| `.github/workflows/release-windows.yml` | 203 |
| `.github/workflows/rules.yml` | 41 |
| `.github/workflows/trt-package.yml` | 90 |
| `tests/CMakeLists.txt` | 19 |
| `tests/fuzz/CMakeLists.txt` | 48 |
| `tests/fuzz/corpus/build_log/built_in_inference.log` | 0 |
| `tests/fuzz/corpus/build_log/rife_shapes.log` | 0 |
| `tests/fuzz/corpus/build_log/span.log` | 0 |
| `tests/fuzz/corpus/cube/seed.cube` | 0 |
| `tests/fuzz/corpus/edid/displayid_oled.bin` | 0 |
| `tests/fuzz/corpus/edid/qd_oled.bin` | 0 |
| `tests/fuzz/corpus/inflate/seed.bin` | 0 |
| `tests/fuzz/corpus/inflate/stored.bin` | 0 |
| `tests/fuzz/corpus/manifest/seed.toml` | 9 |
| `tests/fuzz/corpus/onnx/seed.bin` | 0 |
| `tests/fuzz/corpus/options/seed.txt` | 5 |
| `tests/fuzz/corpus/osd_script/leading_empty.txt` | 2 |
| `tests/fuzz/corpus/osd_script/stats.txt` | 1 |
| `tests/fuzz/corpus/play_settings/defaults.toml` | 69 |
| `tests/fuzz/corpus/play_settings/languages.toml` | 5 |
| `tests/fuzz/corpus/play_settings/partial.toml` | 13 |
| `tests/fuzz/corpus/play_settings/table_header_control.toml` | 19 |
| `tests/fuzz/corpus/resume/one.tsv` | 0 |
| `tests/fuzz/corpus/resume/two.tsv` | 0 |
| `tests/fuzz/corpus/settings/array_closing_brace.toml` | 18 |
| `tests/fuzz/corpus/settings/first_run.toml` | 2 |
| `tests/fuzz/corpus/settings/full.toml` | 15 |
| `tests/fuzz/corpus/settings/table_header_equals.toml` | 6 |
| `tests/fuzz/corpus/srt/seed.srt` | 0 |
| `tests/fuzz/fuzz_build_log.cpp` | 22 |
| `tests/fuzz/fuzz_check.h` | 19 |
| `tests/fuzz/fuzz_edid.cpp` | 18 |
| `tests/fuzz/fuzz_osd_script.cpp` | 45 |
| `tests/fuzz/fuzz_parsers.cpp` | 94 |
| `tests/fuzz/fuzz_play_settings.cpp` | 21 |
| `tests/fuzz/fuzz_resume.cpp` | 58 |
| `tests/fuzz/fuzz_settings.cpp` | 21 |
| `tests/fuzz/fuzz_targets.h` | 25 |
| `tests/fuzz/libfuzzer_entry.cpp` | 9 |
| `tests/fuzz/replay_main.cpp` | 220 |
| `tests/support/CMakeLists.txt` | 8 |
| `tests/support/vulkan_test_main.cpp` | 20 |
| `tests/support/vulkan_validation.cpp` | 92 |
| `tests/support/vulkan_validation.h` | 50 |
| `tools/ci/check-comments.sh` | 12 |
| `tools/ci/check-complexity.py` | 70 |
| `tools/ci/check-cuda.sh` | 31 |
| `tools/ci/check-file-size.sh` | 14 |
| `tools/ci/check-module-graph.py` | 142 |
| `tools/ci/check-msl.sh` | 48 |
| `tools/ci/check-no-atoi.sh` | 18 |
| `tools/ci/check-no-getenv.sh` | 23 |
| `tools/ci/check-path-string.sh` | 14 |
| `tools/ci/check-pinned-actions.sh` | 11 |
| `tools/ci/check-shaders.sh` | 46 |
| `tools/ci/check-skips.sh` | 21 |
| `tools/ci/check-utilities.sh` | 36 |
| `tools/ci/check-version.sh` | 11 |
| `tools/ci/clang-tidy-src.sh` | 63 |
| `tools/ci/complexity-baseline.txt` | 111 |
| `tools/ci/coreml_report.py` | 257 |
| `tools/ci/deps_lag.py` | 86 |
| `tools/ci/lsan.supp` | 0 |
| `tools/ci/skip-ceiling.txt` | 7 |
| `tools/release/appimage.sh` | 57 |
| `tools/release/attach-nvidia-linux.sh` | 71 |
| `tools/release/attach-nvidia-windows.ps1` | 61 |
| `tools/release/fetch-models.sh` | 117 |
| `tools/release/linux/LISEZMOI.txt` | 38 |
| `tools/release/nvidia-vendor/LISEZMOI.txt` | 21 |
| `tools/release/nvidia-vendor/tensorrt-linux-x64.lock` | 0 |
| `tools/release/nvidia-vendor/tensorrt-windows-x64.lock` | 0 |
| `tools/release/package-linux.sh` | 237 |
| `tools/release/package-macos.sh` | 160 |
| `tools/release/package-windows.ps1` | 158 |
| `tools/release/preflight/Dockerfile` | 0 |
| `tools/release/preflight/nvidia-package.sh` | 71 |
| `tools/release/preflight/run.sh` | 102 |
| `tools/release/publier-modeles-fp8.sh` | 57 |
| `tools/release/sign-manifest.sh` | 44 |
| `tools/release/trt-fingerprint.sh` | 25 |
| `tools/release/trt-package.sh` | 30 |
| `tools/release/vendor-index.py` | 154 |
| `tools/release/verify-download.sh` | 48 |
| `tools/release/versions.env` | 0 |
| `tools/release/windows/ajp.iss` | 49 |
| `tools/release/windows/sign.ps1` | 48 |

## Tests rattachés

- `tests/unit/app/CMakeLists.txt`
- `tests/unit/audio/CMakeLists.txt`
- `tests/unit/backend_span_fp8_kernels/CMakeLists.txt`
- `tests/unit/backend_tensorrt/CMakeLists.txt`
- `tests/unit/core/CMakeLists.txt`
- `tests/unit/coreml_rt/CMakeLists.txt`
- `tests/unit/export/CMakeLists.txt`
- `tests/unit/gpu/CMakeLists.txt`
- `tests/unit/gpu/boundary_hash_test.cmake`
- `tests/unit/gpu_ir/CMakeLists.txt`
- `tests/unit/gpu_span_fp8/CMakeLists.txt`
- `tests/unit/kernels/CMakeLists.txt`
- `tests/unit/kernels/spec_check.comp`
- `tests/unit/media/CMakeLists.txt`
- `tests/unit/prepare/CMakeLists.txt`
- `tests/unit/render/CMakeLists.txt`
- `tests/unit/subs/CMakeLists.txt`
- `tests/unit/vk_backend/CMakeLists.txt`
- `tests/unit/vk_compute/CMakeLists.txt`

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
