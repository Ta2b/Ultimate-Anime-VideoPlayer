# Audit 27. Build, dépendances et ports

**Audit par zone. Priorité : haute.** 30 fichiers, 2 932 lignes ; tests rattachés : 0 fichiers, 0 lignes.

## Mission

Comment le lecteur se construit : CMake et préréglages, options de compilation, ports vcpkg patchés (libplacebo, libass), dépendances Linux et Windows.

## Points à auditer en profondeur

Un point de départ : l'audit doit aussi chercher ce que la liste ne nomme pas.

- Options de la release : LTO, symboles, avertissements, mêmes options que celles mesurées.
- Reproductibilité : versions épinglées, CUDA, marqueurs de cache (SEC-19).
- Ports patchés : patchs justifiés, à jour avec l'amont.
- En-têtes précompilés et contrôles de qualité (QUA-09).
- Construction sous Windows et sous CachyOS (Arch) : écarts de versions.

## Frontières

Auditer l'interface avec ces zones (ce qui entre, ce qui sort, qui possède quoi, quel thread appelle), pas leur intérieur. Les parcours complets sont couverts par les audits transverses.

## Défauts déjà connus

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier un par un, sans s'y limiter.

| ID | Gravité | Défaut |
|---|---|---|
| SEC-01 | Bloquant | Aucune clé de secours n'est compilée et la clé unique reste sur un PC personnel ; la révocation prévue par P1.14 est ignorée sans erreur. Un vol de clé donne l'exécution de code sur tout le parc. |
| QUA-03 | Majeur | Environ 2 500 lignes de modules orphelins restent compilées et testées (`receipt`, `debug_flags`, `drawings`…). La chaîne `profile` (852 lignes) n'a plus d'appelant. |
| QUA-09 | Mineur | `AJP_PCH` vaut OFF et aucun preset ne l'active. Le gain annoncé n'existe pas. |
| SEC-19 | Mineur | CUDA n'est pas épinglé, l'archive de `trtexec` est extraite avant son SHA, et les marqueurs de cache suivent l'étiquette. `pip` installe sans version ni empreinte. |
| SEC-20 | Mineur | Le repli `trtexec` n'est retiré qu'en `Release`. Les préréglages `RelWithDebInfo` recommandés le gardent. |

## Fichiers

| Fichier | Lignes |
|---|---|
| `.clang-format` | 17 |
| `.clang-tidy` | 36 |
| `.gitattributes` | 8 |
| `.gitignore` | 16 |
| `CMakeLists.txt` | 114 |
| `CMakePresets.json` | 492 |
| `LICENSE` | 373 |
| `cmake/AjpModule.cmake` | 78 |
| `cmake/AjpSpirv.cmake` | 192 |
| `cmake/Dependencies.cmake` | 164 |
| `cmake/TomlFixes.cmake` | 109 |
| `cmake/WindowsUtf8.cmake` | 29 |
| `cmake/utf8.manifest` | 9 |
| `ports/libass/portfile.cmake` | 39 |
| `ports/libass/vcpkg.json` | 15 |
| `ports/libplacebo/0001-ajp-present-timing.patch` | 260 |
| `ports/libplacebo/clang-cl.cmake` | 9 |
| `ports/libplacebo/portfile.cmake` | 56 |
| `ports/libplacebo/vcpkg.json` | 15 |
| `ports/libunibreak/CMakeLists.txt` | 49 |
| `ports/libunibreak/fix_export.patch` | 24 |
| `ports/libunibreak/libunibreak-config.cmake.in` | 32 |
| `ports/libunibreak/portfile.cmake` | 41 |
| `ports/libunibreak/vcpkg.json` | 14 |
| `src/app/play/embed_text.cmake` | 16 |
| `tools/deps/linux-from-source.sh` | 172 |
| `tools/deps/linux-tensorrt.sh` | 138 |
| `tools/deps/macos-from-source.sh` | 226 |
| `tools/deps/windows-tensorrt.ps1` | 111 |
| `vcpkg.json` | 78 |

## Hors de cet audit

Tout fichier non listé ci-dessus. Un défaut trouvé ailleurs va dans « Renvois » du rapport, avec le numéro de l'audit concerné.
