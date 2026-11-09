# Complément de l'audit 27. Build, dépendances et ports

Branche : `claude/project-thread-n40ubw`. Commit vérifié : `bdfe44c`. Date : `2026-11-09`.

Renvois reçus après le rapport `27-build-dependances.md`, vérifiés dans le code. Les IDs suivent ceux du
rapport (dernier : A27-13).

## Verdict

Le seul renvoi (de l'audit 24) est confirmé, et un peu plus large qu'annoncé : aucun backend d'ImGui n'est
employé, ni sous Linux ni sous Windows. Défaut mineur de construction, sans effet sur l'image ni sur la vitesse
de lecture.

## Renvois traités

| De | Où | Conclusion |
|---|---|---|
| 24 | `cmake/Dependencies.cmake:123` | Confirmé : A27-14. |

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A27-14 | Mineur | `cmake/Dependencies.cmake:123-125`, `vcpkg.json:58-64` | Le préfixe Linux compile `backends/imgui_impl_sdl3.cpp` dans `ajp_imgui` et lie `SDL3::SDL3` en `PUBLIC` pour lui seul, alors que rien n'appelle `ImGui_ImplSDL3_*` : `PlayerUi::handle` (`src/app/play/ui.cpp:385` et suivantes) remplit `ImGuiIO` lui-même (`AddMousePosEvent`, `AddKeyEvent`), et le rendu passe par libplacebo (`placebo_imgui.cpp`). Sous Windows, le port `imgui` est demandé avec `sdl3-binding` et `vulkan-binding`, deux fonctions tout aussi inutilisées (aucun `imgui_impl_` ni `ImGui_Impl` dans `src/` ni `tests/`). | Chaque construction propre compile un fichier mort (et, sous vcpkg, deux backends) ; une montée d'ImGui ou de SDL3 qui casse `imgui_impl_sdl3.cpp` (l'API SDL3 et ce backend changent souvent ensemble) fait échouer la construction pour du code que le lecteur n'exécute jamais. Le `target_include_directories(... backends)` laisse aussi croire à un relecteur que l'interface passe par ce backend. Rien de visible pour l'utilisateur : l'éditeur de liens n'emporte pas l'objet inutilisé de la bibliothèque statique. | Retirer `imgui_impl_sdl3.cpp`, le dossier `backends` des inclusions et le lien `SDL3::SDL3` de `ajp_imgui` (ImGui lui-même n'en a pas besoin) ; retirer `sdl3-binding` et `vulkan-binding` des fonctions du port `imgui` dans `vcpkg.json`. Le script du préfixe peut ne plus copier `backends/` (`tools/deps/linux-from-source.sh:108`). Vérifier ensuite que `ajp_app_play` lie toujours SDL3 directement (`ajp::dep::sdl3`). | Lu : `grep -rn "ImGui_Impl\|imgui_impl" src tests` ne trouve rien ; seule `cmake/Dependencies.cmake:123` cite le fichier. Dépôt superficiel : l'historique ne dit pas si le backend a servi autrefois. |

## Doublons

Aucun : `SYNTHESE.md` ne cite ce fichier qu'au renvoi 24 → 27 (ligne 879), que ce complément ferme.

## Baisse de qualité pour la vitesse

Aucune dans ce renvoi.
