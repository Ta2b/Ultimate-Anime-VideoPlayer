# Complément de l'audit 19. Intégration au système

Branche : `claude/project-thread-k4vta3`. Commit audité : `0cfcdee`. Date : `2026-11-09`.

## Verdict

Les deux renvois sont des doublons : le double `update-desktop-database` est **A25-07**, le canal nommé à nom
prévisible est **A34-03** (et le défaut connu **SEC-15**, déjà revérifié par le rapport 19). Aucun nouveau
défaut, et aucune baisse de qualité d'image faite pour la vitesse dans ces renvois.

## Cadrage retenu

Seuls les renvois de `complements/19.md` sont vérifiés, dans le code, sans question à Tatoub. Rien n'a été compilé
ni lancé.

## Défauts

Aucun nouveau défaut.

## Doublons

| Renvoi | Où | Conclusion |
|---|---|---|
| de 25 : `update-desktop-database` lancé deux fois de suite | `src/app/play/default_player.cpp:334-336`, `:300` ; `src/app/play/engine_screen.cpp:276-291` | Doublon de **A25-07**, qui le décrit déjà (renvoi 19 → 25) avec la même correction. Précision sur la portée : le double lancement n'arrive qu'à la fin de l'assistant, quand l'inscription n'est pas encore faite (`!engines.open_with_registered`, `engine_screen.cpp:276`) et que « lecteur par défaut » est coché (`:286`) : `register_player` appelle `refresh_desktop_database` (`default_player.cpp:300`, 15 s au plus), puis `choose_as_default` rappelle `register_player` (`:335`), qui réécrit le même `.desktop` et relance l'outil, avant `xdg-mime default` (30 s au plus). Depuis le panneau (`settings_system.cpp:53`), le bouton « Make this the default player » ne lance l'outil qu'une fois, et dans une tâche à part (`start_task`) : pas de gel. Le coût réel est donc celui de A25-07 (fenêtre figée sur le thread de l'assistant), la répétition ne faisant que l'allonger. La correction de A25-07 (passer par `run_pumped`, ne pas réinscrire si l'inscription vient d'être faite) couvre les deux. |
| de 34 : canal nommé à nom prévisible | `src/app/play/single_instance.cpp:76-83`, `:111-115`, `:213-217`, `:178-205` | Doublon de **A34-03** et du défaut connu **SEC-15** (revérifié « toujours présent » par le rapport 19). Le nom est `\\.\pipe\ajp-` suivi du FNV-1a 64 bits du chemin UTF-8 du dossier de réglages (`instance_key`, `:76-83`), calculable par tout autre compte. `CreateNamedPipeW` reçoit `nullptr` comme descripteur de sécurité (`:217`), donc l'ACL par défaut ; `PIPE_REJECT_REMOTE_CLIENTS` n'écarte que le réseau. `send_files` (`:178-205`) n'identifie pas le serveur, et un « ok » suffit pour que le lancement se ferme (`:227`). La correction de A34-03 (SID et session dans le nom, ACL réservée à l'utilisateur, `GetNamedPipeServerProcessId` puis contrôle du jeton côté client) couvre le renvoi. Linux n'est pas touché : socket dans le dossier de l'utilisateur, en `0600`. |

## Points positifs

- Le bouton du panneau lance l'inscription et le choix du lecteur par défaut hors du thread de la fenêtre
  (`settings_system.cpp:46-58`), avec un bouton désactivé tant que la tâche tourne.
- `update-desktop-database` et `xdg-mime` sont lancés par chemin absolu, avec un délai borné.
