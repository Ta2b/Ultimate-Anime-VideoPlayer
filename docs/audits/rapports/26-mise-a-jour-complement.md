# Complément de l'audit 26. Mise à jour automatique et signature

Branche : `claude/project-thread-ig6b3d`. Commit audité : `bdfe44c`. Date : `2026-11-09`.

## Verdict

Des trois renvois en attente, un seul apporte un défaut nouveau (A26-13 : une sauvegarde `updates/previous`
périmée reste proposée au retour arrière après une installation à la main). Les deux autres sont des doublons
d'A30-03 et A26-06 pour l'un, de SEC-05 et A28-02 pour l'autre, dont la partie « sans journal » est réfutée.

## Cadrage retenu

Seuls les trois renvois de `complements/26.md` sont vérifiés, sans question à Tatoub. Aucun ne touche la qualité
d'image : ce code ne fait aucun compromis entre qualité et vitesse.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A26-13 | Mineur | `src/app/play/updater.cpp:646-651`, `:605`, `:616-631`, `src/app/play/settings_system.cpp:222` | `updates/previous` et `previous_version` survivent à un changement de version qui ne passe pas par `apply_staged_update`. `guard_new_version` ne regarde rien quand `unconfirmed_version` est vide (`:646`, cas d'une version déjà confirmée). Quand elle voit qu'une autre version tourne (« installée à la main, ou déjà revenue », `:647-651`), elle efface `unconfirmed_version` mais garde `previous_version` et la copie. Le bouton « Roll back to the previous version » reste actif, et `roll_back_update` ne vérifie pas que la version en cours est celle qui a remplacé `previous`. | 0.10.3 se met à jour seule en 0.10.4 (`previous` = 0.10.3, `.ajp-files` = liste de 0.10.4). Tatoub installe 0.10.5 avec l'installeur Windows, qui ne réécrit pas `.ajp-files` (SEC-26). Plus tard, il clique « Roll back » : `restore_previous` copie 0.10.3 sur 0.10.5 et ne retire que les fichiers de la liste de 0.10.4. Les fichiers ajoutés par 0.10.5 restent : installation mélangée, retour de deux versions au lieu d'une, et « Previous version restored : 0.10.3 ». Si 0.10.5 a réécrit `play.toml` ou le cache des moteurs, la 0.10.3 les relit aussi (renvoi 26→36). | Noter dans `update.toml` la version qui a remplacé `previous` (`replaced_by`, posée à `:605` avec `unconfirmed_version`). Dans `guard_new_version`, avant le retour de `:646`, quand la version en cours n'est ni celle-là ni `previous_version`, retirer `updates/previous` et vider `previous_version`. `roll_back_update` refuse si `installed_version() != replaced_by`. | Lu. Test décrit : `apply_staged_update` vers 0.2, puis `guard_new_version` avec `running = "0.3"` : `previous_version` doit être vide et `updates/previous` absent. |

## Renvois traités

| Renvoi | Conclusion | Preuve |
|---|---|---|
| 30→26 `src/app/play/updater.cpp:466-474` : téléchargement sans `.part` ni verrou | **Doublon** d'A30-03 (verrou de processus et `.part` renommé après vérification, déjà dans sa correction) et d'A26-06 (fichier partiel laissé sur un échec). | `fetch` écrit directement le nom final (`curl -o`, `:90-91`), et seul un échec de `verify_file` efface le fichier (`:470-473`), pas un échec ni une annulation de `fetch` (`:469`). Le fichier partiel n'est pourtant jamais appliqué : `staged_file` n'est posé qu'après vérification (`:528-534`), et `apply_staged_update` revérifie l'empreinte (`:561-566`, sauf SEC-12). Entre deux processus, l'instance unique (`main.cpp:363-365`, `single_instance.cpp`) évite le cas, sauf quand elle n'a pas pu se prendre (`single_instance.cpp:230`, `:411`). Rien de plus qu'A30-03. |
| 28→26 `src/app/play/updater.cpp:410-415` : un 404 du manifeste se lit « à jour » sans journal | **Doublon** de SEC-05 (fiche 26, revérifié « toujours présent » dans le rapport 26) pour la lecture « à jour », et d'A28-02 pour sa conséquence (aucun manifeste publié depuis la 0.5.1). La partie « sans journal » est **réfutée**. | Le fil « updates » journalise le message de toute vérification, 404 compris : `player.cpp:202`, `info("mise à jour : {}", …)`, au niveau `info`, qui est le niveau par défaut (`src/core/log.cpp:31`). `ajp.log` contient donc « No version is published on the "stable" channel yet: the player is up to date ». Le défaut réel est que ce cas est un `info` et non un `warn`, et qu'il vide `latest_version` comme une vraie réponse « à jour » : c'est SEC-05. Correction inchangée : journaliser le 404 en `warn` et le montrer dans la page Application comme « aucun manifeste trouvé », pas « à jour ». |
| 36→26 `src/app/play/updater.cpp:670-679` : `updates/previous` jamais retiré après confirmation | **Réfuté** pour la confirmation, **doublon** pour la place, **confirmé** dans un cas voisin (A26-13). | Garder `previous` après confirmation est voulu : le bouton « Roll back to the previous version » (`settings_system.cpp:222-235`) s'en sert justement après une version confirmée, et `confirm_version` (`:674-679`) n'a pas à le retirer. La copie est unique : chaque application refait `remove_all(previous)` (`:585`), donc pas d'accumulation. Sa place dans le profil itinérant Windows est A26-06 et WIN-03 (`updates` hors de `kHeavy`, `data_dirs.cpp:24`). Son maintien après la désinstallation est SEC-26. Le défaut nouveau est la copie gardée après une version installée à la main : A26-13. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 36 | `src/app/play/updater.cpp:616-631` | Un retour arrière de deux versions (A26-13) rend à une ancienne version un `play.toml` et un cache de moteurs écrits par une version qu'elle n'a jamais précédée directement. |

## Points positifs

- Un fichier partiel ne peut pas être appliqué : la préparation n'est notée qu'après la vérification, et l'archive
  est revérifiée juste avant l'application.
- Une seule copie `previous` à la fois : la place prise est bornée à une installation.

## Questions ouvertes

Aucune.
