# Complément de l'audit 46. Outils de recherche, modèles et données livrées

Branche : `claude/project-thread-ib55fq`. Commit audité : `9ead2be`. Date : `2026-11-09`.

## Verdict

Le seul renvoi reçu (de l'audit 40) est exact, mais c'est le défaut A40-03 déjà inscrit dans `SYNTHESE.md` : aucun
nouveau défaut A46.

## Renvois traités

| Renvoi | Où | Conclusion | Preuve |
|---|---|---|---|
| de 40 : « Le banc de qualité refuse DistilDRBA » | `tools/bench/rife_triplets.py:96-100` | **Doublon d'A40-03** (`SYNTHESE.md:647`, Majeur). Le constat tient : `run_model` ne lit que la première entrée du modèle (`sess.get_inputs()[0]`, ligne 98) et s'arrête si elle n'a pas 11 plans (ligne 100). DistilDRBA a deux entrées de 7 plans, `pair01` et `pair12` (`src/app/play/upscale_drba.cpp:7`, `:50` ; `src/prepare/drba_split.cpp:121-123`) : le banc sort donc sur `SystemExit` dès le premier modèle DistilDRBA, et ne mesure jamais sa qualité contre la vraie image du milieu ni contre RIFE 4.22 lite et 4.25. La correction d'A40-03 suffit : une branche DistilDRBA qui remplit `pair01` et `pair12` comme `upscale_drba.cpp:193-211` (trois images, temps au plan 6) et juge l'image produite contre une vraie image tenue à l'écart des entrées, sur les mêmes triplets que RIFE. | Lu. |

## Qualité d'image contre vitesse

Rien à signaler dans ce renvoi : le banc ne dégrade pas d'image, il ne mesure pas. Son absence de mesure laisse en
revanche sans chiffre le choix de DistilDRBA « plus rapide à qualité proche », déjà porté par A40-03, A32-03 et
A49-14.

## Nouveaux défauts

Aucun.
