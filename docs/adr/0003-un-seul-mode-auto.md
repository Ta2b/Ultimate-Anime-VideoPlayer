# 0003 — Un seul mode Auto : Rapide par défaut, Stable au choix

## Contexte
Deux modes d'Auto coexistaient : Stable (paliers de multiplicateur, `core/auto_rate.h`) par défaut, et Rapide
(chaque image calculée à l'instant exact où elle sera vue, `core/free_grid.h`), marqué expérimental. Mesuré sur la
scène lourde : Stable 43 à 51 i/s, Rapide 61 à 67 i/s, pertes comparables.

## Décision
Rapide devient l'Auto par défaut ; Stable reste au choix dans le menu. Là où Rapide ne tient pas (écran sans VRR
armé ni fréquence connue, moteur qui ne calcule pas à un instant libre), Stable le remplace sans message ; s'il
perd trop d'images, la lecture repasse sur Stable avec un message. Les profils déjà enregistrés gardent leur mode.
Le choix de la chaîne au démarrage lit le banc mesuré de la machine quand il existe (`effective_tier`), la
table des cartes sinon.

## Conséquences
- La fluidité la plus haute sans réglage sur un écran VRR.
- Aucun nouveau réglage : le libellé « expérimental » disparaît.
