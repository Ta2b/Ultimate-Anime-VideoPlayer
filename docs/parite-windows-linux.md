# Banc de parité Windows contre Linux (P7.10)

But : sous Windows, au moins les mêmes performances et la même stabilité que sous Linux, mesurées sur la même
machine. Le banc demande le PC de Tatoub en double démarrage (RTX 40, puis Radeon) et ne tourne pas en CI tant
qu'aucun runner GPU n'existe (P7.3).

## Préparer

- Le même PC, les deux systèmes à jour, pilotes de la même branche (NVIDIA : même numéro majeur ; AMD : même
  version d'Adrenalin et de Mesa à date égale).
- Le binaire livré de la même version sur les deux systèmes : `linux-release` et `windows-release`, jamais un
  Debug.
- Les mêmes extraits : 1080p, 1440p et 4K, en 23,976 et 29,97 i/s, dans un `extraits.tsv`
  (`tools/bench/extraits.exemple.tsv`), au même chemin relatif.
- Écran à la même fréquence, rien d'autre ne tourne, secteur branché.

## Jouer

Recopier le même `play.toml` sur les deux systèmes (même chaîne, même niveau), puis, sur chacun, la même
commande (Git Bash sous Windows), sans option propre à A ou à B :

```sh
tools/bench/banc.sh --a="" --b="" --extraits=extraits.tsv \
  --sortie=parite-<système>-<carte> --tours=3 --secondes=120
```

Puis, sur un des deux systèmes, avec les deux dossiers :

```sh
python3 -I tools/bench/parite.py parite-linux-rtx parite-windows-rtx
```

## Seuils

Par extrait, Windows passe si :

| Mesure | Seuil |
|---|---|
| 99e centile du temps par image | au plus 5 % au-dessus de Linux |
| Images perdues (sautées et manquées) | pas une de plus que Linux |
| Temps médian, rendu, acquisition | au plus 10 % au-dessus de Linux |

`parite.py` sort en erreur dès qu'un seuil est franchi. Le résultat, tenu ou non, va dans `docs/reports/` avec
les deux dossiers résumés.

## Stabilité, à la main

- 8 h de lecture en boucle sur chaque système (`tools/bench/endurance.py`), sans plantage, gel ni croissance
  de la mémoire.
- Sous Windows seulement : HDR coupé puis rallumé, fenêtre déplacée d'un écran à l'autre, veille puis reprise,
  TDR provoqué (`dxcap -forcetdr`). Le lecteur doit reprendre seul à chaque fois.
