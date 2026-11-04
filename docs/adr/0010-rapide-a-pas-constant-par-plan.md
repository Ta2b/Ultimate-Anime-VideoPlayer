# 0010 — Rapide à pas constant par plan

## Contexte
Le mode Rapide (ADR 0003, `docs/auto-libre.md`) place chaque instant de sortie là où le GPU sera prêt
(`core/free_grid.h`). Le débit suit le GPU au plus près, mais les intervalles entre images varient d'une paire à
l'autre : σ ≈ 2 ms en simulation à 159 i/s. L'œil voit ce rythme irrégulier sur un panoramique lent. Les pertes
se concentrent aussi au moment des coupes vers un plan plus lourd. Un modèle de coût commun à tous les plans y
perd 14 à 16 % des images.

## Décision
1. **N constant dans un plan.** On sort N images par image source (N ≥ 1, par demi-pas en VRR). À fréquence
   fixe, N est un facteur qui cale l'écran (`lock_factor(hz / (k·fps))`). La grille est uniforme et exacte
   (`core/steady_grid.h`, sur `CadenceAccumulator`) : en simulation, σ = 0,47 µs sur une heure, sans dérive.
2. **N choisi par un modèle de coût par plan** (`core/steady_rate.h`). On prend le plus grand candidat tel que
   (t(N) + d·N)·(1 + 0,15) ≤ P, où t(n) = a + b·n est le coût GPU d'une paire et d le coût d'affichage d'une
   image.
   - À chaque coupe, N repart de l'a priori multiplié par l'échelle mesurée sur les plans précédents.
   - Un seul réajustement par plan : après 24 paires, ou dès les premières paires si elles dépassent le budget.
3. **Urgence et repli.**
   - Urgence : 2 pertes en 1 s font descendre N d'un demi-pas, et N reste gelé jusqu'à la coupe suivante.
   - Repli vers Stable : seulement à N = 1, avec 5 % de pertes sur 5 s. Il vaut pour la séance seulement.
4. **Une image à la fois.** Juste avant son calcul, chaque instant est comparé à l'horloge prédite
   (`Fate::Late`). Un instant passé n'est ni calculé ni écrit.
5. **Derrière `--fast-steady`**, éteint par défaut. Le Rapide libre reste le défaut jusqu'aux mesures du
   chantier 7.10 :
   - cinq extraits, en A/B/A/B × 3 ;
   - marge tranchée entre 0,15 et 0,10.

## Conséquences
- `ChainStats::output_fps()` vaut fps × N. La synchro de l'écran et le compte des images manquées marchent
  comme en Stable.
- Le débit moyen est un peu sous celui du Rapide libre : 47,95 / 35,96 / 95,6 i/s contre 59,7 / 48,0 /
  107,7 i/s au banc T9. C'est le prix d'une marge et d'un N tenu, en échange de 0 perte au banc.
- Sans a priori (banc ou profil pas encore branché), N part de 1. Après 4 paires, une droite prudente sert
  d'amorce : a = 0, b = coût moyen par image.
- Le coût d'affichage d reste à 0 tant qu'il n'est pas mesuré à l'écran (chantier 7.10).
- Smooth (chantier 7.9) est une option à part, éteinte par défaut.
