# 0001 — Retrait de RTX HDR et de Maxine

## Contexte
RTX HDR passait par le SDK Maxine de NVIDIA : un import d'environ 135 Mo, Linux seulement (l'ouverture échouait
sous Windows et retombait sur le HDR adaptatif), un rendu que le lecteur ne contrôle pas et deux chaînes HDR à
tenir à parité. Le look voulu est celui du lecteur, jamais calé sur un master de studio.

## Décision
Le lecteur ne propose plus RTX HDR ni Maxine. Le HDR vient de son grade (`src/render`), le même sur tous les GPU
et tous les systèmes.

## Conséquences
- Une seule chaîne HDR à régler et à tester ; plus aucun fichier NVIDIA téléchargé pour le HDR.
- Les réglages et profils qui demandaient RTX HDR passent au HDR adaptatif.
- Le grade porte seul la qualité HDR : son analyse (0004) et son modèle d'écran (`render/display_model.h`) en
  découlent.
