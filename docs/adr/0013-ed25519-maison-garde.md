# 0013 — La vérification Ed25519 maison est gardée, sous vecteurs RFC 8032 et Wycheproof

## Contexte
L'audit 0.9.1 (Tab 1, S4 et A4) relève que `src/app/play/ed25519.cpp` réécrit l'arithmétique de TweetNaCl, que
`sha512()` rendait un condensat nul si `av_sha512_alloc()` échouait (la vérification se réduisait alors à
R == S·B, qu'un attaquant fabrique), et qu'un seul vecteur de la RFC 8032 la testait. Il demande libsodium,
OpenSSL ou Monocypher, ou, si le code maison reste, un échec fermé et les vecteurs RFC 8032 et Wycheproof.

## Décision
Le code maison reste : il ne fait que vérifier (aucune clé privée dans le lecteur), en 340 lignes, sans
dépendance de plus à construire et livrer sur trois plateformes.
- **Échec fermé** : `sha512()` rend `std::nullopt` si libavutil ne donne pas de contexte ou refuse
  l'initialisation, et `ed25519_verify` rend alors false.
- **Vecteurs** : les essais 1 à 3 de la RFC 8032 et les 210 vecteurs Wycheproof (`ed25519_test.json`, recopiés
  dans `tests/unit/app/wycheproof_ed25519.h`) passent : petits ordres, S ≥ L, encodages non canoniques,
  signatures tronquées ou allongées.

## Conséquences
- Toute modification de `ed25519.cpp` doit garder ces vecteurs verts.
- Si une bibliothèque cryptographique entre un jour dans les dépendances (OpenSSL pour le SHA-256, P4.10), la
  vérification passe par elle et ce fichier disparaît.
