# Contribuer

*English: issues and pull requests in English are welcome. Read [`CLAUDE.md`](CLAUDE.md) first: it is the
project's hand-over guide (build, tests, rules). Code comments and docs are in French; UI strings are in
English.*

Lire d'abord [`CLAUDE.md`](CLAUDE.md) (construire, tester, où est quoi), puis [`docs/README.md`](docs/README.md).
Une faille de sécurité ne passe pas par une issue : voir [`SECURITY.md`](SECURITY.md).

## Une pull request

- une PR par sujet, un titre qui dit ce que voit l'utilisateur ;
- un test qui échoue sans le changement, quand c'est possible ; pas de test qui en double un autre ;
- `clang-format-20 -i` sur les fichiers touchés avant de pousser.

## Ce que la CI vérifie

Format, taille des fichiers (800 lignes au plus), aucun `getenv`, shaders compilés, `-Werror` sous Linux,
Windows et macOS, clang-tidy sur les fichiers touchés, tests sans GPU (Vulkan logiciel), CUDA compilé sans
carte, actions épinglées par SHA, CodeQL.

## Ce que la CI ne vérifie pas

À faire soi-même, et à dire dans la PR :

- **un vrai GPU** : la CI n'a que lavapipe. Tout changement de chaîne (agrandissement, interpolation, grade)
  se regarde sur une vraie carte, en lecture et en export ;
- **TensorRT** : les tests qui construisent des moteurs ne tournent que sur un poste NVIDIA
  (`linux-prefix-cuda`). Un changement de `src/core`, `src/gpu`, `src/kernels` ou `src/prepare` change
  l'empreinte du module (`tools/release/trt-fingerprint.sh`) ;
- **l'HDR** : sortie PQ sur un écran HDR réel, Windows et KDE ;
- **les performances** : temps par image et images perdues (`--stats=FICHIER`), avant et après ;
- **l'ouverture des anciens fichiers** : `play.toml` et `user.toml` d'une version précédente doivent
  rester lisibles (ajouter un test qui les relit) ;
- **les textes** : un `tr("…")` nouveau a sa traduction dans `src/app/play/i18n_fr_*.cpp` (le test i18n
  le signale), et le français se relit ;
- **les captures** : une PR qui change le menu montre une capture (`--snapshot`, voir `CLAUDE.md`).

## Licence

Une contribution est publiée sous la licence du dépôt ([`LICENSE`](LICENSE)).
