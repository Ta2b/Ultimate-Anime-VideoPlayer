# Sécurité

*English: please report vulnerabilities privately through
[GitHub's private reporting](https://github.com/Ta2b/Ultimate-Anime-VideoPlayer/security/advisories/new),
never in a public issue. French or English are both fine.*

## Signaler une faille

Par le [signalement privé de GitHub](https://github.com/Ta2b/Ultimate-Anime-VideoPlayer/security/advisories/new),
jamais dans une issue publique. Décrire la version (menu Général, page Application), le système, et de quoi reproduire.

Réponse visée sous 7 jours, correctif ou contournement sous 30 jours pour une faille confirmée, puis
publication de l'avis avec le correctif.

## Ce qui compte surtout

- la mise à jour automatique : manifeste signé (Ed25519), téléchargement, installation, retour arrière ;
- le téléchargement des modèles et des moteurs (`engines.lock`, empreintes SHA-256) ;
- la lecture de fichiers non sûrs : conteneurs et flux (FFmpeg), sous-titres (libass, SRT), LUT `.cube`,
  modèles ONNX, fichiers de réglages ;
- les bibliothèques chargées à l'exécution (moteurs, pilotes).

## Versions suivies

Seule la dernière version publiée reçoit les correctifs.
