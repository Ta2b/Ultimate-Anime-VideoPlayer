# Support : où sont le journal et les plantages

Le bouton « Signaler un problème » (Général > À propos) copie un rapport nettoyé (chemins réduits au nom du
fichier, dossier personnel remplacé) : version, système, carte, pilote, moteur, **réglages changés** par rapport au
défaut, dernière séance et fin du journal. Quand le lecteur ne s'ouvre plus, les fichiers se prennent à la main.

| Système | Dossier des données (`ajp.log`, `ajp.log.1`, `crashes/`) |
|---|---|
| Windows | `%LOCALAPPDATA%\ajp\ajp\` |
| Linux | `~/.local/share/ajp/ajp/` |

- `ajp.log` : la séance en cours ; `ajp.log.1` : celle d'avant (la rotation se fait à chaque lancement).
- `crashes/` : les cinq derniers minidumps (Windows). Un minidump peut contenir de la mémoire du processus : ne
  le déposez pas dans une issue publique ; le mainteneur le demandera en privé.
- Linux : une pile (module + décalage) est écrite dans `ajp.log` ; elle se résout avec le build-id donné en tête
  du rapport.
- Plus de détails au journal : non disponible pour l'instant (le niveau est `info`).
