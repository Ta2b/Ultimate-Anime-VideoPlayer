# Support : où sont le journal et les plantages

Le bouton « Signaler un problème » (Général > Application) copie un rapport nettoyé (chemins réduits au nom du
fichier, dossier personnel remplacé) : version, système, carte, pilote, moteur, **réglages changés** par rapport au
défaut, dernière séance et fin du journal. Quand le lecteur ne s'ouvre plus, les fichiers se prennent à la main.

| Système | Dossier des données (`kirari.log`, `kirari.log.1`, `crashes/`) |
|---|---|
| Windows | `%LOCALAPPDATA%\kirari\` |
| Linux | `~/.local/share/kirari/` |

- `kirari.log` : la séance en cours ; `kirari.log.1` : celle d'avant (la rotation se fait à chaque lancement).
- `crashes/` : les cinq derniers minidumps (Windows). Un minidump peut contenir de la mémoire du processus : ne
  le déposez pas dans une issue publique ; le mainteneur le demandera en privé.
- Linux : une pile (module + décalage) est écrite dans `kirari.log` ; elle se résout avec le build-id donné en tête
  du rapport.
- Plus de détails au journal : non disponible pour l'instant (le niveau est `info`).
