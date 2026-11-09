# Audit 39. Diagnostic et support

**Audit transverse.** Il traverse les zones au lieu de les découper. Cette fiche se suffit à elle-même : elle donne les points d'entrée dans le code et le numéro de la zone de chacun.

## Pourquoi

Quand un utilisateur ouvre une issue, ce qu'il fournit doit suffire à trouver la cause, sans exposer sa vie privée.

## Méthode

1. Relire ce que le journal contient pour chaque erreur importante.
2. Suivre le parcours « Report a problem » jusqu'à l'issue GitHub.
3. Vérifier les minidumps et les messages d'erreur visibles.

## À couvrir

- Informations manquantes : versions, carte, pilote, réglages.
- Chemins et noms d'utilisateur dans le journal et les minidumps.
- Messages d'erreur utiles à l'utilisateur.

## Points d'entrée dans le code

Relevés au commit audité. Partir de ces lignes et suivre les appels ; si une ligne a bougé, chercher le nom indiqué.

### Ce qui part avec une issue

| Point | Où | Zone |
|---|---|---|
| bouton « Report a problem » | `src/app/play/settings_system.cpp:497` | [23](23-reglages-panneau.md) |
| modèle de rapport de bug | `.github/ISSUE_TEMPLATE/bug.yml` | [28](28-ci-publication.md) |
| journal | `src/core/log.cpp:82` | [20](20-socle-core.md) |
| masquage des chemins personnels | `src/core/log_redact.cpp:36` | [20](20-socle-core.md) |
| nom du fichier journal | `src/app/main.cpp:420` | [18](18-demarrage-arret.md) |
| minidumps | `src/app/play/crash_dump.cpp:80` | [19](19-integration-systeme.md) |
| ligne de télémétrie | `src/app/play/gpu_telemetry.cpp:63` | [11](11-ecran-materiel.md) |

## Défauts déjà connus de ce type

Relevés par le [contre-audit du 08/11](https://claude.ai/artifact/4rfGFb3P7zNiyHiwkfcnUU). Les revérifier, et chercher les autres.

| ID | Gravité | Défaut |
|---|---|---|
| UX-13 | Majeur | « Report a problem » promet un journal sans dossiers, mais les chemins complets hors du dossier personnel y restent (voir WIN-08). |
| WIN-07 | Majeur | Les minidumps sont sans PDB et ne voient ni `__fastfail` ni débordement de pile. Ils gardent les chemins en clair, et Linux n'a rien. |
| UX-17 | Mineur | Le message d'erreur se répète (« ne se lit pas : le fichier ne se lit pas »). La boîte fatale renvoie deux fois au journal. |
| WIN-08 | Mineur | Le nettoyage du journal est sensible à la casse et ignore les noms courts. Le nom d'utilisateur fuit, et `/home/al` abîme `/home/alice`. |

## Zones traversées

Les fiches de ces zones donnent la liste complète de leurs fichiers et leurs défauts connus. Les lire pour le contexte, sans refaire leur audit : un défaut local trouvé en chemin va dans « Renvois ».

- [11. Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md)
- [18. Démarrage et arrêt](18-demarrage-arret.md)
- [19. Intégration au système](19-integration-systeme.md)
- [20. Socle commun (core)](20-socle-core.md)
- [23. Réglages : le panneau](23-reglages-panneau.md)
- [28. CI, publication et signature](28-ci-publication.md)
