# Audits

## Critère de cible

La cible est un lecteur parfait sur carte NVIDIA sous TensorRT, sous **Windows 11** et sous **CachyOS**. Un audit
est dans la cible s'il touche, même indirectement, ce que vit un utilisateur de cette cible, qualité du code
comprise. **Hors cible pour l'instant :** macOS et Core ML, l'export vidéo. Ce code reste dans le lecteur tel quel,
ni supprimé ni caché : il fonctionne, ses bugs et son optimisation viendront plus tard. **Réduits :** Vulkan
(seulement son rôle de repli sur une RTX), l'IR ONNX (utile à ce repli), les outils (seulement les données
livrées). En cas de doute, l'agent applique cette règle et le signale à Tatoub.

## Deux sortes d'audits

- **36 audits par zone.** Le dépôt est découpé en zones qui ne se recouvrent pas : chaque fichier du produit
  appartient à une seule zone. Ces audits trouvent les défauts locaux.
- **12 audits transverses.** Ils suivent un parcours, une ressource ou une propriété à travers toutes les zones.
  Ils trouvent les défauts nés entre deux zones, les plus graves du contre-audit (TRT-01, MED-01, UX-01).

Il y a **48 audits en tout, dont 46 dans la cible** (3 réduits). Ils sont numérotés dans l'ordre où
les faire :

- **01 à 07** : le chemin TensorRT, très haute priorité.
- **08 à 28** : le reste du lecteur, haute priorité.
- **29 à 40** : les audits transverses.
- **Ensuite** : la priorité moyenne, puis les audits réduits, puis les deux hors cible.

| # | Audit | Priorité | Fichiers | Lignes | Défauts connus |
|---|---|---|---|---|---|
| 01 | [TensorRT : exécution, voies, anneau et mémoire](01-tensorrt-execution.md) | très haute | 30 | 4 186 | 3 |
| 02 | [SPAN FP8 : noyaux CUTLASS et dérivation](02-span-fp8.md) | très haute | 25 | 3 526 | 2 |
| 03 | [Interpolation fusionnée sous TensorRT (RIFE, DistilDRBA, warp)](03-interpolation-tensorrt.md) | très haute | 18 | 3 452 | 0 |
| 04 | [Analyse et découpe des modèles ONNX](04-onnx-modeles.md) | très haute | 21 | 2 978 | 0 |
| 05 | [Construction des moteurs, cache et trtexec](05-construction-moteurs.md) | très haute | 28 | 3 104 | 2 |
| 06 | [Installation des moteurs et téléchargements](06-installation-moteurs.md) | très haute | 14 | 2 082 | 8 |
| 07 | [Calibration, niveaux et mode Auto](07-calibration-niveaux.md) | très haute | 31 | 4 509 | 5 |
| 08 | [Boucle du lecteur et ouverture de la chaîne](08-boucle-chaine.md) | haute | 15 | 3 117 | 13 |
| 09 | [Étages d'agrandissement et d'interpolation](09-etages-traitement.md) | haute | 30 | 3 407 | 1 |
| 10 | [Rendu et présentation par libplacebo](10-presentation-libplacebo.md) | haute | 18 | 3 438 | 3 |
| 11 | [Écran et matériel : EDID, VRR, HDR du système, télémétrie](11-ecran-materiel.md) | haute | 13 | 2 079 | 8 |
| 12 | [Cadence et horloges](12-cadence-horloges.md) | haute | 27 | 2 522 | 1 |
| 13 | [Session de lecture : seek, épisodes et reprise](13-session-lecture.md) | haute | 17 | 2 512 | 5 |
| 14 | [Sources, démultiplexage et erreurs d'E/S](14-sources-demux.md) | haute | 26 | 2 440 | 4 |
| 15 | [Décodage, matériel, formats et couleurs](15-decodage-formats.md) | haute | 17 | 3 145 | 5 |
| 16 | [Audio](16-audio.md) | haute | 24 | 2 111 | 1 |
| 17 | [Sous-titres](17-sous-titres.md) | haute | 30 | 2 598 | 4 |
| 18 | [Démarrage et arrêt](18-demarrage-arret.md) | haute | 18 | 3 327 | 18 |
| 19 | [Intégration au système](19-integration-systeme.md) | haute | 10 | 1 417 | 6 |
| 20 | [Socle commun (core)](20-socle-core.md) | haute | 33 | 3 336 | 5 |
| 21 | [Frontière des moteurs (interface et ABI)](21-frontiere-moteurs.md) | haute | 13 | 1 606 | 0 |
| 22 | [Réglages : modèle, application et persistance](22-reglages-modele.md) | haute | 15 | 2 767 | 4 |
| 23 | [Réglages : le panneau](23-reglages-panneau.md) | haute | 12 | 2 324 | 10 |
| 24 | [Interface : rendu, contrôles et messages](24-interface-rendu.md) | haute | 31 | 4 092 | 7 |
| 25 | [Assistant de premier lancement](25-assistant.md) | haute | 6 | 1 443 | 7 |
| 26 | [Mise à jour automatique et signature](26-mise-a-jour.md) | haute | 7 | 1 483 | 12 |
| 27 | [Build, dépendances et ports](27-build-dependances.md) | haute | 30 | 2 932 | 5 |
| 28 | [CI, publication et signature](28-ci-publication.md) | haute | 101 | 4 895 | 16 |
| 29 | [Parcours utilisateur de bout en bout](29-parcours-utilisateur.md) | transverse | — | — | 7 |
| 30 | [Concurrence, threads et durée de vie](30-concurrence-duree-de-vie.md) | transverse | — | — | 9 |
| 31 | [États dégradés et reprise sur erreur](31-etats-degrades.md) | transverse | — | — | 8 |
| 32 | [Budget par image sur la cible](32-budget-par-image.md) | transverse | — | — | 4 |
| 33 | [Longue durée, mémoire et ressources](33-longue-duree-ressources.md) | transverse | — | — | 3 |
| 34 | [Parité Windows et CachyOS](34-parite-windows-cachyos.md) | transverse | — | — | 10 |
| 35 | [Entrées non sûres](35-entrees-hostiles.md) | transverse | — | — | 7 |
| 36 | [État persistant et compatibilité entre versions](36-etat-persistant.md) | transverse | — | — | 6 |
| 37 | [Configurations NVIDIA variées](37-configurations-nvidia.md) | transverse | — | — | 5 |
| 38 | [Promesses tenues](38-promesses-tenues.md) | transverse | — | — | 3 |
| 39 | [Diagnostic et support](39-diagnostic-support.md) | transverse | — | — | 4 |
| 40 | [Tests et couverture](40-tests-couverture.md) | transverse | — | — | 3 |
| 41 | [HDR, grade et looks](41-hdr-grade.md) | moyenne | 29 | 3 849 | 5 |
| 42 | [Analyse des plans, coupes et pré-analyse](42-plans-coupes.md) | moyenne | 17 | 1 768 | 3 |
| 43 | [Textes et traductions](43-textes-traductions.md) | moyenne | 13 | 1 735 | 0 |
| 44 | [Représentation intermédiaire et import ONNX](44-ir-onnx.md) | réduit : utile au repli Vulkan | 50 | 8 037 | 0 |
| 45 | [Moteur Vulkan : calcul, noyaux et shaders](45-backend-vulkan.md) | réduit : repli sur une RTX | 217 | 26 562 | 4 |
| 46 | [Outils de recherche, modèles et données livrées](46-outils-donnees.md) | réduit : données livrées | 63 | 6 731 | 1 |
| 47 | [Export vidéo](47-export.md) | hors cible | 24 | 4 060 | 2 |
| 48 | [Moteur Core ML (macOS)](48-coreml-macos.md) | hors cible | 23 | 4 008 | 0 |

`third_party/cutlass` (code tiers recopié) n'est pas audité : son épinglage relève de l'audit 27,
son usage de l'audit 02. L'audit 45 (Vulkan, environ 26 500 lignes) devra être
coupé en trois le jour où il sort de « réduit ». Le compte des défauts connus de l'audit
18 est gonflé : beaucoup de défauts citent `main.cpp` ou `player.cpp`.

Aucun audit n'a besoin d'un autre pour être fait : chaque fiche se suffit à elle-même. L'ordre sert la priorité, et
permet aux audits suivants de profiter des rapports déjà fusionnés. Les audits 47 et 48 ne se lancent pas.

## Fonctionnement

Les audits sont faits par des agents sans mémoire commune. Un **coordinateur** (la conversation principale du projet)
lance un **fil** par audit ; chaque fil tourne sur sa propre branche et part de zéro. Rien ne passe d'un fil à l'autre
sauf par les fichiers de la branche principale, et une seule règle évite les conflits : **chaque fil n'écrit qu'un
fichier, son rapport ; seul le coordinateur écrit `SYNTHESE.md` et fusionne.**

```
docs/audits/
  README.md            ce fichier (lecture seule)
  NN-<nom>.md          les fiches (lecture seule)
  SYNTHESE.md          écrit par le coordinateur seul
  rapports/
    _MODELE.md         le format de rapport
    NN-<nom>.md        écrit par le fil de l'audit NN seul
```

### Le coordinateur

1. **Au départ**, demander à Tatoub :
   - combien de fils à la fois (3 recommandé : chaque fil pose ses questions de cadrage, au-delà elles s'accumulent) ;
   - les audits à sauter ou à avancer.
2. **Lancer les audits par vagues, dans l'ordre des numéros** : 01 à 07, puis 08 à 28, puis 29 à 40,
   puis la suite jusqu'à 46. Chaque fil part de la branche principale à jour, sur la branche
   `audit/NN-<nom>`. Son prompt est celui de la section « Prompt d'un fil », avec NN et le nom remplacés.
3. **Quand un fil a fini**, relire son rapport avant de le fusionner :
   - le format suit `rapports/_MODELE.md` ;
   - chaque défaut cite `fichier:ligne` et un scénario ;
   - le fil n'a modifié aucun autre fichier. Sinon, lui demander de corriger.

   Puis fusionner le seul fichier `rapports/NN-<nom>.md` dans la branche principale.
4. **Mettre à jour `SYNTHESE.md`** après chaque fusion :
   - ajouter ses défauts au tableau ;
   - repérer les doublons : même `fichier:ligne`, ou même cause. L'ID le plus ancien reste, l'autre est marqué
     doublon ;
   - recopier ses renvois dans le tableau des renvois.

   Puis fusionner `SYNTHESE.md`. Un fil lancé ensuite voit tout cela.
5. **Renvois vers un audit déjà fini.** Les garder « en attente ». À la fin de toutes les vagues, lancer un fil court
   par audit concerné : il vérifie seulement ces renvois et écrit `rapports/NN-<nom>-complement.md`.
6. **À la fin**, écrire le bilan en tête de `SYNTHESE.md` :
   - comptes par gravité et par audit ;
   - les dix défauts les plus graves pour la cible ;
   - la liste des points « à confirmer » que Tatoub doit vérifier sur sa machine.

   Puis le présenter à Tatoub.

Le coordinateur ne corrige pas de code et ne refait pas un audit lui-même. Les corrections passent par
`../chantiers/`.

### Le fil d'un audit

1. **Lire**, dans cet ordre :
   - ce README, en entier ;
   - la fiche `NN-<nom>.md` ;
   - dans `SYNTHESE.md`, les renvois adressés à NN et les défauts déjà trouvés ailleurs (pour ne pas en créer de
     doublon) ;
   - les rapports déjà fusionnés des zones citées dans la fiche, s'ils existent. C'est un bonus, pas un prérequis.
2. **Cadrage avec Tatoub, avant de lire le code.**
   - Ce qui compte pour lui dans cet audit, l'usage réel, et ce qui pourrait être retiré ou mis de côté.
   - Pour un audit transverse : les configurations et les parcours à couvrir en priorité.
   - Les questions passent par un outil de questions à choix quand il existe : 4 au plus par tour, l'option
     recommandée en premier. Puis attendre les réponses.
3. **Lecture.**
   - Audit par zone : lire chaque fichier de la fiche en entier, puis les zones voisines seulement là où les appels
     traversent la frontière.
   - Audit transverse : partir des points d'entrée de la fiche et suivre les appels dans toutes les zones traversées.
4. **Axes à couvrir, dans l'ordre :**
   1. exactitude ;
   2. erreurs et états dégradés ;
   3. concurrence et durée de vie ;
   4. ressources ;
   5. Windows contre CachyOS ;
   6. entrées non sûres ;
   7. performance ;
   8. tests ;
   9. dette (complexité, doublons, code mort).
5. **Défauts connus.** Revérifier un par un ceux de la fiche et les renvois reçus : toujours présent, corrigé, ou
   réfuté.
6. **Preuve.**
   - Chaque constat cite `fichier:ligne` et un scénario concret.
   - Le fil tourne sans carte NVIDIA : ce qui demande une mesure ou un GPU est marqué « à confirmer », avec la
     manipulation qui le confirmerait sur la machine de Tatoub.
   - S'il peut compiler et lancer les tests unitaires, il écrit un test qui échoue pour chaque défaut majeur.
     Sinon, il décrit ce test.
7. **Rapport.** Écrire `rapports/NN-<nom>.md` selon `rapports/_MODELE.md`.
   - Les défauts ont pour ID `ANN-xx`.
   - Un défaut hors de la fiche va dans « Renvois », avec le numéro de l'audit concerné. Le fil ne l'audite pas
     lui-même.
8. **Ne modifier aucun autre fichier**, ni code, ni fiche, ni `SYNTHESE.md`. Valider le rapport sur la branche du fil,
   puis prévenir le coordinateur.

### Prompt d'un fil

> Tu fais l'audit NN de `docs/audits/` : `docs/audits/NN-<nom>.md`. Lis d'abord `docs/audits/README.md` en entier et
> suis la section « Le fil d'un audit ». Commence par lire `SYNTHESE.md`, puis pose-moi tes questions de cadrage
> avant de lire le code. Tu n'écris qu'un fichier : `docs/audits/rapports/NN-<nom>.md`. Tu ne modifies rien d'autre.

## Fichiers non rattachés

Aucun : chaque fichier suivi par Git (hors `docs/`, images et `third_party/`) appartient à une zone.
