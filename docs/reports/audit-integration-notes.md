# Passe d'intégration de l'audit : notes

Branche `claude/project-thread-ta5cur`, vers `cloud/vulkan`. Suite complète : 1180 tests, 0 échec, sous Vulkan logiciel (lavapipe). Seuls les tests qui exigent un vrai GPU, un SDK fabricant ou Windows/macOS sont ignorés (« Skipped »).

## Traité (sans test matériel)
- **Lecteur et session** : Pipeline, Session, wiring, bridges, chaîne, `player_run`, écran d'accueil, réglages, options, hôte, étage d'agrandissement, ui, installation des moteurs. Mise en pause conservée à l'application des réglages, retour à l'accueil sur échec, persistance PL-04 déplacée après `session.begin`, `unsaved` des moteurs, `text_fit.h` (troncature des textes de l'interface).
- **Raccourcis (UI-01)** : Ctrl+G (grade), Ctrl+I (interpolation) et Ctrl+0 (source telle quelle) agissent, en direct si la chaîne le permet, sinon par réouverture. `.`/`,` (image par image) affichent « pas encore disponible » au lieu de ne rien faire.
- **Médias** : choix des pistes (sous-titres lisibles, dialogues complets, titres signs/songs/SDH), reprise de position après erreur de lecture, couleur de la source lue depuis les paramètres du codec (RD-01/RD-19) avec approximations signalées, HDR10 invalide ignoré, rapport d'aspect des pixels (SAR) transmis au rendu et aux sous-titres ASS, SRT : cues ignorées journalisées.
- **Export** : fichier écrit en `.partial` puis renommé seulement à la fin réussie (un fichier existant n'est jamais détruit par un export refusé ou annulé), erreur de copie des pistes remontée dans l'avis, fermeture du fichier vérifiée.
- **IR / noyaux / runtimes** : formes, types ONNX, unshuffle, descriptions, GMFSS (splat, référence, spec de rééchantillonnage), CUDA/TensorRT, autotune et noyaux Vulkan (VK-05, VK-08), MIGraphX (fermeture de descripteur), OpenVINO, WinML, Core ML.
- **Préparation** : refus des liens symboliques qui sortent du dossier du modèle, vérification du journal.
- **Dépendances** : libcurl retirée (vcpkg, CMake, versions.env).

## Non traité, et pourquoi
- **Décisions humaines** : DS-01 à 05, 08, 13, 16, 17 (licence, poids redistribuables, signatures).
- **Fonctions ou idées nouvelles** : NM-05/Q-27 (sous-titres externes), NM-06, NM-09/10, MD-03, MD-04 au-delà du SAR, RD-14, RD-12 (métadonnées de plage et HDR à l'export : demande le chemin de conversion), NE-01, pas à pas image par image.
- **Exige un GPU ou une mesure** : IR-05 (PQ avant le réseau : décision produit), IR-12, VK-06, VK-09 à 11, RT-02/05/06, RD-04 à 09, MD-14/15/16 (politique de décodeur : flux abîmés, hwaccel), PL-12, PL-17.
- **Décision à prendre avant** : IR-04 (heuristique de hauteur des noyaux contre la règle de l'app), DS-14 (`--disable-autodetect` de ffmpeg : risque pour la version livrée), DS-11 (dossier système Windows : invérifiable ici).
- **Reste de l'audit, non entrepris** (faible valeur ou risque sans test) : UI-02, UI-07/08/10/12/17, PL-07 à 09, PL-14/15/18, DS-20, MD-08/11/18, NE-02 à 08, VK-01 à 04, VK-07, VK-12 à 16, RT-03/04/07 à 11, IR-01/06/07/08/11/15, DS-06/07/09/10/18/21/23, NC-02 à 08, NA-03/04/07/08, NM-01 à 04.

## Tests à faire sur du vrai matériel
- Compiler et lancer les builds CUDA/TensorRT, HIP/MIGraphX, OpenVINO, WinML et Core ML : aucun n'a pu être compilé dans le cloud.
- Windows, macOS, NVIDIA, AMD, Intel : lecture, changement de réglages, retour à l'accueil sur échec.
- Écran HDR : sortie PQ ; source PQ/HLG ; source BT.601 ; rip DVD en SAR non carré (vidéo et sous-titres ASS).
- Export : exporter vers un fichier existant puis annuler (l'ancien doit rester intact) ; export d'un fichier dont l'audio est illisible (avis de troncature).
- Appliquer des réglages en pause (la pause doit rester) ; glisser-déposer d'un `.srt`.
- Raccourcis Ctrl+G, Ctrl+I, Ctrl+0 avec et sans RIFE chargé.
- Autotune Vulkan (VK-05/08) sur RDNA3/4, Intel Arc et NVIDIA : sortie identique à la référence.
- Modèle avec lien symbolique sortant du dossier : refusé.

## Deuxième passe (sans test matériel, suite de suites verte : 1180 tests, lavapipe)
- **Synchronisation Vulkan** : détruire ou libérer une ressource d'une file bloquée la laisse fuir au lieu de la détruire (`VkContext::stalled()`) ; propriété de file des images du ring (`EXTERNAL` ou `IGNORED` selon placebo) ; mémoire hôte importée cohérente seulement.
- **Runtimes différés** : trois échecs consécutifs d'une étape hôte arrêtent la chaîne au lieu de répéter l'échec.
- **IR** : poids et biais F16 fusionnés arrondis à la demi-précision.
- **Export** : durée des images posée (NE-06) ; pas de lecture de tout le fichier source sans piste à copier (NE-07).
- **Audio** : amorce, remplissage et changements d'extradata des paquets atteignent le décodeur (MD-11).
- **Lecture** : la fin n'est déclarée que lorsque le thread gpu est au repos, en plus de la grâce d'une seconde (PL-09).
- **Release** : tests retirés du workflow Linux (lancés sur le poste et par `ci.yml`).
- À vérifier sur matériel : fin de lecture avec un moteur lent à la première exécution ; AAC/MP4 et Opus/MKV (pas d'amorce audible au début) ; export dont la durée d'image compte (lecteurs stricts) ; reprise après un GPU bloqué.
- **Vulkan** : autotune.tsv fusionné à l'écriture (VK-15) ; groupes de dispatch coopmat bornés à la construction (VK-07) ; cache de pipelines du pilote enregistré dans `cache/vulkan/pipelines-<vendeur>-<appareil>.bin` (VK-09). VK-08 (pente PRelu) était déjà faite.
- **IR** : pliage coûteux (sortie × entrée > 2^32) laissé à l'exécution (IR-01).
- **CUDA** : purge du cache de textures avant le lancement (NC-02) ; à compiler sur un poste CUDA.
- **Interface** : infobulles en pause (UI-17), libellés et avertissement du panneau (UI-10), actions de l'onglet Système sur un autre fil (UI-07).
- Déjà bornés avant cette passe : PL-10 (kNoPts), PL-11 (fenêtre de 4096 attentes).
- À vérifier sur matériel en plus : second lancement sans compilation de pipelines ; infobulle de la barre en pause ; « Définir par défaut » sans geler la vidéo ; `lecture` d'un modèle hostile (pliage).
- **Suite** : UI-02 (`[keys]` de user.toml appliquée aux écrans et à la lecture ; une `[keys]` invalide ramène aux défauts avec un avertissement, mais fait encore refuser le profil entier par `parse_layered`) ; RD-06 (lissage de l'APL qui n'avance qu'avec une nouvelle image vidéo : `2-analysis.hook`, variable `prev_nits`) ; NC-03 (levels 219/255 selon la plage ingérée) ; NA-08 (modèle dérivé sous nom propre, fichier vide refait) ; NE-08 (dossier de destination vérifié avant les moteurs) ; MD-08 (`interrupt_callback` + `rw_timeout` : la fermeture n'attend plus un partage muet) ; PL-15 (SHA-256 des téléchargements interruptible) ; PL-18 (la demande de sous-titres la plus récente attend sa place).
- Déjà faits avant cette passe, constatés à la relecture : MD-09, PL-10, PL-11, PL-16, RD-16, VK-08, NE-08 (troncature UTF-8).
- À vérifier sur matériel en plus : fermeture avec un fichier sur partage réseau coupé ; raccourcis redéfinis dans user.toml (lecture, accueil, écran des moteurs) ; grade HDR adaptatif (stable en pause, lissage normal en lecture) ; téléchargement d'un moteur annulé pendant la vérification (le .part est gardé) ; sous-titres animés à forte cadence.
- Toujours non entrepris : UI-08/12, PL-07/08/14, MD-18, NE-02 à 05, VK-01/16, RT-03/04/07 à 11, IR-06/07/08/11/15, DS-06/07/09/10/18/21/23, NC-04 à 08, NA-03/04/07, NM-01 à 04, RD-16 (fait). Ils demandent soit une décision de conception (PL-07 : code mort du profil, PL-08 : écran changé en cours de lecture, UI-08 : code mort), soit du matériel (RT-04, RT-08/09, NC-04 à 08), soit une refonte (VK-01, NE-05).
