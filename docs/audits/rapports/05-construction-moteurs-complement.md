# Rapport de l'audit 05, complément. Renvois reçus des autres audits

Branche : `claude/project-thread-kp376e`. Commit audité : `94c6bc7`. Date : `2026-11-09`.

## Verdict

Sur six renvois, trois sont confirmés comme défauts nouveaux, tous mineurs (A05-09 à A05-11), un est réfuté
(formats des frontières de DistilDRBA) et deux sont des doublons (A20-02 pour les `path::string()`, A27-02 pour le
`RUNPATH` de `trtexec`). Aucun ne touche la qualité d'image.

## Cadrage retenu

Celui du rapport 05. Ce complément ne vérifie que les six renvois de la liste, dans le code, sans carte NVIDIA ni
Windows : ce qui demande l'une ou l'autre est marqué « à confirmer », avec la manipulation.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A05-09 | Mineur | `src/prepare/builder.cpp:174`, `src/prepare/model_source.cpp:76-79` et `:91`, `src/prepare/identity.cpp:153-156` | Chaque appel de `build()`, même quand le moteur est en cache, relit tout l'ONNX et le hache deux fois : SHA-256 du fichier (`model_source.cpp:77-79`), puis la clé de contenu (`onnx_source_key`) qui rehache les mêmes octets. L'appelant vient pourtant de calculer ce SHA par `sha256_file_known`, mémorisé par taille et date (`player_engines.cpp:130`, `tensorrt_rife_fused.cpp:51`, `tensorrt_drba_fused.cpp:49`). Après une vraie construction, `:244` refait les trois étapes pour vérifier que l'ONNX n'a pas changé. | Ouverture d'une vidéo avec SPAN, RIFE fusionné (six morceaux) et DistilDRBA fusionné (quatre morceaux), tous en cache : onze lectures et vingt-deux hachages de fichiers déjà connus, sur le fil de préparation pendant que la fenêtre attend. Avec les instructions SHA (Zen, Intel depuis la 11e génération), quelques dizaines de millisecondes en tout ; sans elles (Intel de 6e à 10e génération, nombreux avec une RTX), de l'ordre de 300 Mo/s, soit 0,1 à 0,3 s ajoutées à chaque ouverture (estimé, non mesuré). | Ne lire l'ONNX que si la clé n'est pas trouvée : dériver la clé de contenu du SHA du fichier déjà connu (le schéma `kOnnxSourceScheme` change, donc une seule reconstruction) ou mémoriser `ModelSource` par chemin, taille et date comme `sha256_file_known`. Après la construction, comparer taille et date au lieu de rehacher. Gain attendu : à mesurer par la ligne « moteur … prêt en … s » du journal avec un cache plein. | Lu. |
| A05-10 | Mineur | `src/app/play/engine_build.cpp:28-31` | `~EngineBuild` demande l'arrêt puis joint le fil de construction sur le thread de la fenêtre, sans pomper les événements. Seuls l'attente des verrous (`key_lock.cpp:88`) et `trtexec` (`process_posix.cpp:196`, `process_win.cpp:172`) regardent le jeton d'arrêt. La création du backend (`engine_build.cpp:38`), `read_model` et la désérialisation de contrôle (`check_`, `builder.cpp:250`) vont à leur terme. Même en cas de succès, `finish()` publie le résultat avant que le `TensorRtBackend` local soit détruit (`:53`) : `wait_engine` rend la main et la destruction attend ce démontage. | L'utilisateur appuie sur « Annuler et regarder » ou sur `q` juste quand `trtexec` vient de finir : le lecteur relit l'ONNX puis désérialise le moteur publié avant de pouvoir joindre. La fenêtre reste figée sans dessin pendant ce temps (environ une seconde, plus pour un gros moteur ; à confirmer sur la 4070 Ti en journalisant la durée de la jointure). Cas voisin : annulation pendant la création du contexte CUDA d'un backend neuf (A08-10). | Pomper avant de détruire : tant que le fil n'a pas fini, `SDL_WaitEventTimeout` et redessiner, comme `wait_engine` ; ou confier l'objet à un ramasseur qui le joint hors du thread de la fenêtre (à la fermeture au plus tard). Détruire le backend avant `finish()`. Corriger A08-10 (un backend partagé) retire la plus grande part du coût. | Lu. |
| A05-11 | Mineur | `src/prepare/process_win.cpp:128` et `:151`, `src/prepare/builder.cpp:208-216` | Le lecteur passe à `trtexec.exe` une ligne de commande UTF-16 juste, mais `trtexec` est un programme à `main(int, char**)` sans le manifeste `activeCodePage = UTF-8` : son CRT reconvertit la ligne dans la page ANSI du système, et il ouvre ses fichiers en chemins étroits. Tout caractère hors de cette page devient `?`. Or l'ONNX (dossier des modèles), le moteur, le timing cache et le rapport des couches sont tous sous le profil de l'utilisateur (`%APPDATA%\ajp\ajp\…`). Même cause que A20-02 et A06-02, pour un autre enfant. | Windows 11 en français, compte « Тарас » ou « 太郎 » : `trtexec` reçoit `C:\Users\?????\AppData\…\model.onnx`, ne trouve pas le fichier, et chaque construction finit en `BuilderFailed` ; TensorRT n'est jamais utilisable sur ce poste. Les accents français (`é`, dans la page 1252) ne sont pas touchés. | Ne pas modifier `trtexec.exe` (fichier NVIDIA dont le module garde l'empreinte). Lancer `trtexec` avec `working_dir` sur le dossier de travail du cache et des chemins relatifs ASCII : y lier en dur l'ONNX et ses poids externes (`CreateHardLinkW`, copie si les volumes diffèrent), y écrire moteur, timing cache et rapport, puis publier comme aujourd'hui. À défaut, noms courts 8.3 (`GetShortPathNameW`), qui peuvent être désactivés sur le volume. Refuser tôt, avec un message clair, un chemin que `WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS)` ne sait pas rendre. | Lu (`trtexec.cpp` des sources publiques de TensorRT : `int main(int argc, char** argv)`). À confirmer : compte Windows au nom cyrillique, lancer une construction et lire le `.build.log`. |

## Renvois vérifiés

| De l'audit | Où | Constat reçu | Conclusion |
|---|---|---|---|
| 04 | `src/prepare/builder.cpp:174` | `read_model` relit et rehache tout l'ONNX avant et après chaque construction | **Confirmé : A05-09.** Plus large que le constat : la relecture a lieu aussi quand le moteur est en cache, et le graphe est haché deux fois par lecture. Le hachage après la construction est voulu (point positif du rapport 05) ; seule sa forme coûteuse est en cause. |
| 08 | `src/app/play/engine_build.cpp:28-31` | `~EngineBuild` joint le fil de construction sur le thread de la fenêtre | **Confirmé : A05-10.** Lié à A08-10 (un backend par construction), qui en est la plus grosse part, sans en être un doublon : la jointure sans pompe reste après sa correction. |
| 03 | `src/prepare/trtexec_command.cpp:105-106` | Formats des frontières des morceaux DistilDRBA laissés à TensorRT : reformatages possibles | **Réfuté.** Les lignes 105-106 retirent `--inputIOFormats` et `--outputIOFormats` pour trtexec 11 (`trtexec_command.h:57-59`), mais rien n'est « laissé à TensorRT » : une entrée ou une sortie de réseau est en format linéaire par défaut (`ITensor::setAllowedFormats` vaut `kLINEAR` seul tant qu'on ne l'élargit pas), et son type suit l'ONNX sous `--stronglyTyped` (`:80`), FP16 pour les morceaux. C'est exactement ce que `fp16:chw` demandait : la frontière est FP16 linéaire dans les deux cas, celle qu'écrit et lit la colle. Un reformatage interne vers un format de tenseurs à blocs, au début ou à la fin d'un morceau, reste possible, mais il existait déjà avec l'ancien drapeau. Le journal le dit pour chaque morceau (« N couches dont M reformatages », `builder.cpp:119-120`), et la mesure de son coût est déjà prévue (`docs/tests-pc.md:291-298`, « E/S NHWC ») ; il faut seulement y ajouter les quatre morceaux de DistilDRBA et les six de RIFE fusionné, qui n'y sont pas nommés. Pas de perte de qualité. |
| 20 | `src/prepare/process_win.cpp:63-65` | `trtexec.exe` sans manifeste UTF-8 : chemin hors page ANSI abîmé | **Confirmé : A05-11.** `widen` (`:63-65`) est juste ; le défaut est dans l'enfant. |
| 24 | `src/app/play/player_engines.cpp:135` | `stem().string()` et `path.string()` | **Doublon de A20-02 (dette).** `.id` ne sert qu'aux journaux (`builder.cpp:179-254`), pas à la clé ; le `path.string()` est à `:143`, dans un journal. Sous Windows 11, le manifeste UTF-8 d'`ajp.exe` empêche l'exception (A20-02), et Linux est en UTF-8. Reste la règle de la CI : à passer par `core::to_utf8` avec le reste de la dette. |
| 27 | `src/gpu/cuda/backend/CMakeLists.txt:83` | Même RUNPATH pour `trtexec` | **Doublon de A27-02**, dont la correction nomme déjà cette ligne. Conséquence propre à la zone 05 : un `libnvinfer.so.11` d'une autre version devant `$ORIGIN` dans `LD_LIBRARY_PATH` (hérité tel quel, `process_posix.cpp:181-185`, aucun `extra_env` pour `trtexec`) fait construire un moteur que `check_` refuse ensuite, à chaque ouverture. `--force-rpath` ne casse pas l'empreinte : elle est calculée après le passage de patchelf (`CMakeLists.txt:91`). Alternative sans toucher au binaire : lancer `trtexec` avec `LD_LIBRARY_PATH` vidé dans `ProcessSpec::extra_env`. |

## Points positifs

- Lancement Windows sans perte côté lecteur : ligne de commande construite en UTF-8, élargie en UTF-16 et passée à
  `CreateProcessW`, avec la citation de `CommandLineToArgvW` testée.
- L'empreinte de `trtexec` est calculée sur la copie déjà rangée par patchelf : corriger le `RUNPATH` ne demande
  aucune étape de plus.
- Le nombre de reformatages de chaque moteur est déjà au journal, ce qui rend la question des frontières
  mesurable sans outil.

## Questions ouvertes

Aucune.
