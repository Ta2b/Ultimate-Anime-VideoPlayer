# À tester avant d'étiqueter la v0.5.0 : Windows et macOS

Tout ce qui suit n'a **pas** été essayé : le développement s'est fait sur Linux (KDE Wayland, RTX 4070 Ti). Les tests unitaires
passent sur Linux (`ajp_app_tests`, `ajp_core_tests`, `ajp_audio_tests`) ; la CI Windows et macOS les rejoue sans GPU.

## Windows (poste NVIDIA, puis un poste AMD/Intel si possible)
1. **Installation propre** : dézipper `ajp-windows-x64.zip`, lancer `ajp.exe` : l'assistant s'ouvre (8 écrans, ou 6 sans écran
   moteur ni HDR selon la machine), « Terminer » inscrit AJP dans « Ouvrir avec » (clés HKCU). Vérifier : clic droit sur un .mkv >
   Ouvrir avec > AJP. « Faire de ce lecteur le lecteur par défaut » ouvre la page Paramètres > Applications par défaut.
2. **Évaluer ma machine** (Général > Configuration) : plusieurs processus `ajp.exe` courts, plein écran, son coupé ; le résultat
   arrive dans `bench.toml` du dossier de données. Vérifier qu'un double clic sur le bouton ne lance pas deux évaluations.
3. **Audio** : la liste des appareils (WASAPI via SDL) ; changer d'appareil rouvre la vidéo ; Casque / Enceintes / Home cinéma
   avec un casque et un ampli 5.1 (Home cinéma grisé sur un appareil à 2 canaux) ; touche B ; brancher/débrancher le casque
   pendant la lecture (l'appareil par défaut doit être suivi par SDL).
4. **Mise à jour** : une release de test signée (`docs/signature-des-releases.md`) plus récente que l'installée ; Général >
   Mises à jour > Vérifier maintenant ; en mode Automatique, l'archive se prépare, **l'application au lancement suivant** doit
   renommer `ajp.exe` en `ajp.exe.old` puis relancer le lecteur (non essayé : c'est le point le plus risqué) ; « Revenir à la
   version précédente ».
5. **TensorRT** : `docs/nvidia-windows.md` (module + `trtexec.exe`, bibliothèques NVIDIA tirées par plages, SHA-256), puis une vidéo
   ×3 avec SPAN et RIFE, et RTX HDR (automatique sous Windows).
6. **Menu** : parcourir toutes les pages à 100 %, 150 % et 200 % d'échelle Windows : rien ne doit déborder (des captures
   ont été relues sur Linux à 100 % seulement).
7. **PowerShell** : `tools/release/package-windows.ps1` avec `AJP_ENGINES_LOCK` et `attach-nvidia-windows.ps1` n'ont pas été exécutés
   (pas de pwsh sur la machine de développement).

## macOS (Apple Silicon)
1. Lancement depuis `AJP.app` : l'assistant (pas d'écran moteur : Core ML), inscription « Ouvrir avec » par l'Info.plist (rien à écrire),
   **le bouton « lecteur par défaut » ne fait que dire comment faire** (Finder > Lire les informations).
2. **Audio** : la liste des appareils (CoreAudio), casque / enceintes, touche B.
3. **Mises à jour** : sur macOS le réglage est grisé (« la nouvelle version se télécharge sur la page des releases ») : vérifier le
   texte, et que « Vérifier maintenant » dit bien la version disponible.
4. **Évaluer ma machine** : même mécanique que sous Linux, mais le chemin de l'exécutable dans `AJP.app/Contents/MacOS`.
5. L'interpolation d'Apple et les pastilles du menu.

## Linux, à faire aussi
- Écoute du **binaural** (casque) et du **crossfeed** : aucun casque n'était branché pendant le développement.
- Les écrans 2 à 8 de l'assistant et les pages HDR, Couleurs, Mises à jour du menu n'ont pas été relus en capture.
- Une release signée de bout en bout (`sign-manifest.sh` puis « Vérifier maintenant » depuis une version plus ancienne).
