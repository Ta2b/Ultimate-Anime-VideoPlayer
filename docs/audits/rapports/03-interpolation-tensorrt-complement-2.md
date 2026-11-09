# Complément 2 de l'audit 03. Renvois reçus après la fin de l'audit

Branche : `claude/project-thread-opy5jn`. Commit audité : `328cc20`. Date : `2026-11-09`.

## Verdict

Les quatre renvois sont confirmés, aucun n'est un doublon : le plus sérieux est la grille de base de RIFE v1,
arrondie en FP16 dans les deux RIFE du menu (4.25 et 4.22 lite), donc sur le chemin par défaut de la cible, sans
aucun gain de vitesse en échange ; le renvoi sur le profil GPU n'est confirmé qu'en partie, et cache un second
défaut (le profil de fin cesse de compter après 262 144 mesures, sans le dire).

## Cadrage retenu

Consigne du coordinateur : vérifier seulement les quatre renvois de `complements/03.md`, sans question à
Tatoub ; conclure confirmé (nouvel ID à la suite d'A03-19), réfuté ou doublon ; signaler toute baisse de qualité
faite pour la vitesse, avec son gain, sauf si une mesure la justifie.

Le fil tourne sans carte NVIDIA : rien n'a été lancé sur GPU. Les chiffres de temps viennent des documents du
dépôt, cités avec leur fichier. Les erreurs de grille d'A03-20 ont été calculées ici (numpy, mêmes expressions
que `rife_input`, arrondi FP16 au plus proche) ; leur effet sur l'image reste « à confirmer ».

Les numéros de ligne des renvois datent d'un commit plus ancien ; ceux de ce rapport sont ceux de `328cc20`.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A03-20 | Majeur | `src/gpu/cuda/backend/rife_kernels.cu:52-58`, `src/gpu/cuda/backend/rife_fused_kernels.cu:22-27`, `:189-193`, `:250-254`, `src/kernels/glsl/rife_fused_glue.comp:121-128`, `src/kernels/glsl/rife_fused_finish.comp:56-64`, `src/app/play/model_choice.h:47-51` | **Qualité, renvoi de 45 confirmé.** Les deux RIFE du menu (4.25 et 4.22 lite) sont des exports v1 à 11 plans : la grille de base du warp (plans 7 et 8) et son pas (plans 9 et 10) entrent dans le réseau par un tenseur **FP16**. `rife_input` les arrondit (`__float2half_rn`), le moteur entier les lit tels quels, RIFE fusionné TensorRT les recalcule avec le même arrondi « au bit près » (`mesh_at`), RIFE fusionné Vulkan les relit dans l'entrée. La somme `base + flot × pas` est bien en FP32, mais la base est déjà fausse : jusqu'à **0,156 px** à 1280 de large (0,068 px efficace, 43 % des colonnes au-delà de 0,05 px), 0,117 px à 960, 0,234 px à 1920 ; en FP32 l'erreur est de 0,00004 px. L'écart entre deux colonnes voisines, qui devrait valoir 1 px, va de 0,94 à 1,25 px. C'est ce que `tools/rife/convert_rife_fp16.py:7-9` interdit pour le v2 (« en fp16, la grille normalisée perd […] de précision ; vs-mlrt force ces couches en fp32 ») : pour le v1 la règle est contournée par le type de l'entrée. Le pas, lui, ne compte pas (erreur relative de 2 × 10⁻⁴, 0,008 px pour un flot de 40 px). **Gain de vitesse : aucun.** Le moteur fusionné TensorRT calcule déjà la grille sur le périphérique, l'arrondi est du travail en plus ; aucune mesure ne justifie ce choix, hérité de l'entrée v1. | RIFE 4.25 ou 4.22 lite, chemin par défaut sur une RTX. Chaque warp interne (blocs 1 à 3, images et encodages) échantillonne les deux images sur une grille en dents de scie : les blocs suivants reçoivent des images décalées de 0 à 0,16 px selon la colonne, et corrigent un flot à partir d'elles. En variante image (sans agrandissement, ou warp avant SPAN), le warp final du réseau a le même défaut à pleine résolution : une image créée d'un plan fixe est rééchantillonnée de 0 à 0,16 px selon la colonne, donc un peu plus molle que ses voisines réelles, par bandes verticales. En variante mouvement (warp après SPAN, le défaut avec agrandissement), le warp final est celui du lecteur, juste (`warp_kernels.cu:118`) ; seuls les warps internes sont touchés. | **Fusionné** (le chemin de la cible) : calculer la base en FP32, sans arrondi, dans `mesh_at` et dans les deux shaders (qui la calculent depuis les coordonnées au lieu de lire les plans 7 à 10) ; coût nul. Les tests cachés se comparent alors à une référence FP32 de la grille (`kernels/ref/rife_fused.h`) et non plus au moteur entier, ce qui va dans le sens d'A03-02. **Entier** : livrer les exports `rife_v2` (grille dans le graphe, gardée en FP32 par le convertisseur, déjà prévu) ; il faut alors lever le refus « modèle à entrée v2 » de `rife_fused_choice.cpp:12`, ou garder le v1 pour le seul fusionné. Demande un nouveau paquet Linux (empreinte TensorRT). | Lu, et calculé pour 576 à 1920 colonnes. À confirmer sur la RTX : (1) plan fixe, `--no-upscale`, `--snapshot` d'une image créée contre l'image source, PSNR par colonnes avant et après correction ; (2) panoramique lent, PSNR de l'image créée contre ONNX Runtime FP32 du modèle d'origine de vs-mlrt, comme A44-01 l'a fait pour DistilDRBA (2,07 dB d'écart trouvés là-bas). |
| A03-21 | Mineur | `src/app/play/chain_setup.cpp:395-400`, `:470-471`, `src/app/play/settings_chain_tabs.cpp:363-374` | **Renvoi de 24 confirmé.** Quand le moteur fusionné est demandé puis refusé (découpe non reconnue, taille refusée, échec de construction ou de chargement, essai raté), le lecteur charge le réseau entier et ne l'écrit qu'au journal (`log->warn`). Rien à l'écran : la case « Fused DistilDRBA » reste cochée, et le seul indice est l'absence du mot « fused » dans l'en-tête du panneau (`chain_setup.cpp:78-81`). RIFE fusionné a le même défaut (`:471`), et c'est le moteur par défaut. L'image ne change pas (≥ 50 dB entre les deux chemins) : ce n'est pas une baisse de qualité, c'est une perte de vitesse cachée. Distinct d'A22-12 et d'A23-06 (fusionné coupé par le cran de précision, avant tout essai) et d'A24-01 (durée des messages). | Mise à jour de TensorRT ou cache de moteurs illisible : la construction d'un des quatre morceaux échoue. DistilDRBA tourne à 2,10 ms par passe au lieu de 0,92 à 1920 × 1088 (`docs/drba-fusionne.md:54`), RIFE environ trois fois plus lentement (texte de l'infobulle, `settings_chain_tabs.cpp:367`). À 144 ou 165 Hz des images sautent ; l'utilisateur voit sa case cochée et ne peut ni comprendre ni signaler la cause. | Un avis à l'écran avec le motif (« Fused DistilDRBA unavailable: … »), et sous la case du panneau l'état réel : « demandé, non chargé : motif ». Garder `fused.declined` dans `ChainEngines` pour le panneau et pour « Signaler un problème ». Même traitement pour RIFE. | Lu. Test à écrire : `prepare_distil_drba` avec un manifeste refusé, puis vérifier que l'avis et l'état du panneau portent le motif. |
| A03-22 | Mineur | `src/gpu/cuda/backend/tensorrt_backend.cpp:205-223`, `:406-411`, `:180-185`, `src/gpu/cuda/backend/gpu_timing.cpp:48-63`, `src/gpu/cuda/backend/gpu_timing.h:52-53`, `src/app/play/player_run.cpp:746-747` | **Renvoi de 39 confirmé en partie.** Deux rôles, « agrandissement » et « interpolation », écrivent déjà une ligne toutes les 240 mesures (`GpuTiming::collect`, rapporteur posé à `:180-184`), et le journal est vidé chaque seconde (`main.cpp:448`) : leur coût survit à un plantage. Tout le reste ne sort qu'à la fin : les étapes de `stage()` sont créées sans rapporteur (`:409`), soit « entrée de RIFE », « ingestion », « écriture du ring », « mesures de la paire », les moteurs par nom, le rôle « mesure » et la latence « source jusqu'à son SPAN ». Ce sont justement les lignes qui départagent RIFE et DistilDRBA (A03-14, A03-17). **Second défaut, dans le même code :** le profil de fin garde chaque mesure dans `all_`, plafonné à 262 144 (`kMaxAll`), et ignore les suivantes sans notice. À 165 Hz depuis 24 i/s (141 images créées par seconde) le plafond tombe à 31 minutes ; à 60 Hz, à 2 heures. Le profil est aussi écrit deux fois à la fermeture, sous deux formes (`keep_gpu_profile` puis le destructeur). | « Saccades au bout de 40 minutes » à 165 Hz : le profil de fin ne contient que la première demi-heure, celle qui allait bien, et affiche « 262144 mesures » comme si de rien n'était. Ou plantage après une heure : seuls les deux rôles sont au journal, pas l'étape qui a dérivé. | Donner un rapporteur aux étapes, à période longue (toutes les 60 s, une ligne par étape : médiane, p95, max). Remplacer `all_` par un histogramme à taille fixe (ou un échantillon réservoir) qui couvre toute la séance, et écrire le nombre réel de mesures. Garder une seule des deux sorties de fin. | Lu. Test à écrire sans GPU : sortir le résumé de `GpuTiming` dans une classe sans CUDA, y pousser 300 000 mesures dont les 40 000 dernières à 10 fois la valeur, et vérifier que le p95 les voit. |
| A03-23 | Mineur | `tools/drba/export_distil_drba.py:46`, `:55`, `src/app/play/drba_model.h:16-19`, `.github/workflows/export-interpolation-models.yml:36-44`, `tools/release/fetch-models.sh:47-49`, `src/kernels/spec/drba_fused.h:1`, `src/app/play/auto_config.h:19-22` | **Qualité, renvoi de 46 confirmé.** Le script sait exporter DistilDRBA v1 (cinq échelles : 16, 8, 4, 2, 1, flot affiné jusqu'à la pleine résolution) et v2 lite (trois échelles, flot arrêté à 1/4). Seul v2 lite est exporté par la CI, livré, cherché par le lecteur et reconnu par la découpe. Aucun document ne compare v1 à v2 lite ni à RIFE, en image ou en temps : le modèle le plus léger a été pris pour sa vitesse, sans mesure de ce qu'il coûte à l'image. **Gain : inconnu**, v1 n'a jamais été chronométré ; v2 lite entier fait 0,86 ms par passe à 1280 × 768 (`docs/reports/gmfss-distil-mesures-2026-10-28.md`), ce qui laisse de la marge face aux 2,2 ms de RIFE 4.25. De plus `kDistilDrbaInAuto` est faux « tant que sa porte de qualité n'est pas passée (docs/adr/0003) », or l'ADR 0003 ne parle pas de DistilDRBA : la porte n'est définie nulle part. | DistilDRBA choisi à la main (« Très rapide ») : petits mouvements (mèches, bouches, traits fins) estimés au quart de la résolution puis étirés. Si v1 rend une image proche de RIFE 4.25 pour moins de 2 ms, le lecteur passe à côté de son meilleur compromis. | Avant tout code, une mesure d'une heure : exporter v1 (`--weights v1.pkl`), chronométrer au `trtexec` à 1280 × 768 et 1920 × 1088, puis comparer v1, v2 lite, RIFE 4.22 lite et RIFE 4.25 sur des triplets d'anime dont l'image du milieu est connue (PSNR, SSIM, et l'œil sur un panoramique et sur une animation en deux). Écrire le tableau dans `docs/interpolation.md` avec la porte de qualité, chiffrée. Corriger A03-18 d'abord : v1 passerait par le même convertisseur et hériterait de la grille FP16. Si v1 est retenu : nom de fichier dans `drba_model.h`, export dans la CI ; il tournerait en entier, la découpe ne connaît que trois blocs. | Lu : `grep -ri` de « v1 » près de « drba » hors `docs/audits/` ne trouve que le script d'export. À confirmer par la mesure ci-contre. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A03-18 | Toujours présent, étendu | Même cause qu'A03-20 pour un autre modèle : dans DistilDRBA la somme entière est en FP16, dans RIFE v1 la base seule. Les deux se corrigent ensemble (grille en FP32 partout), et A03-20 touche le chemin par défaut. |
| A46-04 | Toujours présent, à nuancer | Son intitulé dit que « RIFE fusionné calcule la grille en FP32 ». C'est vrai de la somme (`rife_fused_kernels.cu:192-193`), pas de la base, arrondie en FP16 par `mesh_at` (`:22-27`) : voir A03-20. |
| A03-02 | Toujours présent | La correction d'A03-20 demande la référence stable que A03-02 réclame déjà : après elle, le fusionné s'écarte exprès du moteur entier v1. |
| A22-12, A23-06 | Toujours présents, voisins d'A03-21 | Ils couvrent le fusionné coupé par le cran de précision (`drba_fused_choice.cpp:9`, journal en `info` à `chain_setup.cpp:371-372`). A03-21 couvre le fusionné essayé puis refusé. Un seul affichage de l'état réel sous la case règle les trois. |
| A39-06 | Toujours présent, distinct d'A03-22 | A39-06 porte sur la télémétrie de la carte (fréquence, température, écrêtage) ; A03-22 sur les temps par étape. |

## Renvois

Aucun nouveau renvoi : les quatre constats restent dans la question de l'interpolation, et les zones voisines
touchées par les corrections (shaders Vulkan pour A03-20, panneau pour A03-21) sont citées dans la colonne « Où ».

## Points positifs

- Le warp du lecteur, celui qui fabrique l'image en « warp après SPAN », calcule ses positions en FP32 depuis
  les coordonnées entières (`warp_kernels.cu:85`, `:118`) : la grille FP16 d'A03-20 ne touche pas l'image finale
  sur ce chemin, seulement les warps internes du réseau.
- Les rôles « agrandissement » et « interpolation » ont déjà un relevé périodique avec le maximum
  (`gpu_timing.cpp:57-63`) : la correction d'A03-22 l'étend, elle ne l'invente pas.
- Le refus du fusionné garde toujours une image : le réseau entier prend la suite sans rouvrir la lecture, avec
  le motif au journal.
- Le script d'export reconnaît seul v1 et v2 lite d'après les poids (`export_distil_drba.py:51-62`) : la mesure
  d'A03-23 ne demande aucun code.

## Questions ouvertes

Aucune question pour Tatoub. Trois mesures à faire sur sa machine, dans cet ordre :

1. A03-20 : PSNR d'une image créée contre la référence FP32, avant et après la grille en FP32. C'est la seule
   des quatre qui touche l'image du chemin par défaut.
2. A03-23 : temps et image de DistilDRBA v1 face à v2 lite et aux deux RIFE.
3. A03-22 : rien à mesurer, mais la correction conditionne les relevés des deux premières sur une longue séance.
