# Rapport de l'audit 15. Décodage, matériel, formats et couleurs

Branche : `claude/project-thread-mgmee8`. Commit audité : `db39c7f`. Date : `2026-11-09`.

## Verdict

Le décodage NVDEC de la cible est propre (contexte primaire choisi par UUID, flux non bloquant, événements en pool,
surfaces directes bornées) et la traduction des couleurs suit mpv, mais une source 4:4:4 (ou 4:2:2 sur RTX 50)
décodée par NVDEC arrête toujours la lecture sous TensorRT (MED-02), un enregistrement TV 1080i télécinéma passe
entièrement par bwdif sans IVTC, et aucune erreur NVDEC en cours de flux ne retombe en logiciel. Rien dans cette zone
ne pénalise DistilDRBA plus que RIFE : le décodage coûte la même chose aux deux.

## Cadrage retenu

Pas de questions posées, à la demande de Tatoub : audit au maximum. RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent
autant ; la question de la vitesse de DistilDRBA a été cherchée dans cette zone (voir le verdict et les points
positifs) ; le VRR est hors sujet. Cible : NVIDIA sous TensorRT, Windows 11 et CachyOS. Sous TensorRT, les essais du
décodeur sont `Cuda` puis `Software` (`src/app/play/chain_setup.cpp:43`), sans copie hôte (`download = false`) :
D3D11VA, VA-API et Vulkan Video ne servent qu'au repli Vulkan et n'ont été lus que pour leur interface.

Aucune carte NVIDIA ni dépendance du projet dans le conteneur : la suite Catch2 n'a pas pu être construite. Les
FFmpeg 6.1 de la distribution ont permis deux petits programmes qui reproduisent à l'identique `Deinterlacer`
(`deinterlace.cpp`) et `to_chain_yuv` (`frame_data.cpp`) : ce qu'ils montrent est marqué « démontré ».

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A15-01 | Majeur | `src/media/video_decoder.cpp:320-323`, `src/media/frame_data.cpp:289-290`, `src/media/color.cpp:205-229` ; `src/gpu/cuda/backend/tensorrt_kernels.cpp:24-41` (appelé) | MED-02 toujours présent sous TensorRT. Seul H.264 4:2:2 et 4:4:4 est écarté de NVDEC d'avance (`hardware_hopeless`). Une HEVC RExt 4:4:4 (ou VP9 profil 1/3) s'ouvre sous NVDEC : en 8 bits, `sw_format` = `yuv444p`, refusé par `map_pixel_format` dans `to_decoded_frame` ; en 10 ou 12 bits, `yuv444p16` passe le décodeur mais l'ingestion TensorRT n'accepte en mémoire CUDA que NV12, P010 et P016 (`import_formats`, `tensorrt_backend.cpp:285`). Sur RTX 50, NVDEC décode aussi la HEVC 4:2:2 (FFmpeg ≥ 7.1 rend `p210`/`p216`/`nv16`), refusée de même, alors qu'une RTX 40 la retombe en logiciel et la lit. | Release fansub HEVC 4:4:4 10 bits (il en existe en anime), TensorRT : la première image est refusée, la lecture s'arrête sur une erreur de format. Le même fichier se lit sous Vulkan (copie hôte puis conversion). | Court terme : à l'ouverture, sauter `Cuda` quand le format de la piste (`av.format`, ou profil HEVC RExt / AV1 profil 1-2 / VP9 profil 1-3) n'est pas en 4:2:0 et que le backend ne l'importe pas ; le chemin logiciel convertit déjà. Mieux : une ingestion TensorRT 4:4:4 (le format `kernels::Format::Yuv444p16` existe) qui garde la chroma pleine. | Lu. À confirmer : `ffmpeg -f lavfi -i testsrc2=s=1920x1080:d=5 -c:v libx265 -pix_fmt yuv444p10le 444.mkv`, puis `ajp 444.mkv` sous TensorRT, journal `media`. |
| A15-02 | Majeur | `src/media/deinterlace.cpp:66`, `src/media/video_decoder.cpp:325-326`, `:483-484` | MED-13 toujours présent, et plus grave qu'annoncé pour l'anime. (1) Aucun IVTC : un flux 1080i télécinéma (enregistrement TV japonais en `.ts`, MPEG-2 ou H.264) a toutes ses images marquées entrelacées ; `bwdif` en `send_frame` et `deint=interlaced` les reconstruit toutes à partir d'un champ (demi-résolution verticale interpolée) et garde la cadence 3:2 à 29,97 i/s. SPAN agrandit ces lignes interpolées, puis RIFE ou DistilDRBA interpolent une cadence 3:2 (une image sur cinq est faite de champs déjà vus) : saccade et détail perdu, avant toute IA. (2) Sous NVDEC (champ `field_order` inconnu du conteneur), les images entrelacées sortent tissées et passent sans désentrelacement : la condition de la ligne 483 exige une image hôte. | `[TV] Anime - 01.ts` (1080i, 3:2) : image lissée verticalement, cadence 24 → 30 qui saccade même en ×2. MKV H.264 PAFF sans `FieldOrder` : peignes agrandis par SPAN. | Pour 29,97 i/s entrelacé : `fieldmatch` + `decimate` (repli `bwdif` sur les seules images peignées), sortie à 23,976 que l'Auto et l'interpolation voient comme une source 24p. Pour une image NVDEC marquée entrelacée : rouvrir en logiciel par la bascule P3.22 (`reopen_software`), ou désentrelacer sur le GPU. | Lu. À confirmer : un `.ts` 1080i d'anime, `--snapshot` d'un plan fixe détaillé, comparé à `ffmpeg -vf fieldmatch,decimate`. |
| A15-03 | Mineur | `src/media/video_decoder.cpp:400`, `:464` | Pas de repli logiciel sur une erreur NVDEC en cours de flux. La bascule P3.22 ne part que de `get_format` ; toute autre erreur que `AVERROR_INVALIDDATA` à l'envoi ou à la réception (`AVERROR_EXTERNAL` d'un `cuvidDecodePicture` refusé, `ENOMEM` du pool) devient un refus `DecodeFailed`, donc la fin de la lecture, alors que les essais permettent le logiciel. | Flux dont un changement de résolution ou de niveau dépasse les capacités NVDEC, ou erreur passagère du pilote : la lecture s'arrête au lieu de continuer en logiciel. | Sous une API matérielle avec `software_allowed`, traiter la première erreur non `INVALIDDATA` comme une bascule : `fall_back` puis `reopen_software` (attente de l'image clé suivante), une seule fois par ouverture. | Lu. À confirmer : un flux H.264 qui passe en cours de route de 1080p à 4096×2304 niveau 5.2. |
| A15-04 | Mineur | `src/media/deinterlace.cpp:47-49`, `:77` | Un changement de taille ou de format dans un flux entrelacé perd l'image retenue par bwdif : `push` envoie la fin du flux à l'ancien graphe puis le remplace aussitôt, sans le vider ; le commentaire dit le contraire. | Flux TV qui passe de 1440×1080 à 1920×1080 (pub, changement de programme) : une image manque, son trou de pts est comblé par la durée nominale. | Avant `graph_ = std::move(fresh)`, tirer les images de l'ancien graphe dans une file que `pull()` rend d'abord. | Démontré : 10 images entrées (5 en 720×480, 5 en 1920×1080), 9 sorties. |
| A15-05 | Mineur | `src/media/deinterlace.cpp:79`, `:103` ; `src/media/video_decoder.cpp:165` | bwdif reçoit `frame->pts` et rend `best_effort_timestamp = pts` : pour une image entrelacée sans `pts` (AVI, MPEG-PS, flux élémentaire), l'horodatage deviné par FFmpeg est perdu, alors que sans bwdif `finish` le lirait. Après un seek, `next_pts_us` vaut `kNoPts` : toutes les images sortent en `kNoPts`. | Rip DVD MPEG-2 entrelacé en `.mpg` sans pts sur les images B : après un seek exact, `ends_before` (zone 14) juge chaque image `kNoPts` à jeter ; la cible n'est jamais atteinte. | `frame->pts = frame->best_effort_timestamp` avant `push`. | Démontré : pts absent, `best_effort` 40 000 et 80 000 en entrée, `AV_NOPTS_VALUE` en sortie. |
| A15-06 | Mineur | `src/media/frame_data.cpp:247-275`, `:284` | MED-03 toujours présent, mesuré : la conversion d'une image hôte 4:4:4 ou 4:2:2 alloue une image neuve sans pool et convertit sur un seul cœur, dans `to_decoded_frame`, donc avant que la source décide de jeter l'image (seek exact) et aussi dans la pré-analyse. | 4K 4:4:4 10 bits à 23,976 i/s : 7,5 ms par image et 25 Mo alloués puis rendus à chaque image, sur le thread source qui décode aussi. Supportable en 24p, 45 % d'un cœur en 60p ; ajouté au décodage logiciel, des pertes possibles sur un CPU modeste. | Un `AVBufferPool` (ou `av_frame_get_buffer` sur un pool de `AVFrame`), l'option `threads` de swscale, la conversion faite après la décision de saut ; à terme l'ingestion 4:4:4 (A15-01) qui la supprime. | Démontré (FFmpeg 6.1, Xeon 2,3 GHz) : 1,9 ms et 6,3 Mo en 1080p 4:4:4 10 bits, 7,5 ms et 25,1 Mo en 4K 4:4:4, 8,0 ms en 4K 4:2:2. À confirmer sur la machine de Tatoub. |
| A15-07 | Mineur | `src/media/frame_data.cpp:253`, `:268`, `:298` ; `src/media/color.cpp:96-98`, `:205-229` | MED-04 toujours présent. (1) La chroma d'une source 4:4:4 est réduite en 4:2:0 avant SPAN : la qualité de la source est jetée. (2) La position de chroma de sortie est fixée à `LEFT`, même pour une entrée 4:2:0 12 bits en `TOPLEFT` qui n'est que réduite en profondeur. (3) Profondeur incohérente : 4:4:4 hôte → 10 bits 4:2:0, 4:4:4 NVDEC → `Yuv444p16`. (4) Gris, RGB planaire (`gbrp*`), YCgCo, ICtCp et les transferts log sont refusés à la première image, dans la boucle, alors que `track_color` le sait dès la sonde : un refus de couleur en cours de flux arrête aussi la lecture. | AV1 12 bits 4:2:0 BT.2020 (`TOPLEFT`) en logiciel : chroma décalée d'un demi-pixel vers le bas. Fichier `gbrp10` (capture d'écran sans perte) : erreur à la première image, après la construction de la chaîne. | Garder `chroma_location` de l'entrée quand elle est déjà en 4:2:0 ; refuser à la sonde ce que `track_color` ou le format de piste disent illisible ; convertir RGB planaire et gris par swscale comme le 4:4:4. | Lu. |
| A15-08 | Mineur | `src/media/color.cpp:72`, `:83-84` | Matrice non étiquetée devinée par la seule taille (comme mpv), même quand les primaires sont BT.2020 ou le transfert PQ/HLG : BT.709 sur un contenu BT.2020. | HEVC HDR10 dont le VUI omet `matrix_coefficients` mais donne primaires BT.2020 et PQ : teintes faussées (verts et rouges) dans tout le grade HDR. | Si les primaires sont BT.2020 ou le transfert PQ/HLG, deviner `Bt2020Ncl`. | Lu. |
| A15-09 | Mineur | `src/media/video_decoder.cpp:166-167` | Volet décodeur de MED-16 (A13-05 couvre le pas d'image) : sans durée de paquet, une image reçoit la période moyenne de la piste, et un pts manquant est comblé par elle. En VFR, `duration_us` est fausse pour toute image qui dure plus ou moins que la moyenne. | MKV VFR sans `BlockDuration` (plans fixes à 8 i/s, mouvement à 24 i/s) : le seek exact juge sur des fins d'image fausses. | Durée par différence avec le pts de l'image suivante (une image de retard, déjà le cas sous bwdif), la moyenne en dernier recours. | Lu. |
| A15-10 | Mineur | `src/media/video_decoder.cpp:167` | MED-14 : `pts + duration` sans garde, débordement signé (comportement indéfini) pour un pts ou une durée proches de `INT64_MAX` dans un fichier forgé. | MKV forgé avec un timecode de bloc énorme et une durée : débordement, puis horodatages négatifs dans toute la chaîne. | Addition vérifiée (`__builtin_add_overflow` ou `ckd_add`), `kNoPts` en cas de débordement. | Lu. |
| A15-11 | Mineur | `src/media/stills.cpp:295` | `extract_stills` décode avec le nombre automatique de threads (jusqu'à 8), à priorité normale, alors que la pré-analyse se borne à 2-4 threads (1 sous Windows) pour ne pas disputer les cœurs au rendu. La préparation FP8 automatique tourne pendant la lecture. | Lecture d'un H.264 Hi10P (décodage logiciel) sous CachyOS pendant la préparation FP8 : deux décodeurs à 8 threads, images perdues possibles. | Reprendre la règle de `shot_probe.cpp:166-172`. | Lu. À confirmer : `ajp` sur un Hi10P avec la préparation FP8 lancée, compteur d'images perdues de l'OSD. |
| A15-12 | Mineur | `src/media/stills.cpp:230-236` | Le corpus de calibration du SPAN FP8 prend la chroma au plus proche, alors que la lecture la rééchantillonne (bilinéaire sous TensorRT, A03-01 ; Spline36 du contrat) : les images calibrées ont des marches de chroma que SPAN ne voit jamais en lecture. | Les échelles FP8 sont tirées d'une distribution un peu différente de celle de la lecture ; effet sur les plages de couleurs vives, à mesurer. | Le même rééchantillonnage que l'ingestion (ou `swscale` en Spline). | Lu. À confirmer : PSNR FP8 contre FP16 avec les deux corpus. |
| A15-13 | Mineur | `tests/unit/media/` | Sans test : la conversion hôte `to_chain_yuv` (4:4:4, 4:2:2, 12 bits, plage pleine), la bascule P3.22 après la première image (`wait_key`), le chemin entrelacé de `VideoDecoder` (`field_order` connu ou non), le changement de taille sous bwdif (A15-04), les pts absents sous bwdif (A15-05), une source 4:4:4 sous NVDEC. | Les défauts A15-01, A15-04 et A15-05 passeraient la CI. | Un test par point, sur des clips FFV1 ou x264 générés par `clip_writer` ; NVDEC 4:4:4 dans `gpu_decode_test.cpp` (`[nvidia]`). | Lu. |
| A15-14 | Mineur | `src/media/hw_device.cpp:204`, `src/media/video_decoder.h:51`, `src/media/frame_data.h:33`, `src/media/hw_api.h:52` | Code mort pour le produit : `default_attempts()`, `hw_surfaces()`, `av_frame_of()` et l'option `vulkan_compute` ne servent qu'aux tests. | Aucun effet sur l'utilisateur ; surface à entretenir. | Les retirer, ou les marquer « tests seulement ». | Lu (aucun appelant hors `src/media` et `tests/`). |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| MED-02 | Toujours présent | A15-01. Le chemin hôte est corrigé (conversion `to_chain_yuv`) ; le chemin NVDEC sous TensorRT ne l'est pas. |
| MED-03 | Toujours présent | A15-06, mesuré : 7,5 ms et 25 Mo par image 4K. Mineur en 24p. |
| MED-04 | Toujours présent | A15-07. |
| MED-13 | Toujours présent | A15-02, gravité relevée à Majeur pour les enregistrements TV. |
| MED-16 | Toujours présent | Volet décodeur A15-09 ; volet session A13-05. |
| Renvoi 03 → 15 (`tensorrt_kernels.cpp:142-143`) | Réfuté | `color.cpp:174-191` ne rend que `Left`, `Center` ou `TopLeft` ; l'ingestion les place en (0 ; ½), (½ ; ½) et (0 ; 0), c'est complet. `TOP`, `BOTTOM` et `BOTTOMLEFT` sont approchés dans `color.cpp:184-190`, avec un avertissement, à un demi-pixel de chroma au plus. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 14 | `src/media/source_loop.cpp:153-159` | L'image est décodée et convertie (A15-06) avant la décision de saut du seek exact : décider sur le pts avant `to_decoded_frame`. |
| 14 | `src/media/source_loop.cpp:21-25` | `ends_before` jette toute image `kNoPts` : avec A15-05, un seek exact ne s'arrête jamais. |
| 08 | `src/app/play/player.cpp:95-117` | La sonde ne refuse que Dolby Vision profil 5 : format de pixel, matrice ou transfert illisibles (A15-07) devraient l'être là, avec un message, avant de construire la chaîne. |
| 01 | `src/gpu/cuda/backend/tensorrt_kernels.cpp:24-41`, `tensorrt_backend.cpp:285` | L'ingestion TensorRT ne lit ni 4:4:4 ni 4:2:2 en mémoire CUDA (cause côté backend de A15-01). |
| 42 | `src/media/shot_probe.cpp:173-175` | La pré-analyse en logiciel paie aussi la conversion 4:4:4 en pleine taille (A15-06). |

## Points positifs

- **DistilDRBA contre RIFE** : le décodage est le même pour les deux ; les surfaces directes (`kHeldDecodedFrames`)
  couvrent ce que l'étage tient, DistilDRBA compris. La lenteur de DistilDRBA ne vient pas de cette zone (voir A09-07
  et les rapports 03 et 09).
- NVDEC sur le contexte primaire de l'appareil choisi par UUID, flux non bloquant posé dans le contexte FFmpeg et
  retiré à la destruction, événements de fin de copie en pool : rien de bloquant par image.
- Surfaces directes bornées par la taille (`kMaxDirectSurfacePixels`) avec une raison écrite.
- H.264 Hi10P (vieux fansubs) sauté d'avance vers le logiciel multithread ; repli de `get_format` avant la première
  image avec rejeu des paquets, après elle avec reprise à l'image clé (P3.22).
- Couleur traduite comme mpv, approximations journalisées une fois, métadonnées HDR10 bornées et ignorées en SDR,
  Dolby Vision profil 5 refusé à la sonde avec un message clair.
- `nvcuda.dll` chargé depuis System32 seulement.

## Questions ouvertes

- Tatoub lit-il des enregistrements TV 1080i, ou des releases 4:4:4 ? Cela décide si A15-02 et A15-01 passent avant
  les autres corrections.
- Version de FFmpeg livrée sous Windows et CachyOS : la sortie 4:2:2 de NVDEC (RTX 50) dépend de FFmpeg ≥ 7.1.
