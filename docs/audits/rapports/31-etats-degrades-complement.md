# Complément de l'audit 31. États dégradés et reprise sur erreur

Branche : `claude/project-thread-uspkdf`. Commit audité : `bdfe44c`. Date : `2026-11-09`.

## Verdict

Les deux renvois tiennent : le premier est un doublon d'A31-06 (dont il corrige la portée), le second ajoute un
défaut, A31-16 : au réveil, la perte du contexte CUDA arrive souvent par le décodeur NVDEC sous un autre code que
`DeviceLost`, et la relance prévue par A31-02 ne se déclencherait pas. La position de reprise, elle, est bien
celle d'avant le sommeil.

## Cadrage retenu

Pas de question à Tatoub (consigne du coordinateur). Seuls les deux renvois de
`audits-docs/complements/31.md` sont vérifiés, dans le code du commit audité. Aucun test lancé (fil sans GPU
NVIDIA ni dépendances construites).

## Renvois traités

| De | Où | Constat du renvoi | Conclusion |
|---|---|---|---|
| 34 | `src/app/play/drba_fused_choice.cpp:9` | A31-06 touche les deux systèmes | **Doublon d'A31-06**, portée corrigée (voir ci-dessous). |
| 33 | `src/app/play/player_run.cpp:510` | Au réveil, perte du contexte CUDA et saut d'horloge ensemble : la reprise d'A31-02 doit partir de la position d'avant le sommeil | **Confirmé en partie** : la position est juste, mais la perte n'arrive pas sous `DeviceLost`. Nouveau défaut **A31-16**. |

### Renvoi 34 → 31 : A31-06 sur les deux systèmes

Doublon d'A31-06 (`SYNTHESE.md`, ligne A31-06, colonne « Touche aussi CachyOS (34) »). Vérifié :

- le repli « Équilibre à la place » se décide sur `capabilities().fp8` (`chain_setup.cpp:418`,
  `player_engines.cpp:220`), vrai sur toute carte sm ≥ 8.9 sous les deux systèmes
  (`tensorrt_backend.cpp:278`) ; l'interrupteur CMake du FP8 sous Windows ne touche que `span_fp8_cutlass`
  (`:279`) ;
- sur une RTX 40, Performance cherche `rife_v4.25_fp8.onnx` ou `distil_drba_v2_lite_fp8.onnx` ; RIFE n'en a
  aucun, ceux de DistilDRBA sont facultatifs et jamais téléchargés (`tools/release/publier-modeles-fp8.sh:10-13`) :
  le repli en Équilibre est le cas courant, sous Windows comme sous CachyOS ;
- sur une RTX 30 (sm 8.6, sans FP8), Performance retombe aussi en Équilibre (`chain_setup.cpp:418-419`,
  `player_engines.cpp:220-222`), sous les deux systèmes ;
- dans tous ces cas, `drba_fused_declined` (`drba_fused_choice.cpp:9`) et `rife_fused_declined`
  (`rife_fused_choice.cpp:11`) lisent toujours `settings.motion_precision`, et `chain_setup.cpp:370` et `:459`
  les appellent avec les réglages d'origine : le fusionné est refusé, le moteur entier FP16 tourne.

Le scénario d'A31-06 doit donc dire « Windows 11 et CachyOS, toute RTX » au lieu de « Windows 11 (FP8 éteint) ».
La correction ne change pas : calculer le cran effectif avant le jugement du fusionné. Aucune baisse de qualité
ici (le graphe est le même FP16) ; c'est une perte de vitesse sans contrepartie.

### Renvoi 33 → 31 : réveil, contexte CUDA perdu et horloge qui saute

Ce qui est juste :

- la position retenue est celle de la dernière image montrée, pas celle de l'horloge :
  `Session::view()` rend `current_->pts_us` (`session.cpp:764-771`), et `remember()` la lit
  (`player_run.cpp:362-372`). Au premier tour après le réveil, `remember()` passe avant `session.tick`
  (`player_run.cpp:648-651`, plus de 15 s écoulées), donc avant toute image nouvelle ; à l'arrêt, le
  `remember()` final (`:673`) garde la même valeur si rien n'a été montré depuis ;
- sous Windows, si l'horloge a sauté, `FrameQueue::pick_by` (`frame_queue.cpp:36-50`) montre la dernière image
  prête de la file, produite avant le sommeil : l'écart reste de la profondeur de l'anneau (quelques images) ;
- la réouverture lit aussi `session.view().position_us` (`player_run.cpp:727-729`). Une relance d'A31-02 qui
  reprend cette valeur, ou celle de `resume.toml`, repart donc d'avant le sommeil. Il ne faut pas la prendre de
  l'horloge (`ClockSnapshot::media_at`), qui a sauté.

Ce qui manque : A31-02 propose de relancer « sur `DeviceLost` ». Or, au réveil, l'étage n'est pas toujours le
premier à toucher le contexte mort (A31-16).

Les deux faits ne coïncident d'ailleurs pas souvent : sous CachyOS, `steady_clock` s'arrête pendant la veille
(pas de saut), et c'est là que le contexte CUDA se perd (sans `NVreg_PreserveVideoMemoryAllocations`) ; sous
Windows, l'horloge saute (A33-03), et le contexte survit en général, sauf réinitialisation du pilote au réveil.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A31-16 | Mineur | `src/media/video_decoder.cpp:400`, `:464` | Contexte CUDA perdu au réveil : le décodeur NVDEC est souvent le premier à le toucher. Son échec (`avcodec_send_packet` ou `avcodec_receive_frame` en erreur) devient `DecodeFailed` non recouvrable, fatal pour la lecture (`source_loop.cpp:153-157`, `:196-199`, `wiring.cpp:67-71`) ; le repli logiciel (`reopen_software`) ne joue qu'avant la première image (`video_decoder.cpp:108`, `:390-393`). L'accueil affiche « error DecodeFailed (details in the log) » (`visible_error.h:44-45`), pas la phrase de `DeviceLost`, et la relance d'A31-02, prévue sur `DeviceLost`, ne se ferait pas. Sous Windows avec saut d'horloge, en mode libre hors DistilDRBA, les images en retard sautent l'étage GPU (`upscale_stage.cpp:510-511`, `:589`) : la perte peut ne se voir qu'à l'épisode suivant. | Portable CachyOS, lecture TensorRT avec NVDEC, `systemctl suspend` puis réveil : « "ep01.mkv" stopped: error DecodeFailed », pas de relance ; l'utilisateur relance, la création des voies échoue de nouveau (A31-02). | Au refus fatal d'une lecture (étage ou source), demander `health()` au backend avant de choisir le message et la relance : contexte mort → traiter comme `DeviceLost` (message, position retenue, relance d'A31-02). Avec A31-12/A33-03 : au saut d'horloge, pause et `health()` avant tout décodage. | Lu ; à confirmer : `systemctl suspend` pendant une lecture TensorRT avec NVDEC, sans `NVreg_PreserveVideoMemoryAllocations`, et lire le premier refus du journal. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| A31-06 | Toujours présent | `drba_fused_choice.cpp:9`, `rife_fused_choice.cpp:11` ; portée : les deux systèmes, toute RTX (renvoi 34). |
| A31-02 | Toujours présent | La position de reprise est juste (`session.cpp:764-771`) ; le déclencheur doit couvrir A31-16. |
| A31-12, A33-03 | Toujours présents | Aucune détection du sommeil ni du trou de temps dans `run_chain` (`player_run.cpp:510-669`). |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 15 | `src/media/video_decoder.cpp:464` | Une erreur NVDEC après la première image n'a aucun repli logiciel : tout incident du décodeur matériel en cours de lecture arrête la vidéo. |

## Points positifs

- La position retenue suit la dernière image montrée et non l'horloge : un saut d'horloge ne la déplace pas.
- `remember()` passe avant `session.tick` dans la boucle : le premier tour après un long gel écrit encore la
  position d'avant.

## Questions ouvertes

Aucune.
