# Écran du PC : ce qu'il annonce au lecteur

Relevé du 19 octobre 2026 par le module `present` (branche `pc/present`), KDE Plasma sous
Wayland, MSI MAG 272UP QD-OLED sur DP-1.

## Commandes

    # test de fumée sur la RTX (fenêtre 320x180 points, environ 2 s, jamais en plein écran)
    ajp-gpu timeout 60 build/linux-prefix/tests/unit/present/ajp_present_window_tests
    # même test sur lavapipe, Wayland puis X11 (XWayland) comme la CI
    ajp-run with-lavapipe timeout 60 build/linux-prefix/tests/unit/present/ajp_present_window_tests
    ajp-run env SDL_VIDEO_DRIVER=x11 with-lavapipe timeout 60 build/linux-prefix/tests/unit/present/ajp_present_window_tests
    # ce que le compositeur dit lui-même (lecture seule)
    kscreen-doctor -o
    wayland-info

## Ce que SDL3 lit (pilote wayland)

| Champ | Valeur | Remarque |
|---|---|---|
| Nom | Micro-Star Int'l Co., Ltd. MAG 272U E16 | `SDL_GetDisplayName` |
| Taille | 3840x2160 pixels | mode 1920x1080 points, densité 2 (échelle KDE 200 %) |
| Rafraîchissement | 165000/1000 Hz, nominal 165 | `refresh_rate_numerator/denominator` |
| HDR | actif | `SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN` |
| Marge HDR | 3,77 (1000 / 265) ; 12,05 écran assombri (1000 / 83) | relative au blanc de référence de KWin |
| Blanc SDR | 1,0 | sans valeur sous Wayland : SDL3 jette le blanc de référence |
| Crête retenue | 766 nits de contenu (3,77 × 203) | 2446 quand KDE assombrit l'écran |
| VRR | inconnu | SDL3 ne le dit pas |

Sous X11 (XWayland) : mode 2129000000/12903696 Hz (164,99999), ramené à 165, HDR inactif.

La crête est celle « vue du contenu » : KWin place le blanc 203 nits d'une sortie PQ sur son blanc
de référence (luminosité SDR, 265 nits), donc 766 nits de contenu atteignent les 1000 nits du
panneau. C'est la convention de mpv (luminances relatives recalées sur 203 nits). Conséquence :
la crête change quand KDE assombrit l'écran (référence 83 nits pendant le relevé `wayland-info`),
il faut relire l'écran sur `SDL_EVENT_WINDOW_HDR_STATE_CHANGED`.

## Swapchain et vsync mesurée

| Appareil | Sortie obtenue | Vsync mesurée | Gigue | Latence |
|---|---|---|---|---|
| RTX 4070 Ti, Wayland | HDR10 PQ BT.2020 | 164,78 Hz | 27 µs | 3 |
| lavapipe, Wayland | HDR10 PQ BT.2020 | 164,77 Hz | 129 µs | 3 |
| lavapipe, X11 | SDR | 164,81 Hz | 311 µs | 3 |

FIFO, 120 instants (environ 0,7 s), estimateur robuste (`VsyncEstimator`). Les 0,13 % sous
165,000 Hz viennent probablement du VRR « Always » de KWin : en fenêtre, l'écran suit le rythme
des soumissions. Le redimensionnement 640x360 vers 800x480 pixels est suivi par la swapchain.

Relance du relecteur (19 octobre, 20 h 25, commit de relecture de pc/present, charge 10 à 17) :
- RTX, `ajp-gpu timeout 60 …/ajp_present_window_tests` : 164,787 Hz, gigue 13 µs, PQ ;
- RTX sous ASan (`ASAN_OPTIONS=detect_leaks=0`) : 164,78 Hz, gigue 21 µs ;
- lavapipe Wayland (`ajp-run with-lavapipe …`) : 176,77 Hz, gigue 1592 µs. Sur un appareil
  logiciel, la cadence FIFO dépend de la charge du processeur : ce chiffre ne mesure pas l'écran,
  et le test ne le vérifie pas (il le note seulement). La ligne lavapipe du tableau ne vaut que
  pour une machine calme.
Chaque chiffre est une passe unique de 128 instants après 30 images d'échauffement : c'est une
estimation de période, pas un banc de temps GPU.

## Ce que dit KWin (kscreen-doctor, wayland-info)

- Modes 4K : 60, 120, 165 Hz (165 courant) ; pas de 144 Hz à cette taille.
- VRR : Always.
- HDR : actif ; luminosité SDR 265 nits ; crête EDID 1037 nits, forcée à 1000 ; moyenne max
  265 nits ; noir 0,0006 nits ; profil ICC de l'écran ; gamut large actif.
- color-management-v1 (version 3) : luminances min 0, max 1000, référence 83 (écran assombri au
  moment du relevé) ; cible min 0,005, max 1000 ; primaires BT.2020 ; fonctions PQ, gamma 2.2,
  linéaire étendu, HLG, BT.1886 ; scRGB Windows accepté.
- wp_presentation 2 (CLOCK_MONOTONIC), wp_fifo_v1, wp_commit_timing_v1, tearing control.

## Ce qui manque au lecteur sous Linux

- VRR : à poser dans les réglages (`vrr = true` pour ce PC, KWin est en « Always »).
- Noir de l'écran (0,0006 nits), indice OLED (DisplayID 0x21), moyenne max (ABL) : il faudrait
  l'EDID et le DisplayID, donc libdisplay-info, qui n'est pas une dépendance.
- Blanc de référence de KWin (265 nits) : SDL3 ne transmet que la marge relative ; le lecteur ne
  peut donc pas le proposer. Le lire exigerait color-management-v1 en direct.
- Rafraîchissement réel au retour de présentation (wp_presentation) : libplacebo ne l'expose
  pas ; l'estimateur se fonde sur les retours de `pl_swapchain_start_frame`.
