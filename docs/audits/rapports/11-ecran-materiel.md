# Rapport de l'audit 11. Écran et matériel : EDID, VRR, HDR du système, télémétrie

Branche : `claude/project-thread-vcdny3`. Commit audité : `d9bd99c`. Date : `2026-11-09`.

## Verdict

La lecture de l'EDID et du HDR du système est juste dans le cas simple (un écran, Windows ou KDE), mais elle
suppose partout qu'il n'y a qu'un écran et qu'on peut la refaire sans frais : le contrat d'écran reste celui de
l'écran d'ouverture, deux écrans de même modèle se confondent, et sous Windows le thread de la boucle relance
QueryDisplayConfig, SetupAPI, une fabrique DXGI et NVAPI plusieurs fois par seconde.

## Cadrage retenu

Question posée à Tatoub le 09/11. Réponse : **tout à égalité, sauf le VRR**, mis au second plan (« il bug sur mon
écran »). Les défauts VRR connus (GPU-01, GPU-02, GPU-05, GPU-18, WIN-01) sont revérifiés, sans recherche plus
poussée ni défaut VRR nouveau classé majeur. Le reste (contrat d'écran et multi-écran, HDR et pic du système,
coût des requêtes sur le thread de la boucle, télémétrie, tests, dette) a reçu la même profondeur.

Le conteneur n'a ni GPU, ni Windows, ni les dépendances du projet. `display_edid.cpp` a été compilé seul avec
g++ 13 (doublure de `i18n::tr`) pour démontrer A11-05. Le reste est lu ; ce qui demande une mesure sous Windows
est marqué « à confirmer ».

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A11-01 | Majeur | `src/app/play/display_contract.h:54` | `ContractWatch` copie `DisplayFacts` à sa construction (`player_run.cpp:396`) et ne la change jamais. Quand la fenêtre passe sur un autre écran, `p.display` est remplacé (`player_run.cpp:621`) mais le contrat relit KWin pour l'ancien connecteur et garde l'EDID de l'ancien écran : primaires, noir, OLED, pic de repli. `SDL_EVENT_WINDOW_DISPLAY_CHANGED` pose bien le drapeau (`display_contract.cpp:16`), mais la relecture refait le même contrat. Le MaxFALL annoncé à la swapchain en vient (`placebo_grade.cpp:230`). (GPU-04) | KDE, QD-OLED en DP-1 et LCD HDR en DP-2. Lecture ouverte sur DP-1, fenêtre glissée sur DP-2 : la courbe ABL OLED et le blanc de référence de DP-1 donnent le MaxFALL envoyé au LCD, et le journal « contrat d'écran » décrit encore l'OLED. | Donner à `ContractWatch` un `set_display(DisplayFacts)` appelé là où `p.display` change, qui pose `hdr_changed_` ; ou lui passer une référence à `p.display` comme le fait `Chain::display`. | Lu. À confirmer : deux écrans sous KDE, glisser la fenêtre, lire « contrat d'écran » et « --hdr-metadata » dans le journal. |
| A11-02 | Majeur | `src/app/play/host_facts.cpp:39-81`, `:183-229`, `src/app/play/system_peak.cpp:130-155`, `src/app/play/display_edid.cpp:298-325`, `src/app/play/window_vrr.cpp:101-118` | Sous Windows, le thread de la boucle (celui qui présente : `session.handle` puis `on_frame`) fait toutes les 500 ms deux fois `windows_mode` (`display_refresh_hz` puis `display_period_us`, qui le rappelle : `player_run.cpp:398-400`), donc deux `GetDisplayConfigBufferSizes` + `QueryDisplayConfig` + un `DisplayConfigGetDeviceInfo` par chemin, plus NVAPI (`window_vrr_armed`). Toutes les 2 s, `content_peak_nits` ajoute une énumération SetupAPI avec lecture de registre de tous les écrans, jusqu'à deux fois (`host_facts.cpp:247` et `:255`), une fabrique DXGI créée puis détruite avec l'énumération de tous les adaptateurs et sorties, et un troisième `QueryDisplayConfig` (`windows_sdr_white_nits`). (WIN-04, GPU-16) | 4070 Ti, écran 165 Hz (6,06 ms par image), Windows 11. Si ce lot dure 2 à 5 ms toutes les 2 s, une image sur 330 rate sa vsync : saccade régulière, visible sur un panoramique. Le mode Stable compte l'image manquée et peut redescendre d'un palier. | Déplacer ces relevés sur un fil « écran » qui publie un instantané (Hz, pixels, HDR, pic, VRR) sous verrou ; ne relire qu'à `SDL_EVENT_WINDOW_DISPLAY_CHANGED`, `HDR_STATE_CHANGED`, `WM_DISPLAYCHANGE` ou toutes les 5 à 10 s. Garder la fabrique DXGI et ne la recréer que si `IsCurrent()` rend faux. Un seul `windows_mode` par relevé. | Lu. À confirmer : sous Windows, chronométrer le bloc `player_run.cpp:534-631` (journal en `trace` du temps passé) à 165 Hz et compter les images manquées avec et sans `kHdrPollUs` porté à 60 s. |
| A11-03 | Mineur | `src/app/play/display_edid.cpp:421-429` | `read_display` choisit l'écran par une sous-chaîne du nom SDL. Deux écrans du même modèle donnent le premier dans l'ordre de `/sys/class/drm` (ou de SetupAPI), et un EDID sans descripteur 0xFC, ou un nom SDL qui ne le contient pas, retombe en silence sur `found.front()`. Le connecteur choisi sert ensuite au VRR armé (Linux), à KWin et au contrat. | Deux MAG 272U E16 en DP-1 et DP-2, fenêtre sur DP-2 : KWin est lu pour DP-1 (pic, blanc), le VRR armé aussi. Sur un écran à EDID sans nom : l'autre écran est pris sans un mot. | Linux : faire correspondre l'écran SDL au connecteur (SDL_GetDisplayBounds contre la position du CRTC, ou le nom de sortie Wayland `DP-1` que SDL expose). Windows : passer par `DISPLAYCONFIG_TARGET_DEVICE_NAME::monitorDevicePath` de l'écran de la fenêtre, qui désigne l'instance SetupAPI. Journaliser quand aucun nom ne correspond. | Lu. |
| A11-04 | Mineur | `src/app/play/system_peak.cpp:85-108` | `parse_kwin_output` choisit l'entrée de `kwinoutputconfig.json` par le seul nom de connecteur. KWin garde une entrée par écran déjà branché sur ce port ; celle qui a le HDR actif gagne, même si c'est l'écran d'avant. | Un écran HDR débranché de DP-1 remplacé par un écran SDR (ou un autre HDR) : le pic et le blanc de l'ancien écran sont lus pour le nouveau, donc un pic du système faux et un mauvais choix HDR/SDR. | Choisir l'entrée par `edidIdentifier` ou `edidHash` (fabricant, produit, série de l'EDID déjà lu), le connecteur en dernier recours. | Lu. À confirmer : relire un `kwinoutputconfig.json` réel avec deux entrées pour DP-1. |
| A11-05 | Mineur | `src/app/play/display_edid.cpp:56-63`, `:145` | Un demi-flottant 0x7C00 dans le bloc DisplayID 0x21 donne un pic infini, retenu parce qu'il est « > 0 ». Sous Windows, `content_peak_nits` prend le maximum avec lui (`host_facts.cpp:255-256`) et `system_hdr` refuse un pic non fini : sortie SDR sur un écran en HDR. Le contrat reçoit aussi un pic infini (`display_contract.cpp:35`). | EDID abîmé ou mal programmé (écran bon marché, boîtier HDMI) : la sortie reste en SDR alors que Windows est en HDR, sans explication au journal. | Ne retenir que des luminances finies et plausibles (par exemple 50 à 10 000 nits) dans `read_displayid` et `read_cta`. | Démontré : `parse_edid` sur un EDID avec 0x7C00 aux octets 0x1A-0x1B du bloc 0x21 rend `peak_nits = inf` (harnais g++ seul). |
| A11-06 | Mineur | `src/app/play/display_contract.cpp:24-42` | Hors KDE, `contract_of` prend le blanc de référence à 203 et le pic à marge × 203. Sous Windows, la marge de SDL est relative au blanc SDR réglé dans Windows (`host_facts.cpp:180-182` le dit), pas à 203 : le pic et la courbe ABL du contrat sont faux d'un facteur blanc/203, et le MaxFALL annoncé avec. | Windows 11, écran 1000 nits, « luminosité du contenu SDR » à 280 nits : marge 3,57, contrat à 725 nits ; MaxFALL calculé sur ce pic. | Lire `SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT` (ou `windows_sdr_white_nits`) pour la référence sous Windows, ou prendre le pic DXGI. | Lu. À confirmer : comparer « contrat d'écran » au pic DXGI du journal avec le curseur SDR loin de sa valeur par défaut. |
| A11-07 | Mineur | `src/app/play/host_facts.cpp:247-250` | Sous Windows, le pic calibré DXGI n'est demandé que si un EDID a été lu dans le registre, alors que `system_peak_nits` n'utilise pas le connecteur sous Windows. | Écran sans EDID en registre (certains docks, écrans virtuels, Bureau à distance) : le pic de l'application « Étalonnage HDR » est ignoré, on retombe sur marge × blanc. | Appeler `system_peak_nits` sans condition sous Windows. | Lu. |
| A11-08 | Mineur | `src/app/play/gpu_telemetry.cpp:311-372` | `GpuTelemetry::sample` écrit `Impl::last` et `last_us` sans verrou. Le fil « stats » l'appelle en boucle (`chain_stats.cpp:169`), et le thread de la boucle l'appelle en fin de lecture pour l'énergie (`player_run.cpp:679`) pendant que `chain_stats` vit encore. Comportement indéfini, `std::optional<GpuSample>` lu à moitié écrit. (GPU-09) | Fin de chaque lecture sur NVIDIA : les deux appels se croisent dans la même fenêtre de 500 ms ; énergie de fin fausse au mieux. | Un `std::mutex` dans `Impl` autour de `sample`, ou arrêter le fil de relevé avant la lecture d'énergie. | Lu. À confirmer : construire avec `-fsanitize=thread` et finir une lecture. |
| A11-09 | Mineur | `src/app/play/host_facts.cpp:378-380` | `on_battery()` appelle `SDL_GetPowerInfo` sans cache, au moins deux fois par seconde depuis `apply_auto` sur le thread de la boucle (`player_run.cpp:414`). Sous Linux, SDL essaie d'abord UPower par D-Bus, en appels synchrones. (Renvoi de l'audit 08.) | CachyOS : chaque relevé de 500 ms fait des allers-retours D-Bus vers upowerd sur le thread qui présente. Bénin si upowerd répond vite, saccade s'il est lent ou bloqué. | Garder la réponse 10 à 30 s, ou la relire sur le fil « écran » d'A11-02. | Lu. À confirmer : `strace -f -e trace=sendmsg,recvmsg` sur le lecteur pendant 10 s de lecture. |
| A11-10 | Mineur | `src/app/play/gpu_telemetry.cpp:58` | « chaleur » écrit en dur dans la ligne de télémétrie, sans `i18n::tr` : l'OSD anglais affiche un mot français. Le test d'i18n ne le voit pas, parce que le texte ne passe pas par `tr()`. | Interface en anglais, carte qui chauffe : « … · power limit + chaleur ». | `i18n::tr("heat")` avec son entrée française. | Lu. |
| A11-11 | Mineur | `src/app/play/display_edid.cpp:139-147`, `:113-120` | Le pic et le noir dépendent de l'ordre des blocs : le pic CTA écrase toujours celui de DisplayID, mais le noir de DisplayID écrase celui du CTA quand DisplayID vient après, alors que le noir CTA est calculé à partir du pic CTA. Les descripteurs Adaptive-Sync sont lus par pas de 6 octets alors que leur taille dépend de la révision du bloc. | EDID avec CTA puis DisplayID : pic de l'un, noir de l'autre, d'où une heuristique OLED et un contrat incohérents. | Lire tous les blocs, puis choisir une seule source par grandeur (DisplayID d'abord, comme pour l'OLED). Prendre la taille des descripteurs dans la révision. | Lu. |
| A11-12 | Mineur | `tests/unit/app/` | Aucun test pour `content_peak_nits`, `system_hdr` sous ses deux systèmes, `ContractWatch` (changement d'écran), `read_display` à plusieurs écrans, le choix d'entrée de KWin à connecteur partagé, les luminances non finies, ni la course de `GpuTelemetry`. | Les défauts A11-01, A11-03 à A11-05 et A11-08 passent la CI. | Rendre `read_display` testable (liste d'écrans injectée), tester `parse_kwin_output` avec deux entrées, `parse_edid` avec 0x7C00, et `ContractWatch` avec un `set_display`. | Lu. |
| A11-13 | Mineur | `src/app/play/host_facts.cpp:39-81`, `:183-229`, `:291-318` | Quatre copies de la même énumération `QueryDisplayConfig` (mode, blanc SDR, HDR actif, journal) et trois de la recherche du moniteur de la fenêtre (`windows_mode`, `window_monitor`, `gdi_name` de `window_vrr.cpp`). `system_peak.cpp:161-164` calcule `port` deux fois, `gpu_telemetry.cpp:324-325` lit `power1_average` deux fois, `:342` teste `clock` après l'avoir appelé. `filename().string()` aux lignes 326 et 352 va contre la règle de la CI. | Dette : la correction d'A11-02 devra toucher quatre endroits. | Une fonction qui rend, pour l'écran de la fenêtre, le chemin d'affichage avec mode, HDR et blanc, appelée une fois par relevé. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| GPU-04 | Toujours présent | `display_contract.h:54`, `player_run.cpp:396` et `:621` : voir A11-01. La sélection par nom (A11-03) et l'entrée KWin par connecteur seul (A11-04) l'aggravent. Ce n'est pas propre à KDE : sous Windows aussi, primaires, noir et OLED restent ceux de l'écran d'ouverture. |
| WIN-04 | Toujours présent | Voir A11-02 : 2 `QueryDisplayConfig` toutes les 500 ms, puis SetupAPI (×2), DXGI et un troisième `QueryDisplayConfig` toutes les 2 s, tous sur le thread de la boucle. |
| GPU-16 | Toujours présent | Même cause qu'A11-02 et WIN-04 ; le coût réel reste à mesurer sous Windows. |
| GPU-09 | Toujours présent | Voir A11-08. La durée de vie est juste (`telemetry` déclaré avant `chain_stats`, donc détruit après le fil), seule la course reste. |
| GPU-02 | Toujours présent (VRR, second plan) | Côté zone 11 : `window_vrr.cpp:104-114`, cache statique de 250 ms commun à tout le processus, qui ignore le connecteur et la fenêtre (une lecture d'avant un changement d'écran peut servir). La branche plein écran qui ne met pas `vrr_watch` à jour est en zone 08 (A08, GPU-02 revérifié). |
| GPU-18 | Toujours présent (VRR, second plan) | `display_edid.cpp:384` pose `false` dès que `GETCONNECTOR` réussit, et `:389` lit `VRR_ENABLED` absent comme 0 : sous X11 avec le pilote NVIDIA, « non armé » au lieu d'« inconnu ». Rendre `nullopt` si la propriété manque ou si le connecteur n'a pas d'encodeur. |
| GPU-01 | Toujours présent (VRR, second plan) | Cause en zone 11 : `display_edid.cpp:320`, sous Windows la capacité ne vient que d'une plage VRR de l'EDID, et `window_vrr.cpp:96` ne remplit l'état armé que si un EDID a été lu. NVAPI dit pourtant la capacité (`bIsVRRPossible`, bit 1 de `NvVrrInfo::flags`, non lu). |
| GPU-05 | Hors zone | Plancher VRR non recalculé : `player_run.cpp`, zone 08, déjà revérifié par l'audit 08. `vrr_floor_hz()` (`display_edid.h:60`) est juste. |
| GPU-13 | Toujours présent | Côté zone 11 : `ContractWatch` reçoit `SDL_EVENT_WINDOW_HDR_STATE_CHANGED` (`display_contract.cpp:16`) mais ne l'expose pas ; la bascule du HDR attend la relecture des 2 s (`player_run.cpp:549`). Exposer le drapeau (`take_hdr_changed()`) pour relire le HDR tout de suite. |
| WIN-01 | Toujours présent, hors cible | `window_vrr.cpp:79-88` : NVAPI seulement. Radeon et Intel sous Windows sont hors de la cible du README. |
| QUA-16 | En partie présent | Dans la zone : statiques mutables (`window_vrr.cpp:105-107`, `system_peak.cpp:115-116`, `window_vrr.cpp:50-58`) ; `strtoul` sans contrôle (`display_edid.cpp:357-358`, `:376-377`), sans conséquence ici (un identifiant 0 fait échouer l'ioctl). Les traductions orphelines (zone 43), le CHANGELOG (zone 28) et `moving_area` (zones 09 et 45) sont hors de cette zone. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 08 | `src/app/play/player_run.cpp:398-400` | `read_display_facts` appelle `display_refresh_hz` puis `display_period_us`, qui le refait : un relevé coûte double sous Windows (A11-02). |
| 08 | `src/app/play/player_run.cpp:549-560` | La relecture du HDR devrait partir de l'événement SDL plutôt que du minuteur de 2 s (GPU-13, A11-02). |
| 08 | `src/app/play/player_run.cpp:619-621` | `p.display` est remplacé sans prévenir `contract_watch` (A11-01). |
| 24 | `src/app/play/chain_stats.cpp:184-208` | Le fil « stats » relève NVML toutes les 500 ms même quand l'OSD de télémétrie est caché (coût à vérifier, notamment sous Windows où NVML est plus lent). |
| 30 | `src/app/play/player_run.cpp:679` | Appel de `GpuTelemetry::sample` depuis le thread de la boucle pendant que le fil « stats » tourne (A11-08). |
| 34 | `src/app/play/host_facts.cpp:236-275` | Windows et Linux calculent le pic dans deux unités différentes (nits affichés contre unités du contenu) avec deux suites de replis : à garder en tête pour la parité. |

## Points positifs

- L'analyse de l'EDID est sans dépendance, bornée bloc par bloc, et suit les normes (CTA-861.3, DisplayID 2.0,
  HF-VSDB, FreeSync) ; le descripteur 0xFD n'est pris pour du VRR que sur un EDID 1.4 à fréquence continue.
- `nvapi64.dll`, `nvml.dll` et `libnvidia-ml.so.1` sont chargées par chemin absolu, System32 seulement sous
  Windows : aucune DLL piégée possible.
- Le choix du moniteur de la fenêtre (`MonitorFromWindow`) est fait partout sous Windows, pas le premier écran.
- `VrrWatch` et `window_size_for` sont petits, purs et testés.
- La télémétrie s'éteint proprement quand NVML manque, et rend `nvmlShutdown` à la destruction.

## Questions ouvertes

- Le coût réel du lot d'A11-02 sous Windows 11 sur la 4070 Ti à 144-165 Hz : c'est lui qui décide si WIN-04 est
  majeur ou mineur.
- Faut-il garder le contrat d'écran (8.2) tant que le contrat v2 est éteint ? Il ne sert aujourd'hui qu'au
  MaxFALL annoncé et au journal.
