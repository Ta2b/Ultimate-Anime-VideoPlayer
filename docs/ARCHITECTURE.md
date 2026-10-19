# Architecture

Un seul cœur possède le profil, la politique et l'ordonnancement. Il résout un
`Plan` immuable ; les autres modules exécutent ce `Plan` et rien d'autre.

## Modules

Un dossier de `src/` = un module = une cible CMake `ajp::<nom>`, déclarée par
`ajp_add_module` avec ses dépendances. Le graphe est acyclique : `app` dépend de
tout, rien ne dépend de `app`. `kernels` est une feuille ; `media`, `subs`,
`audio`, `vendor_hdr` et `gpu_ir` ne dépendent publiquement que de `core_types` ;
`present` dépend de `core` (`DisplayInfo`, `CadenceChoice`, `Ratio`).

| Module | Responsabilité | Thread |
|---|---|---|
| `core` | Profil TOML v5, `resolve()`, `Plan`, cadence, choix des moteurs, `Capabilities` | main, puis valeurs immuables |
| `core_types` | Contrats entre modules : `types.h`, `inbox.h`, `messages.h`, `frame.h`, `thread.h`, `gpu_device.h`, `ring.h` | aucun |
| `core_log` | Canal de journal de chaque module, puits installés par `app` | aucun |
| `media` | Démultiplexage, décodage matériel, horodatage | source |
| `gpu` | Exécution du `Plan`, ring de sortie, backends chargés à l'exécution (`backend.h`, `kernel_ops.h`) | gpu |
| `gpu_ir` | Représentation interne des réseaux ONNX, commune aux moteurs Vulkan et Metal | gpu, preparer |
| `prepare` | Construction des moteurs, reçus de qualification | preparer |
| `vendor_hdr` | Interface `TrueHdr` : RTX Video SDK, Maxine VFX ; ordre des bloqueurs | appelé par gpu |
| `audio` | Décodage audio, sortie SDL3, horloge | audio |
| `subs` | Rendu libass en atlas bornés | subs |
| `render` | Composition libplacebo, grade, sous-titres, OSD | main |
| `present` | Swapchain, cadence, détection de l'écran | main |
| `export` | Encodage, mux, métadonnées HDR10 | export |
| `app` | Boucle SDL3, commandes, interface, threads | main |
| `kernels` | Spécification des noyaux, référence CPU FP32, demi-flottants | aucun |

Chaque thread n'attend que sa boîte `Inbox`, bornée par type de message.

### Graphe des cibles

Une flèche va d'une cible vers ce qu'elle lie ; « privé » ne se propage pas.

```
core_types ─────────────────────────────── (bibliothèque standard seulement)
core_log ── spdlog
core ────── core_types · privé : toml++
kernels ─── (feuille)
media ───── core_types · privé : FFmpeg, Vulkan, core_log
subs ────── core_types · privé : libass, core_log
audio ───── core_types · privé : FFmpeg, SDL3, core_log
vendor_hdr ─ core_types · privé : toml++, core_log, dl
gpu_ir ──── core_types · privé : kernels, core_log
gpu ─────── core, kernels, vendor_hdr
prepare ─── core, gpu · privé : toml++, core_log
render ──── core, subs
present ─── core · privé : SDL3, libplacebo, Vulkan, core_log
export ──── core, render
app ─────── tout, dont core_log
```

`core` ne dépend pas de `vendor_hdr` : `app` lui passe le texte du premier
bloqueur TrueHDR (voir plus bas). `gpu_ir` lit les demi-flottants de
`kernels/ref/half.h`, la seule implémentation du dépôt.

## Contrats entre modules

Les contrats vivent dans `core_types`, sans dépendance : chaque module les
inclut sans tirer FFmpeg, Vulkan ni spdlog.

### Images décodées (`core/frame.h`)

`media` livre des `DecodedFrame` ; `FrameData` décrit leurs plans sans exposer
FFmpeg. Mémoire hôte : `host_plane` et `host_stride`. Mémoire GPU : `native_plane`
(`CUdeviceptr`, `VkImage`, `ID3D11Texture2D*`…) et, pour la mémoire linéaire
CUDA, `native_pitch` en octets (0 pour une image opaque ou hôte). L'attente avant
lecture est `ready_event` : un `CUevent`, ou le sémaphore timeline de l'`AVVkFrame`
sous Vulkan, dont `ready_value` donne la valeur (0 pour un `CUevent` ou sans
attente). Une image Vulkan à plusieurs `VkImage` est refusée : une seule attente
par image.

Une image Vulkan se lit entre `begin_vulkan_access()` et `end_vulkan_access()`,
sous le verrou de l'`AVVkFrame` : le décodeur peut garder l'image comme référence
et la relire pendant que le lecteur la tient. La soumission attend `wait_value`,
part de la disposition et de la file notées dans l'image, et signale
`signal_value` ; `end_vulkan_access()`, appelé juste après `vkQueueSubmit`, note
la nouvelle valeur et la nouvelle disposition (rien si rien n'a été soumis).
`ready_value` n'est qu'un instantané de la livraison. Une `FrameData` qui ne
redéfinit pas l'accès Vulkan le refuse (`FormatUnsupported`).

### Horodatage inconnu

Un pts inconnu vaut `kNoPts` (`std::numeric_limits<std::int64_t>::min()`, la
valeur d'`AV_NOPTS_VALUE`) partout où un pts tient dans un `std::int64_t` :
`DecodedFrame`, `media::Packet`, `RingFrame`. `media` traduit `AV_NOPTS_VALUE`
en `kNoPts` ; jamais un 0 inventé, même après un flush. Un module qui préfère
`std::optional` (`core/cadence`) passe par `known_pts()` et `pts_or_unknown()`,
les deux seules conversions ; `stamp_of()` en fait usage pour la cadence. Un
calcul sur un pts écarte d'abord `kNoPts`.

### Appareil de rendu (`core/gpu_device.h`)

`present` crée l'appareil de rendu et le décrit par `GpuDevice` : API, identifiants
PCI (le fabricant s'en déduit par `vendor_from_pci`, jamais du nom), nom, UUID de
16 octets, LUID sous Windows, version du pilote. Sous Vulkan, `VulkanDevice` porte
ce que libplacebo et FFmpeg demandent pour partager un `VkDevice` : handles
opaques, `vkGetInstanceProcAddr`, version d'API, extensions et fonctionnalités
activées, familles de files, files graphique, calcul et décodage vidéo, et le
verrou `lock_queue` / `unlock_queue` qu'un module prend autour de chaque
soumission. `DeviceUuid` n'a qu'une définition, partagée par `media` et `gpu`.

### Backends (`gpu/backend.h`, ABI 3)

Chaque backend est une bibliothèque chargée à l'exécution. La frontière passe des
types C++ (`std::string`, `std::function`, `std::expected`) : le backend se
construit avec la même bibliothèque standard que le lecteur. Le chargeur lit
`ajp_backend_abi()` et `ajp_backend_layout()`, puis `check_backend_abi()` refuse
une autre version ou une autre empreinte binaire (`AbiMismatch`) avant d'appeler
`ajp_backend_create(kBackendAbi, &init)`. L'empreinte `kBackendLayout` couvre la
taille et l'alignement de chaque type qui traverse la frontière et la bibliothèque
standard (libstdc++ et son ABI de `std::string`, libc++, STL de MSVC et son CRT) :
elle trahit des en-têtes changés sans changement de `kBackendAbi`. Les quatre
points d'entrée sont `noexcept` (sous MSVC, une exception sortie d'une fonction
`extern "C"` est un comportement indéfini) ; `BackendAbiFn`, `BackendLayoutFn`,
`BackendCreateFn` et `BackendDestroyFn` typent les adresses lues par `dlsym`.

`BackendInit` porte l'appareil de rendu, le dossier de cache, le journal du
backend et l'adresse où écrire le refus quand la création échoue, valable le temps
de l'appel. `enqueue()`, `signal()` et `wait()` rendent l'erreur de l'API
(`enqueueV3`, `vkQueueSubmit`, perte de l'appareil) en refus nommé. Un backend
alloue ses tenseurs (`alloc_tensor`, un `TensorBuffer` qui libère à sa
destruction) et expose par `kernels()` les noyaux de `kernels/spec.h` sur ses
tenseurs (`gpu/kernel_ops.h`) : ingestion YUV vers RGB planaire,
rééchantillonnage Spline36, bilinéaire ou Area, remplissage mod-64 et recadrage de
RIFE, warp et mélange à l'instant t, écriture d'un slot du ring en RGBA16 UNORM,
mesures de coupe et de pire bloc 32 × 32 relues sans attente pour la paire
suivante. Un format ou une taille hors contrat est un refus nommé, avant toute
soumission.

Durées de vie : chaque `Engine` et `TensorBuffer` meurt avant le `Backend` qui l'a
créé, donc avant `ajp_backend_destroy` et le déchargement de la bibliothèque, où
vit leur code. Une opération qui lit une `DecodedFrame` en garde une copie de
`frame.data` jusqu'à la fin de sa lecture côté GPU, constatée sans attente hôte :
l'image ne revient au pool du décodeur qu'ensuite. `write_ring` relâche les imports
des générations précédentes du ring au premier appel d'une plus récente.

### Ring de sortie (`core/ring.h`)

`render` alloue les slots RGBA16 UNORM et les exporte (`RingSlotDesc` : image,
mémoire et sémaphore timeline exportés). `gpu` écrit un slot, signale une valeur
impaire et envoie `RingFrame` au thread main (slot, pts, paire de sources, instant
t, `REPEATED`, coupe, époque, valeur à attendre). Main signale la valeur suivante
après son rendu et renvoie `release_of(frame)`, un `SlotFree` (slot, valeur,
génération du ring). Un seul écrivain à la fois par slot : la suite des valeurs
croît sans attente hôte.

Chaque `RingFrame` reçu rend son slot, même une image jetée (époque passée,
retard) : main soumet l'attente et le signal sans rien dessiner. Un slot passe
d'un côté à l'autre en `VK_IMAGE_LAYOUT_GENERAL` (`kRingLayout`) ; vers CUDA, main
le cède à `VK_QUEUE_FAMILY_EXTERNAL`, et sur le même `VkDevice` l'image est en
partage concurrent, échantillonnable et inscriptible. Un ring recréé change de
génération : `gpu` cesse d'écrire l'ancienne et ignore ses `SlotFree` ; main ne
présente plus ses images mais attend leur valeur, et ne détruit l'ancien ring
qu'après le premier `RingFrame` de la nouvelle génération et la fin de ces
attentes côté GPU.

### Écran et cadence (`core/resolve.h`)

`DisplayInfo` décrit l'écran : taille, rafraîchissement (0 sans VRR : inconnu),
HDR actif, VRR, crête, noir et blanc SDR en nits, espace de sortie (sRGB, PQ,
scRGB), et la provenance de chaque champ (défaut, profil, système, `--display`).
Quand l'écran est connu, `resolve()` passe la règle choisie par `pace_rule()` :
une interpolation unique prend le facteur calé sur les vsyncs, et le `Plan` porte
sa `CadenceChoice`, comprise dans son hachage.

### TrueHDR dans `resolve()`

`Capabilities::true_hdr_blocker` porte la phrase du premier bloqueur, vide quand
TrueHDR peut être tenté. `app` la tire de `vendor_hdr::first_blocker_reason()` et
la passe à `to_resolve_capabilities()` ; la valeur par défaut refuse, tant que
personne n'a vérifié. Un profil TrueHDR bloqué est refusé avec ce texte, tel quel.

## Threads

Un thread se lance par `start_thread()` (`core/thread.h`), jamais par un
`std::jthread` construit à la main. Un refus du système (`std::system_error`)
devient `ThreadUnavailable`, rendu par `std::expected`. Le corps reçoit un
rapporteur : une exception qui s'en échappe devient `ThreadFailed` et lui est
remise, et le thread s'arrête sans emporter le processus. Un module dont le
thread démarre dans `start()` plutôt que dans son constructeur peut ainsi
rendre le refus à son appelant.

## Journal

spdlog en `SPDLOG_USE_STD_FORMAT`, par la cible `ajp::core_log` :

- un module journalise par `log_channel("<module>")`, jamais par le journal par
  défaut ni par le registre de spdlog ; son `log.h` ne fait que nommer le canal ;
- `app` appelle `install_log_sinks()` depuis main, avant de lancer les threads ;
  tous les canaux partagent alors un répartiteur, si bien qu'un puits installé
  plus tard (fichier, OSD) reçoit aussi les messages des canaux déjà créés ;
- le niveau d'un canal est celui du journal par défaut à sa création ;
- un backend a sa propre copie de spdlog : il écrit par `BackendInit::log`, que
  `app` relie à un canal.

## Moteurs

Seul notre moteur Vulkan est livré partout (Metal en plus sous macOS). Les autres
sont des paquets facultatifs : l'écran « Moteurs » du premier lancement les
présélectionne selon le GPU détecté (`core/engines.h`), avec des badges
(Recommandé, NVIDIA uniquement, Compatibilité…). On y revient depuis les
paramètres. Un moteur n'est retiré que s'il n'a plus d'intérêt nulle part.

## Construction sous Linux

Les préréglages `linux-prefix*` lient chaque exécutable et bibliothèque partagée
avec `--disable-new-dtags` (`AJP_TRANSITIVE_RPATH`) : un `DT_RPATH` vaut aussi pour
les dépendances indirectes, et les bibliothèques FFmpeg du préfixe, sans RUNPATH,
se trouvent sans `LD_LIBRARY_PATH`.

## Règles de code

- C++23 pour l'hôte, avec `expected`, `format`, `print`, `span`, `ranges`,
  `jthread` et initialiseurs désignés. Modules C++, `generator`, `stacktrace`,
  `flat_map`, `mdspan`, `move_only_function` interdits. `.cu` en C++20.
- Un fichier vise moins de 500 lignes ; la CI échoue au-delà de 800.
- Aucun `getenv` : toute option vient du profil ou de `DebugFlags`.
- Erreurs : `std::expected<T, Refusal>` ; une exception ne traverse jamais un
  module ni la boucle d'un thread. Une valeur hors plage est refusée, jamais
  corrigée en silence.
- Aucun global mutable ; chaque thread possède ses données.
- Journal : la convention de la section Journal.
- Tests : Catch2, un exécutable par module dans `tests/unit/<module>/`. Tous les
  tests passent sous ASan et UBSan, et sous TSan, lecteur compris (Xvfb et
  lavapipe en CI).
- Code repris d'un dépôt existant : dépôt et commit d'origine en tête de fichier,
  et un test de parité avant sa première modification.

## Décisions

Les décisions d'architecture vivent dans `docs/adr/`, une par fichier.
