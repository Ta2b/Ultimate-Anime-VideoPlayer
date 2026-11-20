# RTX Video HDR du pilote sous Windows : comment il s'active, et comment l'intégrer

Demande de Tatoub (29/10) : le « vrai » RTX HDR vidéo sous Windows, celui que mpv, VLC, MPC-HC et les navigateurs activent, sans SDK. Ce document
ne code rien : il consigne les faits (lus dans les sources de mpv) et les options. Rien n'a été testé : il n'y a pas de Windows RTX ici.

## Deux fonctions NVIDIA à ne pas confondre

| | RTX HDR (jeux, temps réel) | RTX Video HDR (vidéo) |
|---|---|---|
| Où | filtre de l'overlay de l'NVIDIA App, par application | processeur vidéo D3D11 du pilote |
| Activation | l'utilisateur, dans l'NVIDIA App | l'application, par un appel D3D11 (extension de flux) |
| Entrée | SDR de n'importe quelle application DirectX/Vulkan | SDR passé au processeur vidéo D3D11 |
| Sortie | PQ BT.2020 de la chaîne d'échange | une texture R10G10B10A2, PQ BT.2020, que l'application présente |

Seule la seconde concerne un lecteur vidéo ; c'est celle de mpv (`--vf=d3d11vpp=nvidia-true-hdr`, mpv 0.40), de VLC (module d3d11) et de Chromium.

## Comment mpv l'active (`video/filter/vf_d3d11vpp.c`)

1. Un `ID3D11VideoProcessor` ordinaire (`ID3D11VideoDevice::CreateVideoProcessor`, `ID3D11VideoContext`).
2. Test de prise en charge : `ID3D11VideoContext::VideoProcessorGetStreamExtension(vp, stream 0, &NVIDIA_TRUE_HDR_INTERFACE_GUID, sizeof(UINT), &supported)` ;
   `supported` non nul.
3. Activation : `VideoProcessorSetStreamExtension(vp, 0, &NVIDIA_TRUE_HDR_INTERFACE_GUID, sizeof(ext), &ext)` avec
   `ext = {version = 4, method = 3, enable = 1 : 1 bit, reserved = 31 bits}` (trois entiers de 32 bits).
4. `NVIDIA_TRUE_HDR_INTERFACE_GUID = {FDD62BB4-620B-4FD7-9AB3-1E59D0D544B3}` (la super-résolution RTX VSR utilise `{D43CE1B3-1F4B-48AC-BAEE-C3C25375E6F7}`
   avec `{version 1, method 2, enable 1}`).
5. Les espaces de couleur : `VideoProcessorSetStreamColorSpace1` côté entrée (le SDR de la source, par exemple `RGB_FULL_G22_NONE_P709`) et
   `VideoProcessorSetOutputColorSpace1` côté sortie (`RGB_FULL_G2084_NONE_P2020`). La sortie doit être en **BT.2020 PQ, RGB 10 bits
   (`DXGI_FORMAT_R10G10B10A2_UNORM`)** ; mpv étiquette l'image de sortie « HDR10, 1000 nits » **par supposition** (« This is only a guess »).
6. Appel par image : `CreateVideoProcessorInputView` / `OutputView` puis `VideoProcessorBlt`.
7. Refusé si la source est déjà HDR. Aucun réglage (contraste, saturation, gris moyen, pic) n'est exposé par l'extension : ils sont ceux du pilote
   (NVIDIA App : réglages vidéo), contrairement au SDK RTX Video / Maxine que le lecteur utilise sous Linux.

Références : `video/filter/vf_d3d11vpp.c` de mpv ; Chromium `ui/gl/swap_chain_presenter.cc` (identifiants des extensions) ; VLC `modules/video_output/win32/d3d11_processor.c`.

## État du dépôt

- `src/vendor_hdr/rtx_video.cpp` : chargeur du **SDK** RTX Video 1.1 (chemin CUDA, DLL du SDK), jamais validé sur Windows (prototypes « à confirmer sur le PC de
  référence »), sans rapport avec l'extension du pilote ci-dessus : à abandonner (le retirer change l'empreinte TensorRT, à faire avec le prochain paquet).
- Sous Windows, aucune route RTX n'est utilisable aujourd'hui : le paquet TensorRT Windows n'est pas publié, et sans lui le lecteur retombe sur Vulkan et le HDR
  adaptatif (émulation libplacebo de l'aspect TrueHDR).
- Le décodage Windows passe par FFmpeg (`d3d11va`, `src/media/hw_device.cpp`) ; le rendu par libplacebo sur **Vulkan** ; le moteur d'upscale et d'interpolation
  écrit dans des images Vulkan partagées (anneau de sortie). Aucun contexte D3D11 n'existe dans le rendu.

## Faisabilité avec le rendu Vulkan

Le traitement doit venir **après** SPAN et RIFE (l'image finale en SDR), donc dans l'étage de sortie, pas au décodage. Il faut un aller-retour D3D11 :

1. une texture D3D11 `R10G10B10A2` (ou `B8G8R8A8`) partagée en entrée (`D3D11_RESOURCE_MISC_SHARED_NTHANDLE`) ;
2. libplacebo (ou une copie) y écrit l'image SDR finale : import de cette texture dans Vulkan par `VK_KHR_external_memory_win32`
   (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT`) ;
3. `VideoProcessorBlt` avec l'extension → texture de sortie `R10G10B10A2` PQ BT.2020 (partagée aussi) ;
4. Vulkan la lit comme image PQ BT.2020 (`pl_frame` PQ, pic supposé) et la présente (chaîne d'échange HDR10) ;
5. synchronisation : fence D3D11 partagée (`ID3D11Fence` ↔ `VK_KHR_external_semaphore_win32`, `D3D12_FENCE_BIT`/D3D11 fence) ou mutex à clé
   (`VK_KHR_win32_keyed_mutex`), par image ; pas de copie vers l'hôte ; le coût attendu est d'environ une copie GPU et un `Blt` par image affichée
   (l'interpolation ×3 fait trois par image source).

C'est faisable (Chromium et mpv font la même chose avec D3D11 seul), mais c'est un chantier Windows de plusieurs centaines de lignes (création du
périphérique D3D11 sur le même adaptateur LUID que Vulkan, formats partagés, sémaphores, durée de vie des textures par image, repli propre si
l'extension manque ou si le pilote refuse) qui ne peut être **ni compilé ni exécuté depuis le PC Linux** : seul le CI Windows compile, et seul un PC
Windows RTX valide l'effet.

## Options

- **A. Intégration complète (D3D11 VP + interop Vulkan)** : vrai RTX Video HDR du pilote dans la chaîne ; route « RTX Video HDR (pilote) » du menu,
  grise sur les autres systèmes et GPU ; réglages (contraste, saturation, gris moyen) non pilotables (ceux du pilote). Risque : élevé sans PC Windows de
  test ; à faire par étapes (1. banc minimal Windows : un exécutable de test qui prend une image SDR, la passe par le processeur vidéo avec l'extension
  et la sauvegarde ; 2. interop Vulkan ; 3. menu).
- **B. Statu quo amélioré** : le HDR adaptatif (aspect TrueHDR émulé, courbes ajustées sur le SDK à ±6 %, plages étendues de la rc.3) comme route Windows
  par défaut, avec le filtre de l'NVIDIA App (RTX HDR « jeux ») laissé à l'utilisateur sur une sortie SDR. Aucun nouveau code risqué.
- **C. A puis B en repli** : B tel quel pour la rc.4, et A en chantier séparé validé sur le PC de Tatoub.

Recommandation : **C**. Le banc minimal de l'étape 1 de A se code sans toucher au lecteur (un outil Windows-only dans `tools/`), et c'est le seul moyen honnête
de savoir si l'extension répond à nos formats avant d'investir dans l'interop.
