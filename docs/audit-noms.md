# Audit des noms de réglages (1.0)

Demande de Tatoub (29/10) : des noms adéquats, courts, comme dans mpv, VLC, MPC-HC, PotPlayer, Plex ou OBS, sans « overtalk » :
libellé de 1 à 3 mots, infobulle d'une phrase seulement si elle sert. Chaque texte passe par `tr()` (anglais par défaut, français
au choix : Général > Application > Language) ; ce document liste les noms avant et après, page par page, au fil des PR.

## Règles suivies

- Libellé court, un nom plutôt qu'une phrase (« Gris moyen », pas « Régler le gris moyen »).
- Le même mot pour la même chose partout (Étalonnage, Sortie, Moteurs).
- Pas d'explication dans le libellé : elle va dans l'infobulle, d'une phrase.
- Les mots techniques que les autres lecteurs gardent en anglais restent (RTX HDR, SPAN, RIFE, VRR, Deband).
- Pages : ce qu'on cherche se trouve sous le nom qu'un autre lecteur lui donnerait (Lecture, Statistiques, Audio, Export).

## Étape B1 : arbre des réglages, pages HDR, Couleurs, Lecture, Statistiques, Configuration, Réinitialiser

Tableau : texte français d'avant → anglais (par défaut) / français d'après. Les phrases longues d'avant sont raccourcies à une phrase.

| Avant (français) | Après (anglais) | Après (français) |
|---|---|---|
| HDR adaptatif (tous GPU) | Adaptive HDR (any GPU) | HDR adaptatif (tous GPU) |
| Sortie | Output | Sortie |
| Moteur HDR | HDR method | Méthode HDR |
| Réglages du mode HDR (RTX HDR et HDR adaptatif) | HDR settings | Réglages HDR |
| Réglages du HDR adaptatif | Adaptive HDR settings | Réglages du HDR adaptatif |
| Pic automatique ({} nits) | Auto peak ({} nits) | Pic automatique ({} nits) |
| Pic (nits) | Peak (nits) | Pic (nits) |
| Gris moyen | Mid gray | Gris moyen |
| Débandage (RTX HDR) | Debanding (RTX HDR) | Débandage (RTX HDR) |
| Valeurs par défaut | Defaults | Valeurs par défaut |
| Contraste et saturation (HDR et SDR) | Contrast and saturation | Contraste et saturation |
| Contraste | Contrast | Contraste |
| Contraste et saturation par défaut | Reset contrast and saturation | Contraste et saturation par défaut |
| Hors RTX HDR, ils demandent « Grader la sortie » ci-dessous. | Without RTX HDR, turn on "Color grading" below. | Sans RTX HDR, activez « Étalonnage » ci-dessous. |
| Grade | Color grading | Étalonnage |
| Grader la sortie (HDR10 et SDR) | Color grading | Étalonnage |
| Préréglage | Preset | Préréglage |
| Éclats | Highlights | Éclats |
| Aplats lumineux | Bright areas | Aplats lumineux |
| Couleurs | Colors | Couleurs |
| Couleurs étendues | Wide colors | Couleurs étendues |
| Noirs profonds | Deep blacks | Noirs profonds |
| Garde-fou des couleurs | Color limiter | Limiteur de couleurs |
| Gamut Display P3 de l'écran | Display P3 gamut | Gamut Display P3 |
| Route RTX HDR : TrueHDR, puis le grade sur la sortie PQ. | RTX HDR path: TrueHDR, then the grade on the PQ output. | Voie RTX HDR : TrueHDR, puis l'étalonnage sur la sortie PQ. |
| Composition finale | Final scaling | Mise à l'échelle finale |
| Filtre vers la fenêtre | Scaler | Filtre de mise à l'échelle |
| Débander les aplats et dégradés | Debanding | Débandage |
| Noir absolu OLED | OLED black | Noir OLED |
| Noir absolu | True black | Noir absolu |
| Seuil (pas de code) | Threshold (code steps) | Seuil (pas de code) |
| Écran :  | Display:  | Écran :  |
| Soins OLED | OLED care | Soin OLED |
| Effacer la barre en pause (anti-marquage) | Hide the bar when paused (burn-in) | Masquer la barre en pause (marquage) |
| Cadence fixe (VRR sans scintillement) | Steady frame rate (VRR flicker) | Cadence fixe (scintillement VRR) |
| Langues préférées | Preferred languages | Langues préférées |
| Sinon, si la première n'est pas disponible | Fallback | Sinon |
| Langue de l'audio | Audio language | Langue audio |
| Selon le fichier | From the file | Selon le fichier |
| Langue des sous-titres | Subtitle language | Langue des sous-titres |
| Aucun | None | Aucun |
| Taille des sous-titres | Subtitle size | Taille des sous-titres |
| Taille | Size | Taille |
| Taille d'origine | Reset size | Taille d'origine |
| Épisodes | Episodes | Épisodes |
| Lire l'épisode suivant à la fin | Autoplay next episode | Lire l'épisode suivant |
| Statistiques à l'écran | On-screen statistics | Statistiques à l'écran |
| Toutes les statistiques (Tab) | All statistics (Tab) | Toutes les statistiques (Tab) |
| Chaîne (modèles, tailles, moteur) | Pipeline | Chaîne |
| Images (temps par image, sautées) | Frames | Images |
| Interpolation (créées, tenues, coupes) | Interpolation | Interpolation |
| Pistes | Tracks | Pistes |
| Sortie HDR | HDR output | Sortie HDR |
| Position exacte | Exact position | Position exacte |
| Mémoire vidéo | Video memory | Mémoire vidéo |
| Cette machine | This machine | Cette machine |
| Carte graphique inconnue | Unknown graphics card | Carte graphique inconnue |
| Écran {:.0f} Hz | Display {:.0f} Hz | Écran {:.0f} Hz |
| Configuration automatique | Automatic setup | Configuration automatique |
| Priorité | Priority | Priorité |
| Assistant | Setup wizard | Assistant |
| Relancer l'assistant | Run the wizard again | Relancer l'assistant |
| Profil d'origine | Profile defaults | Profil d'origine |
| Revenir au profil d'origine | Reset profile | Réinitialiser le profil |
| Général | General | Général |
| Configuration | Setup | Configuration |
| Lecture | Playback | Lecture |
| Statistiques | Statistics | Statistiques |
| Réinitialiser | Reset | Réinitialiser |
| Image IA | AI image | Image IA |
| Agrandissement | Upscaling | Agrandissement |
| Fluidité | Smoothness | Fluidité |
| HDR et couleurs | HDR and color | HDR et couleurs |
| Conversion HDR | HDR | HDR |
| Export et moteurs | Export and engines | Export et moteurs |
| Moteurs | Engines | Moteurs |
| Paramètres | Settings | Paramètres |
| Réglages avancés | Advanced settings | Réglages avancés |
| Appliquer (rouvre la vidéo) | Apply (reopens the video) | Appliquer (rouvre la vidéo) |
| Abandonner ces changements | Discard changes | Abandonner les changements |
| Auto (après SPAN au-dessus de 1080p) | Auto (after SPAN above 1080p) | Auto (après SPAN au-dessus de 1080p) |
| Avant SPAN (qualité maximale) | Before SPAN (best quality) | Avant SPAN (qualité maximale) |
| Après SPAN (le plus rapide) | After SPAN (fastest) | Après SPAN (le plus rapide) |
| Défaut de libplacebo | libplacebo default | Défaut de libplacebo |
| EWA Lanczos (fin) | EWA Lanczos | EWA Lanczos |
| EWA Lanczos net | EWA Lanczos sharp | EWA Lanczos net |
| Auto (selon l'écran) | Auto (from the display) | Auto (selon l'écran) |
| Activé | On | Activé |
| Désactivé | Off | Désactivé |
| Recommandé | Recommended | Recommandé |
| Le plus rapide | Fastest | Le plus rapide |
| Le plus fin | Finest | Le plus fin |
| Le plus lent | Slowest | Le plus lent |
| Expérimental | Experimental | Expérimental |
| l'appareil de sortie change | the output device changes | l'appareil de sortie change |
| RTX HDR s'allume | RTX HDR turns on | RTX HDR s'allume |
| RTX HDR s'éteint | RTX HDR turns off | RTX HDR s'éteint |
| le moteur du mouvement change | the interpolation engine changes | le moteur d'interpolation change |
| RIFE entier à la place du RIFE fusionné | whole RIFE instead of fused RIFE | RIFE entier à la place du RIFE fusionné |
| DistilDRBA entier à la place du DistilDRBA fusionné | whole DistilDRBA instead of fused DistilDRBA | DistilDRBA entier à la place du DistilDRBA fusionné |
| la paire GMFSS entière à la place de GMFSS fusionné | whole GMFSS pair instead of fused GMFSS | la paire GMFSS entière à la place de GMFSS fusionné |
| le warp passe avant SPAN | warp runs before SPAN | le warp passe avant SPAN |
| le warp passe après SPAN | warp runs after SPAN | le warp passe après SPAN |
| le warp passe selon la sortie | warp order follows the output | l'ordre du warp suit la sortie |
|  ; la lecture reprend au même endroit | ; playback resumes at the same spot |  ; la lecture reprend au même endroit |
| Économe | Economy | Économe |
| Rapide | Fast | Rapide |
| Qualité | Quality | Qualité |
| Personnalisé | Custom | Personnalisé |
| Fluidité | Smoothness | Fluidité |
| Équilibre | Balance | Équilibre |
| Éclatant | Vivid | Éclatant |
| Naturel | Natural | Naturel |

### Infobulles et notes raccourcies

| Avant (français) | Après (anglais) |
|---|---|
| RTX HDR demande le moteur TensorRT (écran des moteurs) : sous Vulkan, le HDR adaptatif le remplace. | RTX HDR needs the TensorRT engine; Adaptive HDR is used instead. |
| RTX HDR : le SDK Maxine VideoFX n'est pas installé (écran des moteurs). | RTX HDR: the Maxine VideoFX SDK is not installed (see Engines). |
| Écran SDR : le HDR est indisponible ; l'image reste en SDR et le grade (page Couleurs) imite le rendu HDR. | SDR display: HDR is unavailable. The Color page's grade imitates the HDR look. |
| Appliqués en direct. Contraste et saturation sont dans la page Couleurs. Gris moyen : luminosité des tons moyens. Le HDR adaptatif suit la scène : moins d'expansion sur les scènes claires, ombres plus douces la nuit. | Applied live. Adaptive HDR follows the scene: less expansion on bright scenes. |
| En direct, pour RTX HDR, le HDR adaptatif et la sortie SDR (autour du gris moyen de la page Conversion HDR, sans les éclats au-dessus du blanc). | Live, for RTX HDR, Adaptive HDR and SDR output. |
| Éclatant : éclats et aplats lumineux, couleurs et noirs poussés, teinte gardée ; le garde-fou adoucit les couleurs saturées près du pic au lieu de les écrêter. Naturel : le grade d'origine du lecteur. Tout s'applique en direct. | Vivid pushes highlights, colors and blacks; Natural is the default look. Live. |
| HDR adaptatif : libplacebo en PQ, aspect TrueHDR émulé selon la scène, puis le grade. | Adaptive HDR path: libplacebo in PQ with a scene-based TrueHDR look, then the grade. |
| Mode SDR : le grade imite le rendu HDR sur l'image SDR, dans le conteneur HDR10 de l'écran. | SDR mode: the grade imitates the HDR look in the display's HDR10 container. |
| Mode SDR : le grade imite le rendu HDR dans la gamme SDR (contraste, couleurs riches, hautes lumières), pour profiter du lecteur sans écran HDR. Le préréglage Éclatant l'accentue. | SDR mode: the grade imitates the HDR look without an HDR display. Vivid strengthens it. |
| Le filtre étire la sortie de la chaîne (par exemple 1440p) vers la fenêtre ou l'écran 4K ; EWA Lanczos est le plus fin, plus lourd pour le GPU. Le débandage adoucit les marches des aplats et des dégradés, courants en anime, sans toucher aux contours. Aucun effet sur l'export. | The scaler stretches the output to the window; EWA Lanczos is sharpest but heaviest. No effect on export. |
| Anti-marquage : la barre s'efface aussi en pause après 20 s sans activité. Cadence fixe : l'image est représentée à chaque rafraîchissement, pour que le VRR garde une fréquence constante (une fréquence qui varie fait scintiller les scènes sombres d'un OLED) ; coûte une composition par rafraîchissement. | Steady frame rate keeps VRR at a constant refresh so dark scenes do not flicker on OLED. |
| Les quasi-noirs deviennent un vrai noir (0) ; le pied rejoint l'image au seuil, compté en pas de code au-dessus du noir de la source. Calé en direct sur une rampe de 32 marches. | Near-blacks become true black (0), up to the threshold. Live. |
| À l'ouverture d'un fichier, la première préférence dont l'audio et les sous-titres existent est prise telle quelle (« Aucun » : pas de sous-titres) ; sinon la deuxième ; sinon la meilleure langue de chaque liste. Vaut au prochain fichier ; les pistes en cours se changent dans la barre. | The first preference whose tracks exist is used when a file opens. Applies to the next file. |
| Agrandit ou réduit toutes les polices, panneaux compris (comme la taille de mpv). Changement en direct ; la piste se choisit dans la barre (j). | Scales all text, panels included. Live. |
| À la fin d'une vidéo ouverte par un fichier, lit la suivante du même dossier (ordre naturel des noms : ep2 avant ep10). N : épisode suivant, Maj + N : précédent, O : passer le générique (chapitre Opening ou Ending, sinon 85 s). | Plays the next file in the folder. N: next, Shift+N: previous, O: skip credits. |
| Pas encore optimisée : les réglages sont une estimation d'après la carte. | Not optimized yet: settings are an estimate from the card. |
| Auto choisit la chaîne selon votre carte, votre écran et la vidéo. Régler un réglage de l'image ou de la fluidité à la main passe en Personnalisé, sans rien perdre. | Auto picks the settings for your card, display and video. Changing one by hand switches to Custom. |
| Ce que l'automatique préfère quand la machine ne tient pas tout : plus d'images (Fluidité) ou une image plus fine (Image). | What Auto favors when the machine cannot do everything. |
| Les écrans de bienvenue (ceux sans objet sont sautés) : moteurs, image, HDR, audio, langues, mises à jour. Rien ne change avant « Terminer ». | Nothing changes until "Finish". |
| Remet le profil actif (Ctrl+1 à Ctrl+9) à ses valeurs d'origine ; les autres profils ne bougent pas. | Resets the active profile (Ctrl+1 to Ctrl+9); other profiles are untouched. |
| le moteur SPAN est rechargé (construit la première fois pour cette taille) | SPAN reloads (built the first time for this size) |
| le moteur d'interpolation est chargé (construit la première fois pour cette taille) | the interpolation engine loads (built the first time for this size) |
| RIFE fusionné (six moteurs, construits la première fois pour cette taille) | fused RIFE (six engines, built the first time for this size) |
| DistilDRBA fusionné (quatre moteurs, construits la première fois pour cette taille) | fused DistilDRBA (four engines, built the first time for this size) |
| GMFSS fusionné (deux moteurs à la place de la paire, construits la première fois pour cette taille) | fused GMFSS (two engines, built the first time for this size) |
| le modèle RIFE est chargé (construit la première fois pour cette taille) | the RIFE model loads (built the first time for this size) |
| SPAN passe en {} (construit la première fois pour cette taille) | SPAN switches to {} (built the first time for this size) |
| {} passe en {} (construit la première fois pour cette taille) | {} switches to {} (built the first time for this size) |
| GMFSS passe en {} (GMFlow en {}, construit la première fois pour cette taille) | GMFSS switches to {} (GMFlow in {}, built the first time for this size) |

## Étape B2 : Image IA (Agrandissement, Fluidité)

| Avant (français) | Après (anglais) | Après (français) |
|---|---|---|
| Source (sans réduction) | Source (no downscale) | Source (sans réduction) |
| RIFE : le moteur de base, rapide, bon partout. | RIFE: the default engine, fast and good everywhere. | RIFE : le moteur de base, rapide et bon partout. |
| Agrandissement | Upscaling | Agrandissement |
| Modèle | Model | Modèle |
| Précision de SPAN | SPAN precision | Précision de SPAN |
| Résolution | Resolution | Résolution |
| Source, sans réduction ({}) | Source, no downscale ({}) | Source, sans réduction ({}) |
| Résolution de travail | Working resolution | Résolution de travail |
| Algorithme | Downscaler | Réduction |
| SSimDownscaler (à venir) | SSimDownscaler (coming soon) | SSimDownscaler (à venir) |
| Pas de réduction : la source entre telle quelle dans SPAN. | No downscale: the source goes into SPAN as is. | Pas de réduction : la source entre telle quelle dans SPAN. |
| Désactivée | Off | Désactivée |
|  (recommandé) |  (recommended) |  (recommandé) |
| Multiplicateur | Multiplier | Multiplicateur |
| Maximum (écran) | Maximum (display) | Maximum (écran) |
| Plafond d'Auto | Auto limit | Plafond d'Auto |
| Rapide (expérimental) | Fast (experimental) | Rapide (expérimental) |
| Mode d'Auto | Auto mode | Mode d'Auto |
| Sans VRR, le mouvement peut vibrer à fréquence fixe. | Without VRR, motion may judder at a fixed refresh rate. | Sans VRR, le mouvement peut vibrer à fréquence fixe. |
| VRR coupé par le système : le mouvement peut vibrer. | VRR is turned off by the system: motion may judder. | VRR coupé par le système : le mouvement peut vibrer. |
| Grisé : au-delà de l'écran ({:.0f} Hz pour {} i/s). | Greyed out: above the display ({:.0f} Hz for {} fps). | Grisé : au-delà de l'écran ({:.0f} Hz pour {} i/s). |
| Moteur de l'interpolation | Interpolation engine | Moteur d'interpolation |
| Modèle RIFE | RIFE model | Modèle RIFE |
|  · conseillé pour cette carte |  · suggested for this card |  · conseillé pour cette carte |
| Garder les poses (DRBA) avec RIFE | Keep poses (DRBA) with RIFE | Garder les poses (DRBA) avec RIFE |
| RIFE fusionné | Fused RIFE | RIFE fusionné |
| DistilDRBA fusionné | Fused DistilDRBA | DistilDRBA fusionné |
| GMFSS fusionné | Fused GMFSS | GMFSS fusionné |
| Warp et SPAN | Warp and SPAN | Warp et SPAN |
| Smooth Motion (pilote NVIDIA) | Smooth Motion (NVIDIA driver) | Smooth Motion (pilote NVIDIA) |
| Doubler les images (pilote NVIDIA) | Double the frames (NVIDIA driver) | Doubler les images (pilote NVIDIA) |

### Infobulles et notes raccourcies

| Avant (français) | Après (anglais) |
|---|---|
| GMFSS Fortuna : réseau fait pour l'animation, les images intermédiaires les plus propres, pour plusieurs fois le coût de RIFE. Il prend ses graphes au flot à demi-échelle (_s05), qui valent le flot entier pour moitié moins de calcul. Construit la première fois (plusieurs minutes). Expérimental. | GMFSS Fortuna: built for animation, the cleanest in-between frames at several times RIFE's cost. Built on first use (several minutes). Experimental. |
| GMFSS Union : GMFSS mêlé à l'image d'un RIFE intégré au réseau, moins de défauts sur les mouvements rapides, pour un peu plus de calcul. Construit la première fois (plusieurs minutes). | GMFSS Union: GMFSS blended with a built-in RIFE, fewer artifacts on fast motion, a bit slower. Built on first use. |
| DistilDRBA : un RIFE à trois images entraîné à imiter DRBA ; le réseau garde lui-même les poses d'un personnage animé en deux, pour à peu près le coût de RIFE 4.22 lite. Une image de retard. | DistilDRBA: a three-frame RIFE trained to imitate DRBA; it holds poses by itself, at about RIFE 4.22 lite's cost. One frame of delay. |
| Apple : le flot optique de macOS (VideoToolbox, macOS 15.4 ou plus), sur le Neural Engine et le GPU, sans modèle à télécharger. Sous le backend Core ML seulement ; ailleurs RIFE prend la place. | Apple: macOS optical flow (VideoToolbox, macOS 15.4+), no model to download. Core ML backend only; RIFE is used elsewhere. |
| Performance : le plus rapide. Balanced : plus fin, plus lent. Sharp1 : contours plus nets. | Performance: fastest. Balanced: finer, slower. Sharp1: crisper edges. |
| Équilibre : FP16, le réseau tel que livré. Performance : le plus rapide (calcul en FP8 sur les RTX 40 et après), fabriqué sur votre PC par le bouton « Préparer le mode Performance » des réglages avancés ; il n'apparaît dans la liste qu'une fois prêt. Qualité : FP32, si sa variante est installée. Recharge la lecture. | Balanced is FP16; Performance is FP8 on RTX 40 and newer; Quality is FP32. Reloads playback. |
| Performance (FP8) : fabriqué sur votre PC (bouton « Préparer le mode Performance » plus bas), pour les RTX 40 et après. | Performance (FP8) is built on your PC (see "Prepare Performance mode" below), for RTX 40 and newer. |
| Taille à laquelle la vidéo est ramenée avant SPAN. SPAN la double : 720p donne une sortie en 1440p. | The size the video is reduced to before SPAN, which doubles it: 720p gives 1440p. |
| Sortie : {}, le double de la résolution de travail (SPAN ×2). | Output: {}, twice the working resolution (SPAN ×2). |
| Sans agrandissement, la sortie garde la résolution de travail : {}. | Without upscaling, the output stays at the working resolution: {}. |
| Area : moyenne exacte. Spline36 : celui d'AnimeJaNai, net. Hermite : net et sans halo, pensé pour l'animation. Catmull-Rom et Mitchell gardent du détail ; Bilinear et Bicubic adoucissent. Changement en direct. | Spline36 is AnimeJaNai's sharp choice; Hermite is sharp without halos. Applies live. |
| La cadence de sortie la plus haute que le mode Auto vise : le maximum de l'écran par défaut, ou 120 / 60 i/s pour ménager le GPU. Il baisse toujours de lui-même quand une scène est lourde. | The highest frame rate Auto aims for. It still drops by itself on heavy scenes. |
| Aucun palier : chaque image est calculée à l'instant exact où elle sera vue. | No steps: each frame is computed for the exact moment it is shown. |
| Sans VRR, la cadence est calée sur le rafraîchissement de l'écran : une image trop tardive laisse la précédente un vsync de plus. | Without VRR, frames follow the display refresh; a late frame repeats the previous one. |
| Stable : paliers de multiplicateur qui suivent ce que le GPU tient (de ×1 à ×6), avec 15 % de marge ; le palier baisse en moins d'une seconde quand une scène est lourde et remonte dès qu'elle s'allège. Rapide (expérimental) : aucun palier, chaque image est calculée à l'instant exact où elle sera affichée, dès que le GPU est libre, sans dépasser l'écran ni le plafond ; la vitesse de la vidéo reste constante. Si trop d'images sont perdues (3 % sur 3 s), la lecture repasse sur Stable avec un message. Lecture seulement : l'export garde un multiplicateur fixe. | Stable follows what the GPU can hold in steps (×1 to ×6). Fast has no steps and falls back to Stable if too many frames are lost. Playback only. |
| VRR indisponible : à fréquence fixe, un multiplicateur qui ne tombe pas sur un nombre entier de rafraîchissements (×k de 23,976 sur 165 Hz) fait vibrer le mouvement (judder). Active G-Sync / FreeSync / Adaptive Sync pour l'éviter. | VRR unavailable: turn on G-Sync / FreeSync / Adaptive Sync to avoid judder. |
| VRR présent mais coupé par le système : active la synchronisation adaptative (KDE : Affichage et écran > Synchronisation adaptative), sinon le mouvement vibre à fréquence fixe. | VRR is off in the system: turn on adaptive sync (KDE: Display > Adaptive sync). |
| À l'export, aucune image n'est sautée : chaque image créée coûte une passe, le temps de l'export suit. | Export skips no frames: each created frame costs a pass. |
| Au-delà de ×4, chaque paire demande une passe RIFE par image créée : il faut un écran rapide et un GPU qui suit, sinon des images sont sautées. | Above ×4 each pair needs a RIFE pass per created frame; a fast GPU is needed or frames are skipped. |
| Aucun modèle d'interpolation trouvé : placez un modèle RIFE ou les graphes GMFSS dans le dossier des modèles. | No interpolation model found: put a RIFE model or the GMFSS graphs in the models folder. |
| Précision de GMFSS. Performance : la paire en FP8 hors du squelette de GMFlow (15 convolutions sur 37), −15 % sur la paire, 47 dB au pire cas, RTX 40 et après. Équilibre : FP16. Qualité : FP32, un écart à l'original divisé par 4 environ, pour +7 ms avec base et +10 ms avec union. Recharge la lecture. | GMFSS precision. Performance is FP8 on RTX 40 and newer (15% faster); Quality is FP32. Reloads playback. |
| Cran Performance (FP8) : ajouter {} (release des modèles) dans {}. | Performance (FP8): add {} (from the models release) to {}. |
| Union mêle à GMFSS l'image d'un RIFE intégré au réseau : moins de défauts sur les mouvements rapides, pour un peu plus de calcul à chaque image intermédiaire. | Union adds a built-in RIFE to GMFSS: fewer artifacts on fast motion, a bit slower. |
| Modèles GMFSS introuvables dans {} ({} et {}) ; RIFE prend la place. | GMFSS models not found in {} ({} and {}); RIFE is used instead. |
| Modèle DistilDRBA introuvable dans {} ({}) ; RIFE prend la place. | DistilDRBA model not found in {} ({}); RIFE is used instead. |
| Sortie 4K (source sans réduction agrandie par SPAN) : ×3 et plus demandent un GPU haut de gamme ; SPAN 1080p → 2160p est la passe la plus lourde. Une résolution de travail de 720p allège tout. | 4K output: ×3 and above need a high-end GPU. A 720p working resolution lightens everything. |
| Sortie 4K : RIFE estime le mouvement sur une copie en 720p, plus légère. | 4K output: RIFE estimates motion on a lighter 720p copy. |
| Les coupes de scène et les dessins tenus ne sont pas interpolés. | Scene cuts and held drawings are not interpolated. |
| Un personnage animé en deux sur un fond qui défile garde son rythme (il tient sa pose puis change d'un coup) pendant que le fond s'interpole, au lieu de se déformer entre deux dessins. Avec RIFE 4.25 ou 4.22 lite, une seule passe par paire. En direct. DistilDRBA le fait de lui-même. | Characters animated on twos keep their rhythm while the background interpolates. Applies live; DistilDRBA does this itself. |
| Coupe RIFE 4.25 ou 4.22 lite en six petits moteurs TensorRT (les convolutions à basse résolution) reliés par des noyaux CUDA pour tout ce qui travaille en pleine résolution : une passe de RIFE prend un tiers du temps (2,2 ms au lieu de 6,5 ms à 768×1280 sur une RTX 4070 Ti), avec la même image à moins d'un dB près. Sous TensorRT seulement ; le RIFE entier reprend si le modèle ne se coupe pas ou si un moteur échoue. Recharge la lecture ; la première fois, six petits moteurs se construisent pour cette taille. | Splits RIFE into small TensorRT engines: about three times faster. TensorRT only. Reloads playback; built on first use for this size. |
| Coupe DistilDRBA en quatre petits moteurs TensorRT (les convolutions à basse résolution) reliés par des noyaux CUDA pour tout le reste (réductions d'images, warps, mélange) : une passe de DistilDRBA prend moins de la moitié du temps (0,92 ms au lieu de 2,10 ms à 1088×1920 sur une RTX 4070 Ti), avec la même image à moins d'un dB près. Sous TensorRT seulement, au cran Équilibre ; le DistilDRBA entier reprend si le modèle ne se coupe pas ou si un moteur échoue. Recharge la lecture ; la première fois, quatre petits moteurs se construisent pour cette taille. | Splits DistilDRBA into small TensorRT engines: more than twice as fast. TensorRT only, Balanced precision. Reloads playback; built on first use for this size. |
| Coupe la paire de GMFSS en deux moteurs TensorRT : les caractéristiques d'une image et le flot. Les caractéristiques d'une image servent à deux paires de suite : elles ne sont calculées qu'une fois, un FeatureNet de moins par paire (docs/gmfss-fusionne.md). Sous TensorRT seulement, au cran Équilibre ; la paire entière reprend si le graphe ne se coupe pas ou si un moteur échoue. Recharge la lecture ; la première fois, deux moteurs se construisent pour cette taille. | Splits the GMFSS pair into two TensorRT engines so each frame's features are computed once. TensorRT only, Balanced precision. Reloads playback. |
| Après SPAN : SPAN agrandit chaque image source une seule fois, puis RIFE déforme les images agrandies : en ×3, une passe de SPAN au lieu de trois. Avec GMFSS, les images agrandies sont splattées par son flot et corrigées du résidu de sa fusion : le détail de SPAN reste. Avant SPAN, SPAN nettoie aussi chaque image intermédiaire. Auto choisit Après dès que la sortie dépasse 1080p. Recharge la lecture. | After SPAN upscales each source frame once, then warps it: faster. Before SPAN also cleans each in-between frame. Auto picks After above 1080p. Reloads playback. |
| La couche du pilote NVIDIA (RTX 40 et 50) génère une image entre deux images présentées : RIFE ×3 puis Smooth Motion donnent environ 144 images par seconde. Vaut au prochain lancement. | The NVIDIA driver (RTX 40 and 50) adds a frame between displayed frames. Applies at next launch. |
| Grisé : activez RIFE à ×2 ou plus, sans quoi la lecture cale. | Greyed out: turn on RIFE at ×2 or more, or playback stalls. |

## À venir

B3 : Audio. B4 : Export, Moteurs, Application, Configuration (benchmark). C : assistant, écran des moteurs,
lecteur par défaut, mises à jour, messages d'erreur visibles.
