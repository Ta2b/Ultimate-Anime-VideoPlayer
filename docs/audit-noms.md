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

## Étape B3 : Audio (sortie, profils, Son anime, synchronisation)

| Avant (français) | Après (anglais) | Après (français) |
|---|---|---|
| Appareil | Device | Appareil |
| Suivre l'appareil par défaut du système | Follow the system default device | Suivre l'appareil par défaut du système |
| Appareil de sortie | Output device | Appareil de sortie |
| Aucun | None | Aucun |
| {} canaux sur cet appareil. | {} channels on this device. | {} canaux sur cet appareil. |
| Écoute | Listening | Écoute |
| Casque | Headphones | Casque |
| Crossfeed en stéréo, binaural en surround. | Crossfeed for stereo, binaural for surround. | Crossfeed en stéréo, binaural en surround. |
| Enceintes | Speakers | Enceintes |
| Home cinéma | Home theater | Home cinéma |
| Multicanal, surround envoyé tel quel. | Multichannel, surround sent as is. | Multicanal, surround envoyé tel quel. |
| Grisé : cet appareil n'a que 2 canaux. | Greyed out: this device only has 2 channels. | Grisé : cet appareil n'a que 2 canaux. |
| Profil | Profile | Profil |
| Personnalisé | Custom | Personnalisé |
| Crossfeed (stéréo) | Crossfeed | Crossfeed |
| Crossfeed : filtre (Hz) | Crossfeed filter (Hz) | Filtre du crossfeed (Hz) |
| Crossfeed : retard (ms) | Crossfeed delay (ms) | Retard du crossfeed (ms) |
| Taille de la tête (m) | Head size (m) | Taille de la tête (m) |
| Ombre de la tête | Head shadow | Ombre de la tête |
| Niveau du centre (voix) | Center level (voices) | Niveau du centre (voix) |
| Gain du mixage stéréo | Stereo mix gain | Gain du mixage stéréo |
| Niveau des arrières | Rear level | Niveau des arrières |
| Niveau des basses (LFE) | Bass level (LFE) | Niveau des basses (LFE) |
| Distance des enceintes | Speaker distance | Distance des enceintes |
| Pièce (réverbération) | Room (reverb) | Pièce (réverbération) |
| Égaliseur : graves (dB) | Equalizer: bass (dB) | Égaliseur : graves (dB) |
| Égaliseur : voix (dB) | Equalizer: voices (dB) | Égaliseur : voix (dB) |
| Égaliseur : aigus (dB) | Equalizer: treble (dB) | Égaliseur : aigus (dB) |
| Nuit : compression | Night compression | Compression de nuit |
| Revenir à Naturel | Back to Natural | Revenir à Naturel |
| Son anime | Anime sound | Son anime |
| Niveau | Level | Niveau |
| Comparer (maintenir : Naturel) | Compare (hold: Natural) | Comparer (maintenir : Naturel) |
| Spatialisation | Spatialization | Spatialisation |
| Ambiance décorrélée | Decorrelated ambience | Ambiance décorrélée |
| Voix : centre porté (dB) | Voices: center boost (dB) | Voix : centre renforcé (dB) |
| Voix : présence 3 kHz (dB) | Voices: 3 kHz presence (dB) | Voix : présence 3 kHz (dB) |
| Voix en avant | Voices forward | Voix en avant |
| Renfort des graves | Bass boost | Renfort des graves |
| Normalisation de sonie (EBU R128) | Loudness normalization (EBU R128) | Normalisation de sonie (EBU R128) |
| Sonie visée (LUFS) | Target loudness (LUFS) | Sonie visée (LUFS) |
| Compresseur doux | Soft compressor | Compresseur doux |
| Plafond du limiteur (dBTP) | Limiter ceiling (dBTP) | Plafond du limiteur (dBTP) |
| Réponse HeSuVi (.wav) | HeSuVi response (.wav) | Réponse HeSuVi (.wav) |
| Aucune##hrir | None##hrir | Aucune##hrir |
| Fidèle (aucun traitement) | Original (no processing) | Fidèle (aucun traitement) |
| Profil propre à cet appareil | Own profile for this device | Profil propre à cet appareil |
| Synchronisation | Sync | Synchronisation |
| Délai audio (ms) | Audio delay (ms) | Délai audio (ms) |
| Remettre à 0 | Reset to 0 | Remettre à 0 |
| Caler la lecture sur l'écran | Sync playback to the display | Caler la lecture sur l'écran |
| Latence de l'écran (ms) | Display latency (ms) | Latence de l'écran (ms) |
| estimée | estimated | estimée |
| Naturel | Natural | Naturel |
| Le réglage d'origine, équilibré. | The balanced default. | Le réglage d'origine, équilibré. |
| Doux | Soft | Doux |
| Enceintes virtuelles | Virtual speakers | Enceintes virtuelles |
| Large | Wide | Large |
| Voix claires | Clear voices | Voix claires |
| Nuit | Night | Nuit |
| Salon | Living room | Salon |
| Voix d'anime | Anime voices | Voix d'anime |
| Anime fort | Strong anime | Anime fort |

### Infobulles et notes raccourcies

| Avant (français) | Après (anglais) |
|---|---|
| Changer d'appareil rouvre la vidéo au même endroit. Le format du son se détecte seul, piste par piste. | Changing the device reopens the video at the same spot. |
| Rendu fidèle ; le surround est mixé en stéréo, centre rehaussé. | Faithful sound; surround is mixed down to stereo with a lifted center. |
| Le profil s'entend surtout en surround ; en stéréo, il règle le crossfeed. | The profile is mostly heard in surround; in stereo it sets the crossfeed. |
| En stéréo, le profil agit par l'égaliseur et la compression ; centre, arrière et basses n'existent qu'en 5.1 et 7.1. | In stereo the profile works through the equalizer and compression; center, rear and bass exist in 5.1 and 7.1 only. |
| Tenez le bouton pour entendre la piste sans le Son anime, relâchez pour le retrouver. Le volume diffère un peu : la normalisation de sonie rapproche aussi les scènes fortes et faibles. | Hold to hear the track without Anime sound. The volume differs a little. |
| Actif pour une piste stéréo sur une sortie stéréo ; les pistes mono, 5.1 et 7.1, le mode Home cinéma et le mode Fidèle gardent leur chemin habituel. | Applies to a stereo track on a stereo output only. |
| Cet appareil a plus de 2 canaux : le Son anime ne s'applique qu'à une sortie stéréo. | This device has more than 2 channels: Anime sound needs a stereo output. |
| Un fichier WAV de réponses (14 canaux HeSuVi, 16 canaux 7.1 ou 2 canaux HRIR) remplace le modèle de tête du casque, pour le stéréo comme pour le 5.1 et le 7.1. Vide : le modèle d'origine. Latence ajoutée : environ 11 ms. | A WAV impulse response (HeSuVi 14 ch, 7.1 16 ch or 2 ch HRIR) replaces the headphone model. Empty: the default. Adds about 11 ms of latency. |
| Le lecteur ne livre aucune réponse : elles ont chacune leur licence, lisez celle du fichier que vous choisissez. | No response is bundled: each has its own licence, read the one of the file you pick. |
| Coupe tout traitement, pour comparer : le son brut de la piste, mixé simplement pour l'appareil. | Turns all processing off: the raw track, simply mixed for the device. |
| Un autre appareil reprend le profil global, sans rien demander. | Another device uses the global profile. |
| Positif : le son est retardé. Ctrl + clic sur le curseur pour saisir une valeur (jusqu'à ±10 000 ms). | Positive delays the sound. Ctrl+click the slider to type a value (up to ±10,000 ms). |
| Sur un écran à fréquence fixe, la lecture accélère ou ralentit d'au plus 0,125 % pour que chaque image tienne un nombre entier de rafraîchissements : plus aucune image doublée par dérive (une toutes les 17 s à 60 Hz sinon). Le son suit, sans changement audible. Sans calage possible (VRR, cadence trop loin), le son reste maître. | On a fixed-rate display, playback speeds up or slows down by at most 0.125% so each frame fits whole refreshes: no more doubled frames. Sound follows, inaudibly. |
| Le temps entre le choix d'une image et son apparition : estimé d'après le rendu et la file de l'écran, ou imposé ici si l'image paraît en avance ou en retard sur le son. | The time between choosing a frame and seeing it. Set it here if the picture seems early or late. |
| Casque : séparation presque intacte, juste de quoi adoucir les écoutes longues. | Headphones: nearly intact separation, just softer for long listening. |
| Casque : comme des enceintes devant soi, le plus loin de la tête. | Headphones: like speakers in front of you, far from the head. |
| Casque : tête plus grande, arrière plus présent (5.1 et 7.1). | Headphones: a bigger head, stronger rear (5.1 and 7.1). |
| Enceintes : centre rehaussé, arrière et basses discrets en 5.1 et 7.1 ; en stéréo, voix relevées et graves allégés à l'égaliseur. | Speakers: lifted center and quieter rear and bass; in stereo, louder voices and lighter bass. |
| Enceintes : dynamique comprimée (explosions réduites, murmures relevés) et voix devant ; en 5.1 et 7.1, arrière et basses très réduits. | Speakers: compressed dynamics (softer explosions, louder whispers) and voices forward. |
| Casque : enceintes dans une pièce, un peu plus loin, avec une réverbération légère. | Headphones: speakers in a room, a bit farther, with light reverb. |
| Voix plus nettes et aigus plus clairs, graves allégés : dialogues d'anime sans fatigue. | Crisper voices, clearer treble, lighter bass: tireless anime dialogue. |
| Aucun traitement du module : le son de la piste, comme avant. | No processing: the track's sound as is. |
| Stéréo élargie avec les voix devant, présence des dialogues, graves renforcés par harmoniques, sonie ramenée à -20 LUFS, compresseur doux et limiteur. | Wider stereo, voices forward, dialogue presence, bass boost, -20 LUFS loudness, soft compressor and limiter. |
| Plus large, voix en avant (+3 dB), graves plus présents, compresseur plus ferme, sonie à -18 LUFS. | Wider, voices +3 dB, more bass, firmer compressor, -18 LUFS loudness. |

## Étape B4 : Export, Moteurs, Application (lecteur par défaut, mises à jour, langue), Configuration (benchmark)

| Avant (français) | Après (anglais) | Après (français) |
|---|---|---|
| Sortie de la chaîne · {}x{} | Playback output · {}x{} | Sortie de la lecture · {}x{} |
| Choisissez le fichier exporté. | Choose the output file. | Choisissez le fichier exporté. |
| FFV1 ne va pas dans un MOV : choisissez .mkv ou .mp4. | FFV1 does not fit in a MOV: use .mkv or .mp4. | FFV1 ne va pas dans un MOV : choisissez .mkv ou .mp4. |
| Ce dossier n'existe pas. | This folder does not exist. | Ce dossier n'existe pas. |
| Ce fichier est la vidéo lue. | This file is the video being played. | Ce fichier est la vidéo lue. |
| Résolution | Resolution | Résolution |
| Taille | Size | Taille |
| Agrandissement | Upscaling | Agrandissement |
| Reprendre les réglages de la lecture | Use the playback settings | Reprendre les réglages de la lecture |
| Encodage | Encoding | Encodage |
| Qualité | Quality | Qualité |
| Cadence fixe | Constant frame rate | Cadence fixe |
| Fichier | File | Fichier |
| Nom proposé | Suggested name | Nom proposé |
| Exporter | Export | Exporter |
| Moteur en service | Active engine | Moteur actif |
| Utiliser ce moteur | Use this engine | Utiliser ce moteur |
| Annuler##moteur | Cancel##moteur | Annuler##moteur |
| Moteurs installés | Installed engines | Moteurs installés |
| Installer d'autres moteurs… | Install more engines… | Installer d'autres moteurs… |
| Moteurs construits | Built engines | Moteurs construits |
| Aucun moteur construit. | No engine built yet. | Aucun moteur construit. |
| · chargé | · loaded | · chargé |
| · en cache | · cached | · en cache |
| Dossier d'installation | Install folder | Dossier d'installation |
| Par défaut | Default | Par défaut |
| Dossier par défaut | Default folder | Dossier par défaut |
| Par défaut##moteurs | Default##moteurs | Par défaut##moteurs |
| Lecteur par défaut | Default player | Lecteur par défaut |
| AJP est déjà le lecteur par défaut des vidéos. | AJP is already the default video player. | AJP est déjà le lecteur par défaut des vidéos. |
| Ne plus me le demander | Don't ask again | Ne plus me le demander |
| Faire de ce lecteur le lecteur par défaut | Make this the default player | Faire de ce lecteur le lecteur par défaut |
| Ajouter à « Ouvrir avec » | Add to "Open with" | Ajouter à « Ouvrir avec » |
| Retirer de « Ouvrir avec » | Remove from "Open with" | Retirer de « Ouvrir avec » |
| En cours… | Working… | En cours… |
| Définir AJP comme lecteur par défaut | Set AJP as default player | Définir AJP comme lecteur par défaut |
| Fermer | Close | Fermer |
| Plus tard | Later | Plus tard |
| Réglage modifiable plus tard : Général > Application. | You can change this later in General > Application. | Réglage modifiable plus tard : Général > Application. |
| Mises à jour | Updates | Mises à jour |
| Me prévenir | Notify me | Me prévenir |
| Une notification ; vous lancez la mise à jour vous-même. | A notification; you start the update yourself. | Une notification ; vous lancez la mise à jour vous-même. |
| Automatique | Automatic | Automatique |
| Téléchargée, vérifiée, installée au prochain lancement. | Downloaded, verified, installed at next launch. | Téléchargée, vérifiée, installée au prochain lancement. |
| Jamais | Never | Jamais |
| Aucune vérification. | No checks. | Aucune vérification. |
| Canal et fréquence | Channel and frequency | Canal et fréquence |
| Canal | Channel | Canal |
| Préversions | Previews | Préversions |
| À chaque lancement | At every launch | À chaque lancement |
| Chaque semaine | Weekly | Chaque semaine |
| Chaque jour | Daily | Chaque jour |
| Fréquence | Frequency | Fréquence |
| Version installée : {} | Installed version: {} | Version installée : {} |
| Disponible : {}{} | Available: {}{} | Disponible : {}{} |
|  (préparée) |  (ready) |  (préparée) |
| Dernière vérification : {} | Last check: {} | Dernière vérification : {} |
| Vérifier maintenant | Check now | Vérifier maintenant |
| Revenir à la version précédente | Roll back to the previous version | Revenir à la version précédente |
|  : relancez le lecteur | : restart the player |  : relancez le lecteur |
| Grisé : aucune mise à jour n'a encore été installée. | Greyed out: no update has been installed yet. | Grisé : aucune mise à jour n'a encore été installée. |
| Dernière erreur :  | Last error:  | Dernière erreur :  |
| Mode Performance de SPAN (FP8) | SPAN Performance mode (FP8) | Mode Performance de SPAN (FP8) |
| Refaire la préparation sur cette vidéo | Prepare again for this video | Refaire la préparation pour cette vidéo |
| Annuler##span_prepare | Cancel##span_prepare | Annuler##span_prepare |
| préparation du mode Performance de SPAN (FP8) | SPAN Performance mode (FP8) preparation | préparation du mode Performance de SPAN (FP8) |
| Optimiser pour ma machine | Optimize for this machine | Optimiser pour ma machine |
| Optimisation annulée : rien n'a changé. | Optimization cancelled: nothing changed. | Optimisation annulée : rien n'a changé. |
| Annuler##bench | Cancel##bench | Annuler##bench |
| Annuler ces réglages | Undo these settings | Annuler ces réglages |
| Réglages d'avant rétablis. | Previous settings restored. | Réglages d'avant rétablis. |

### Infobulles et notes raccourcies

| Avant (français) | Après (anglais) |
|---|---|
| HEVC 10 bits : lu presque partout, encodé par le GPU (NVENC, AMF, Quick Sync, VideoToolbox, VA-API). | HEVC 10-bit: plays almost everywhere, encoded by the GPU. |
| AV1 10 bits : plus compact à qualité égale ; il faut un encodeur AV1 dans le GPU (RTX 40 et 50, Radeon RX 7000 et 9000, Intel Arc et Core Ultra). | AV1 10-bit: smaller at equal quality; needs a GPU AV1 encoder (RTX 40/50, Radeon RX 7000/9000, Intel Arc). |
| H.264 8 bits : pour les lecteurs anciens ; les dégradés y sont moins fins. | H.264 8-bit: for older players; gradients are coarser. |
| FFV1 sans perte du codec, en YUV 4:2:0 10 bits (la couleur est réduite avant l'encodage), encodé par le processeur : des fichiers très lourds, pour retravailler la vidéo. | FFV1 lossless, 4:2:0 10-bit, encoded by the CPU: very large files, for editing. |
| Le fichier exporté est un MKV, un MP4 ou un MOV : son nom finit par .mkv, .mp4 ou .mov. | The output is an MKV, MP4 or MOV: its name ends in .mkv, .mp4 or .mov. |
| Ouvrez une vidéo : l'export reprend sa chaîne et se propose à côté d'elle. | Open a video: the export uses its settings and is named next to it. |
| Au-dessus de la sortie de la chaîne, libplacebo agrandit encore l'image (EWA Lanczos) : un fichier plus lourd, pas plus de détail. | Above the playback output, libplacebo scales further (EWA Lanczos): a bigger file, no more detail. |
| Sous la sortie de la chaîne, libplacebo la réduit avec son meilleur filtre (EWA Lanczos). | Below the playback output, libplacebo scales down with its best filter (EWA Lanczos). |
| Mêmes moteurs et mêmes réglages qu'à la lecture, sans aucune image sautée : au moins la qualité du direct. Ni RTX HDR, ni HDR adaptatif, ni grade : l'image de la chaîne telle quelle. | Same engines and settings as playback, with no skipped frame. No RTX HDR, adaptive HDR or grading: the chain's picture as is. |
| Très haute vise une image identique à l'œil à la lecture ; Haute et Standard donnent des fichiers plus légers. | Very high aims for a picture identical to playback; High and Standard make smaller files. |
| Aux coupes de scène et sur les dessins tenus, la chaîne n'interpole pas : l'image précédente est répétée à ces instants, et le fichier garde une cadence fixe que tout logiciel de montage lit. | At scene cuts and held drawings the previous frame repeats, so the file keeps a constant frame rate any editor reads. |
| Les instants que la chaîne n'interpole pas (coupes de scène, dessins tenus) manquent : le fichier est à cadence variable, un peu plus léger, lu sans souci par les lecteurs. | Frames the chain does not interpolate are left out: variable frame rate, a bit smaller, fine in players. |
| Ce fichier existe : il ne sera remplacé qu'à la fin d'un export réussi. | This file exists: it is replaced only when the export succeeds. |
| L'audio, les sous-titres, les polices et les chapitres de la source sont copiés tels quels. La lecture s'arrête pendant l'export puis reprend au même endroit ; Échap annule l'export. | Audio, subtitles, fonts and chapters are copied as is. Playback stops during the export; Esc cancels. |
| À cette fréquence, le MKV arrondit chaque image à la milliseconde (8, 8 puis 9 ms à 120 i/s) : un nom en .mp4 ou .mov garde les instants exacts, sans les sous-titres (ASS, SRT, PGS) ni les polices. | At this frame rate MKV rounds frames to the millisecond; .mp4 or .mov keeps exact times but drops subtitles and fonts. |
| Le MP4 et le MOV gardent les instants exacts de chaque image. Les chapitres de la source sont copiés, l'audio s'il y va (AAC, AC-3, E-AC-3 ; FLAC, Opus et DTS en MP4 ; jamais TrueHD), mais ni les sous-titres (ASS, SRT, PGS) ni les polices. Pour tout garder, nommez le fichier en .mkv. La lecture s'arrête pendant l'export puis reprend au même endroit ; Échap annule l'export. | MP4 and MOV keep exact frame times, chapters and compatible audio, but no subtitles or fonts. Use .mkv to keep everything. Esc cancels. |
| Un seul moteur installé : « Installer d'autres moteurs » propose ceux qui conviennent à ce GPU. | Only one engine is installed: "Install more engines" lists those that suit this GPU. |
| Rouvre l'écran des moteurs sans relancer le lecteur ; la lecture reprend ensuite au même endroit. | Reopens the engines screen; playback resumes at the same spot. |
| Un moteur est construit une fois par modèle et par taille, puis relu du cache. | An engine is built once per model and size, then read from the cache. |
| Par défaut : {}. Libre à cet endroit : {}. Vaut au prochain lancement ; les moteurs déjà installés ne sont pas déplacés. | Default: {}. Free here: {}. Applies at next launch; installed engines are not moved. |
| Les vidéos (mkv, mp4, webm, avi, mov…) s'ouvrent dans AJP depuis le gestionnaire de fichiers. | Videos (mkv, mp4, webm, avi, mov…) open in AJP from the file manager. |
| Le lecteur ne propose plus de devenir le lecteur par défaut au lancement. | The player no longer offers to become the default at launch. |
| Ouvrir vos vidéos (mkv, mp4, webm…) directement dans AJP depuis le gestionnaire de fichiers ? | Open your videos (mkv, mp4, webm…) directly in AJP from the file manager? |
| Ce paquet n'a pas de clé de signature : les mises à jour ne peuvent pas être vérifiées. | This package has no signing key: updates cannot be verified. |
| Quand le cran « Performance » de la précision de SPAN est choisi, le lecteur prépare tout seul ce mode pour la vidéo ouverte (une minute environ, la lecture reste en FP16 pendant ce temps). Cette page sert à le refaire à la main. | With SPAN's Performance precision, the player prepares this mode by itself for the open video (about a minute, FP16 meanwhile). Redo it by hand here. |
| Le lecteur essaie quelques secondes de vidéo avec ses réglages recommandés, du plus léger au plus lourd, garde ceux que votre PC tient sans saccade, et les applique. Cela dure environ une minute (plusieurs minutes la toute première fois, le temps de préparer les moteurs). | The player tries a few seconds of video with its recommended settings, lightest to heaviest, keeps what your PC holds without stutter and applies it. About a minute (several the first time). |
| To apply: playback reopens at the same spot with {}; its engines are built the first time for each model and size. | To apply: playback reopens at the same spot with {}; its engines are built the first time for each model and size. |

## Étape C1 : assistant de premier lancement et écran des moteurs

| Avant (français) | Après (anglais) | Après (français) |
|---|---|---|
| J'accepte sa licence | I accept its licence | J'accepte sa licence |
| Lire la licence | Read the licence | Lire la licence |
| Clé d'API NGC | NGC API key | Clé d'API NGC |
| Obtenir une clé | Get a key | Obtenir une clé |
| installé | installed | installé |
| à télécharger ({}){} | to download ({}){} | à télécharger ({}){} |
| à télécharger ({}) · demande {} d'espace, {} libres{} | to download ({}) · needs {} of space, {} free{} | à télécharger ({}) · demande {} d'espace, {} libres{} |
| pas encore publié pour cette version | not published for this version yet | pas encore publié pour cette version |
| GPU inconnu | Unknown GPU | GPU inconnu |
|  · cœurs Tensor |  · Tensor cores |  · cœurs Tensor |
|  · unités matricielles (WMMA) |  · matrix units (WMMA) |  · unités matricielles (WMMA) |
| livré avec le lecteur | bundled with the player | livré avec le lecteur |
| absent de ce paquet | not in this package | absent de ce paquet |
| à télécharger ({} Mo), paquet fermé sous licence NVIDIA | to download ({} MB), closed NVIDIA-licensed package | à télécharger ({} Mo), paquet fermé sous licence NVIDIA |
| pas encore installable dans cette version | cannot be installed in this version yet | pas encore installable dans cette version |
| ni TensorRT ni Vulkan choisis : ni SPAN ni RIFE | neither TensorRT nor Vulkan chosen: no SPAN or RIFE | ni TensorRT ni Vulkan choisis : ni SPAN ni RIFE |
| Recommandé | Recommended | Recommandé |
| Compatibilité | Compatibility | Compatibilité |
| Toujours présent | Always available | Toujours présent |
| Le plus rapide pour ce GPU | Fastest for this GPU | Le plus rapide pour ce GPU |
| RTX HDR sur ce GPU | RTX HDR on this GPU | RTX HDR sur ce GPU |
| À comparer avec Vulkan | Compare with Vulkan | À comparer avec Vulkan |
| Ne convient pas à ce GPU | Does not suit this GPU | Ne convient pas à ce GPU |
| Bienvenue | Welcome | Bienvenue |
| Moteurs | Engines | Moteurs |
| Image et fluidité | Picture and smoothness | Image et fluidité |
| Préréglage | Preset | Préréglage |
| Priorité | Priority | Priorité |
| Votre écran reçoit du HDR. | Your display receives HDR. | Votre écran reçoit du HDR. |
| Automatique | Automatic | Automatique |
| Le lecteur choisit selon la machine. | The player chooses for the machine. | Le lecteur choisit selon la machine. |
| Cartes NVIDIA RTX avec le moteur TensorRT et le SDK Maxine. | NVIDIA RTX cards with the TensorRT engine and the Maxine SDK. | Cartes NVIDIA RTX avec le moteur TensorRT et le SDK Maxine. |
| HDR adaptatif | Adaptive HDR | HDR adaptatif |
| Tous les GPU : l'aspect HDR est émulé selon la scène. | Any GPU: the HDR look is emulated per scene. | Tous les GPU : l'aspect HDR est émulé selon la scène. |
| Pas de conversion HDR. | No HDR conversion. | Pas de conversion HDR. |
| RTX HDR, pas à pas | RTX HDR, step by step | RTX HDR, pas à pas |
| à faire | to do | à faire |
| Vérifier de nouveau | Check again | Vérifier de nouveau |
| Renoncer : HDR adaptatif | Give up: adaptive HDR | Renoncer : HDR adaptatif |
| Comment écoutez-vous ? | How do you listen? | Comment écoutez-vous ? |
| Casque : crossfeed en stéréo, binaural en surround. | Headphones: crossfeed in stereo, binaural in surround. | Casque : crossfeed en stéréo, binaural en surround. |
| Multicanal, surround direct. | Multichannel, direct surround. | Multicanal, surround direct. |
| Grisé : l'appareil de sortie n'a que 2 canaux. | Greyed out: the output device only has 2 channels. | Grisé : l'appareil de sortie n'a que 2 canaux. |
| Enceintes : rendu fidèle, mixage soigné. | Speakers: faithful sound, careful mix. | Enceintes : rendu fidèle, mixage soigné. |
| Langues | Languages | Langues |
| Langue de l'audio | Audio language | Langue audio |
| Selon le fichier | From the file | Selon le fichier |
| Langue des sous-titres | Subtitle language | Langue des sous-titres |
| Aucun | None | Aucun |
| Mises à jour | Updates | Mises à jour |
| Me prévenir | Notify me | Me prévenir |
| Une notification ; vous lancez la mise à jour vous-même. | A notification; you start the update yourself. | Une notification ; vous lancez la mise à jour vous-même. |
| Téléchargée, installée au prochain lancement. | Downloaded, installed at next launch. | Téléchargée, installée au prochain lancement. |
| Jamais | Never | Jamais |
| Aucune vérification. | No checks. | Aucune vérification. |
| Résumé | Summary | Résumé |
| Précédent | Back | Précédent |
| Passer | Skip | Passer |
| Terminer | Finish | Terminer |
| Suivant | Next | Suivant |
| Continuer la lecture | Resume playback | Continuer la lecture |
| Commencer | Start | Commencer |
| Profil de démarrage | Startup profile | Profil de démarrage |
| Annuler | Cancel | Annuler |
| Moteur | Engine | Moteur |
| Pas encore dit | Not said yet | Pas encore dit |
| Casque | Headphones | Casque |
| Enceintes | Speakers | Enceintes |
| Home cinéma | Home theater | Home cinéma |
| Étape {} sur {} : {} | Step {} of {}: {} | Étape {} sur {} : {} |
| HDR : sauté, automatique (HDR et couleurs > HDR) | HDR: skipped, automatic (HDR and color > HDR) | HDR : sauté, automatique (HDR et couleurs > HDR) |
| HDR : {} (HDR et couleurs > HDR) | HDR: {} (HDR and color > HDR) | HDR : {} (HDR et couleurs > HDR) |
| Audio : enceintes en rendu fidèle (Audio > Sortie) | Audio: speakers, faithful sound (Audio > Output) | Audio : enceintes en rendu fidèle (Audio > Sortie) |
| Audio : {} (Audio > Sortie) | Audio: {} (Audio > Output) | Audio : {} (Audio > Sortie) |
| Langues : sauté (Général > Lecture) | Languages: skipped (General > Playback) | Langues : sauté (Général > Lecture) |
| Langues : choisies (Général > Lecture) | Languages: chosen (General > Playback) | Langues : choisies (Général > Lecture) |
| me prévenir | notify me | me prévenir |
| Mises à jour : {} (Général > Application) | Updates: {} (General > Application) | Mises à jour : {} (Général > Application) |
| Carte NVIDIA RTX | NVIDIA RTX card | Carte NVIDIA RTX |
| RTX HDR demande une carte RTX (cœurs Tensor). | RTX HDR needs an RTX card (Tensor cores). | RTX HDR demande une carte RTX (cœurs Tensor). |
| Moteur TensorRT | TensorRT engine | Moteur TensorRT |
| Moteur TensorRT (installé à « Terminer ») | TensorRT engine (installed at "Finish") | Moteur TensorRT (installé à « Terminer ») |
| SDK Maxine VideoFX | Maxine VideoFX SDK | SDK Maxine VideoFX |
| SDK Maxine VideoFX (installé à « Terminer ») | Maxine VideoFX SDK (installed at "Finish") | SDK Maxine VideoFX (installé à « Terminer ») |
| Niveau estimé : {} | Estimated level: {} | Niveau estimé : {} |
| Écran {}p · {:.0f} Hz | Display {}p · {:.0f} Hz | Écran {}p · {:.0f} Hz |
| Chargement du moteur | Loading the engine | Chargement du moteur |
| automatiques | automatic | automatiques |
| jamais | never | jamais |

### Infobulles et notes raccourcies

| Avant (français) | Après (anglais) |
|---|---|
| espace disque insuffisant : il faut {} (dont {} à télécharger), il reste {} | not enough disk space: {} needed ({} to download), {} left |
| non publié (licence NVIDIA) : à construire sur cette machine, voir docs/nvidia-linux.md | not published (NVIDIA licence): build it on this machine, see docs/nvidia-linux.md |
| à installer depuis NVIDIA NGC (clé d'API d'un compte NVIDIA gratuit) | to install from NVIDIA NGC (API key of a free NVIDIA account) |
| NVIDIA Maxine non choisi : HDR adaptatif (tous GPU) au lieu de RTX HDR | NVIDIA Maxine not chosen: adaptive HDR instead of RTX HDR |
| Le lecteur choisit seul la chaîne pour cette machine ; tout se retrouve ensuite dans les paramètres (p). Il s'ajoute aussi à « Ouvrir avec » du système à la fin de l'assistant. | The player picks the settings for this machine; everything is in Settings (p) afterwards. |
| Vulkan sert tous les GPU ; les moteurs cochés à télécharger s'installent après « Terminer », vérifiés par leur empreinte, et Vulkan prend le relais de celui qui échoue. | Vulkan works on every GPU. Checked engines install after "Finish", verified by hash; Vulkan takes over if one fails. |
| Vulkan sert tous les GPU ; les moteurs cochés à télécharger s'installent après « {} », vérifiés par leur empreinte, et Vulkan prend le relais de celui qui échoue. | Vulkan works on every GPU. Checked engines install after "{}", verified by hash; Vulkan takes over if one fails. |
| Auto choisit la chaîne selon votre carte, votre écran et la vidéo. | Auto picks the settings for your card, display and video. |
| Une fois les moteurs installés, le bouton « Optimiser pour ma machine » (Paramètres > Général > Configuration, environ une minute) règle tout seul ces choix pour votre PC. En attendant, une estimation d'après la carte suffit pour commencer. | Once the engines are installed, "Optimize for this machine" (Settings > General > Setup) tunes this for your PC in about a minute. |
| Tant que tout n'est pas fait, l'écran reste en HDR adaptatif ; RTX HDR se choisit ensuite dans HDR et couleurs > HDR. | Until everything is done the display uses adaptive HDR; pick RTX HDR later in HDR and color > HDR. |
| Le format du son se détecte seul. Aucune détection du casque ou des enceintes : dites-le ici. | The sound format is detected. Headphones or speakers are not: tell us here. |
| Passer : Enceintes, en rendu fidèle. Le choix se change ensuite dans Audio > Sortie. | Skip: Speakers, faithful sound. Change it later in Audio > Output. |
| Tout se change ensuite dans les paramètres (p). Pour le refaire : ajp --first-run. | Everything can be changed later in Settings (p). To redo this: ajp --first-run. |
| Les moteurs cochés à télécharger s'installent, puis la lecture reprend au même endroit. | Checked engines install, then playback resumes at the same spot. |
| Le lecteur a choisi pour ce GPU ; tout se change ensuite dans les paramètres (p). | The player chose for this GPU; change anything later in Settings (p). |
| Image et fluidité : sauté, préréglage Auto (Général > Configuration) | Picture and smoothness: skipped, Auto preset (General > Setup) |
| Image et fluidité : {}, priorité {} (Général > Configuration) | Picture and smoothness: {}, priority {} (General > Setup) |
| HDR : RTX HDR choisi mais pas prêt (HDR et couleurs > HDR) : en attendant, l'écran reste en HDR adaptatif | HDR: RTX HDR chosen but not ready (HDR and color > HDR); meanwhile adaptive HDR is used |
| Cochez TensorRT à l'écran Moteur : il s'installe à « Terminer ». | Check TensorRT on the Engine screen: it installs at "Finish". |
| Cochez NVIDIA Maxine à l'écran Moteur (licence NVIDIA), ou posez le SDK dans /usr/local/VideoFX et relancez avec --truehdr-sdk. | Check NVIDIA Maxine on the Engine screen (NVIDIA licence), or put the SDK in /usr/local/VideoFX and relaunch with --truehdr-sdk. |
| {} est choisi mais ne s'ouvre pas sur ce GPU (journal) : {} le remplace. | {} is chosen but does not open on this GPU (see log): {} replaces it. |

## Étape C2 : messages (lecteur par défaut, mises à jour, benchmark, SPAN Performance, accueil, installation des moteurs)

| Avant (français) | Après (anglais) | Après (français) |
|---|---|---|
| registre : HKCU\{} (erreur {}) | registry: HKCU\{} (error {}) | registre : HKCU\{} (erreur {}) |
| registre : HKCU\{} : {} (erreur {}) | registry: HKCU\{} : {} (error {}) | registre : HKCU\{} : {} (erreur {}) |
| lecteur par défaut : dossier personnel inconnu | default player: home folder unknown | lecteur par défaut : dossier personnel inconnu |
| AJP ne figure plus dans « Ouvrir avec ». | AJP is no longer in "Open with". | AJP ne figure plus dans « Ouvrir avec ». |
| AJP n'était pas inscrit : rien à retirer. | AJP was not registered: nothing to remove. | AJP n'était pas inscrit : rien à retirer. |
| curl introuvable : impossible de vérifier les mises à jour | curl not found: cannot check for updates | curl introuvable : impossible de vérifier les mises à jour |
| la signature du manifeste est fausse : mise à jour ignorée | the manifest signature is wrong: update ignored | la signature du manifeste est fausse : mise à jour ignorée |
| le manifeste est du canal « {} », pas « {} » | the manifest is for channel "{}", not "{}" | le manifeste est du canal « {} », pas « {} » |
| {} est disponible | {} is available | {} est disponible |
| Le lecteur est à jour | The player is up to date | Le lecteur est à jour |
| cette release n'a pas d'archive pour ce système | this release has no archive for this system | cette release n'a pas d'archive pour ce système |
| nom d'archive illisible dans le manifeste | unreadable archive name in the manifest | nom d'archive illisible dans le manifeste |
| mis à jour par votre gestionnaire de paquets (Flatpak) | updated by your package manager (Flatpak) | mis à jour par votre gestionnaire de paquets (Flatpak) |
|  (déjà préparée) |  (already prepared) |  (déjà préparée) |
|  : préparée, installée au prochain lancement | : prepared, installed at next launch |  : préparée, installée au prochain lancement |
| tar introuvable | tar not found | tar introuvable |
| aucune mise à jour préparée | no update prepared | aucune mise à jour préparée |
| l'archive préparée a disparu | the prepared archive is gone | l'archive préparée a disparu |
| Mis à jour vers {} | Updated to {} | Mis à jour vers {} |
| aucune version précédente gardée | no previous version kept | aucune version précédente gardée |
| Version précédente remise{} | Previous version restored{} | Version précédente remise{} |
| échec | failed | échec |
| Étape {} sur {} : {} | Step {} of {}: {} | Étape {} sur {} : {} |
| le lecteur de l'essai s'est arrêté brutalement ({}) | the test player stopped abruptly ({}) | le lecteur de l'essai s'est arrêté brutalement ({}) |
| le lecteur de l'essai s'est arrêté ({}) | the test player stopped ({}) | le lecteur de l'essai s'est arrêté ({}) |
| L'essai {} n'a rien rendu : {} | Test {} produced nothing: {} | L'essai {} n'a rien rendu : {} |
| agrandissement ×2 depuis {}p | ×2 upscaling from {}p | agrandissement ×2 depuis {}p |
| image d'origine sans agrandissement | original picture, no upscaling | image d'origine sans agrandissement |
| interpolation jusqu'à ×{}, {} | interpolation up to ×{}, {} | interpolation jusqu'à ×{}, {} |
| sans interpolation | no interpolation | sans interpolation |
| Votre PC tient : {}. | Your PC holds: {}. | Votre PC tient : {}. |
|  Performance FP8 pour SPAN : prête. |  SPAN Performance FP8: ready. |  Performance FP8 pour SPAN : prête. |
|  Performance FP8 pour SPAN : non{}{}. |  SPAN Performance FP8: no{}{}. |  Performance FP8 pour SPAN : non{}{}. |
| bench.toml :  | bench.toml:  | bench.toml :  |
| bench.toml : écriture impossible | bench.toml: cannot write | bench.toml : écriture impossible |
| L'essai {} n'a pas pu démarrer : {} | Test {} could not start: {} | L'essai {} n'a pas pu démarrer : {} |
| des réglages « {} » | of the "{}" settings | des réglages « {} » |
| de préparation | of preparation | de préparation |
| Essai {}/4 : {} | Test {}/4: {} | Essai {}/4 : {} |
| de la variante « {} » | of the "{}" variant | de la variante « {} » |
| comparaison annulée | comparison cancelled | comparaison annulée |
| la variante « {} » n'a rien rendu | the "{}" variant produced nothing | la variante « {} » n'a rien rendu |
| comparaison : une variante n'a rien rendu | comparison: a variant produced nothing | comparaison : une variante n'a rien rendu |
| ne tient pas | does not hold | ne tient pas |
| Carte : {}  | Card: {}  | Carte : {}  |
| Seule « {} » tient.  | Only "{}" holds.  | Seule « {} » tient.  |
| Au p95, « {} » est plus légère de {:.1f} ms par image.  | At p95, "{}" is lighter by {:.1f} ms per frame.  | Au p95, « {} » est plus légère de {:.1f} ms par image.  |
| préparation des moteurs et première mesure | preparing engines and first measurement | préparation des moteurs et première mesure |
| test des réglages « {} » | testing the "{}" settings | test des réglages « {} » |
| évaluation annulée | evaluation cancelled | évaluation annulée |
| recherche d'une image plus fine ({}) | searching for a finer picture ({}) | recherche d'une image plus fine ({}) |
| de recherche ({}) | of search ({}) | de recherche ({}) |
| écriture de la séquence de test | writing the test sequence | écriture de la séquence de test |
| Carte : {} (pilote {}) Niveau estimé : {}  | Card: {} (driver {}) Estimated level: {}  | Carte : {} (pilote {}) Niveau estimé : {}  |
| Niveau gardé : {}  | Level kept: {}  | Niveau gardé : {}  |
| Exploration : la machine tient le modèle {}.  | Exploration: the machine holds the {} model.  | Exploration : la machine tient le modèle {}.  |
| un autre moteur | another engine | un autre moteur |
| cette carte | this card | cette carte |
| Préparation du mode Performance… {} % | Preparing Performance mode… {} % | Préparation du mode Performance… {} % |
| Mode Performance impossible, lecture en FP16 : {} | Performance mode unavailable, playing in FP16: {} | Mode Performance impossible, lecture en FP16 : {} |
| Mode Performance prêt : la lecture le prend maintenant | Performance mode ready: playback uses it now | Mode Performance prêt : la lecture l'utilise maintenant |
| Ouvrir une vidéo | Open a video | Ouvrir une vidéo |
| Choisissez la vidéo dans la fenêtre ouverte. | Pick the video in the open window. | Choisissez la vidéo dans la fenêtre ouverte. |
| ou glissez un fichier sur cette fenêtre. | or drop a file on this window. | ou glissez un fichier sur cette fenêtre. |
| Paramètres (p) | Settings (p) | Paramètres (p) |
| q : quitter | q: quit | q : quitter |
| {} absent : seuls les moteurs livrés sont proposés | {} missing: only bundled engines are offered | {} absent : seuls les moteurs livrés sont proposés |
| écarté, {} | discarded, {} | écarté, {} |
| curl introuvable sur ce système | curl not found on this system | curl introuvable sur ce système |
|  : taille nulle | : zero size |  : taille nulle |
| téléchargement de {} | downloading {} | téléchargement de {} |
| {} : {} octets reçus, {} attendus | {}: {} bytes received, {} expected | {} : {} octets reçus, {} attendus |
| curl ou tar introuvables sur ce système | curl or tar not found on this system | curl ou tar introuvables sur ce système |
| vérification de la place | checking free space | vérification de la place |
| inconnue (tous les fichiers) | unknown (all files) | inconnue (tous les fichiers) |
| téléchargement de l'archive du moteur | downloading the engine archive | téléchargement de l'archive du moteur |
| téléchargement de {} ({}) | downloading {} ({}) | téléchargement de {} ({}) |
| vérification de l'archive | verifying the archive | vérification de l'archive |
| vérification de {} | verifying {} | vérification de {} |
| extraction de l'archive | extracting the archive | extraction de l'archive |
| extraction de {} | extracting {} | extraction de {} |
| paquet {} sans {} | package {} without {} | paquet {} sans {} |
| fichier {} ({}/{}) | file {} ({}/{}) | fichier {} ({}/{}) |
| remplacement de la version installée | replacing the installed version | remplacement de la version installée |
| {} : note du paquet | {}: package note | {} : note du paquet |
| {} installé | {} installed | {} installé |
| démarrage | starting | démarrage |
| téléchargement | download | téléchargement |
| downloading {} | downloading {} | téléchargement de {} |
| decompressing {} | decompressing {} | décompression de {} |

### Messages longs raccourcis

| Avant (français) | Après (anglais) |
|---|---|
| AJP figure dans « Ouvrir avec ». Pour en faire le lecteur des vidéos : « Faire de ce lecteur le lecteur par défaut » dans Paramètres > Général > Application, ou Paramètres > Applications > Applications par défaut. | AJP is in "Open with". To make it the default: "Make this the default player" in Settings > General > Application, or Settings > Apps > Default apps. |
| AJP.app déclare déjà les vidéos qu'il ouvre. Pour en faire le lecteur par défaut : clic droit sur une vidéo > Lire les informations > Ouvrir avec : AJP > Tout modifier. | AJP.app already declares the videos it opens. To make it the default: right-click a video > Get Info > Open with: AJP > Change All. |
| Sous macOS, l'inscription suit AJP.app : la retirer, c'est mettre l'app à la corbeille. | On macOS the registration follows AJP.app: remove it by trashing the app. |
| Paramètres ouverts sur {} : « Définir par défaut » en haut de la page, ou un type à la fois. | Settings opened on {}: "Set default" at the top of the page, or one type at a time. |
| AJP est maintenant le lecteur des vidéos (mkv, mp4, webm, avi, mov…). | AJP is now the video player (mkv, mp4, webm, avi, mov…). |
| xdg-mime introuvable (paquet xdg-utils) : clic droit sur une vidéo > Ouvrir avec > AJP, en cochant « toujours » | xdg-mime not found (xdg-utils package): right-click a video > Open with > AJP and tick "always" |
| ce paquet n'a pas de clé de signature : les mises à jour ne peuvent pas être vérifiées | this package has no signing key: updates cannot be verified |
| sur macOS, la nouvelle version se télécharge sur la page des releases | on macOS, download the new version from the releases page |
| AppImage : la nouvelle version se télécharge sur la page des releases (le fichier se remplace) | AppImage: download the new version from the releases page (replace the file) |
| installé là où le lecteur ne peut pas s'écrire : mis à jour par votre gestionnaire de paquets | installed where the player cannot write: updated by your package manager |
| il a dépassé {} s sans finir, sans doute en construisant un moteur de calcul (c'est long la première fois) | it ran over {} s without finishing, probably building an engine (slow the first time) |
| L'essai {} n'a rien rendu : il s'est terminé sans mesures lisibles{}{} | Test {} produced nothing: it ended without readable measurements{}{} |
| Construction d'un moteur de calcul : la première fois seulement, sans doute plusieurs minutes sur un portable. | Building a compute engine: first time only, probably several minutes on a laptop. |
|   {:<24} {:>5.1f} % perdues, retard p95 {:>6.1f} ms, image {:>6.1f} ms (p95 {:>6.1f}), composition {:>5.1f} ms : {}  |   {:<24} {:>5.1f} % lost, p95 delay {:>6.1f} ms, frame {:>6.1f} ms (p95 {:>6.1f}), compositing {:>5.1f} ms: {}  |
| Si c'est la première fois sur ce PC, la construction des moteurs peut prendre plusieurs minutes. | If this is the first time on this PC, building the engines can take several minutes. |
| Les réglages « {} » n'ont pas été testés : leurs moteurs sont longs à construire. Relancez l'optimisation pour continuer. | The "{}" settings were not tested: their engines take long to build. Run the optimization again to continue. |
| La recherche d'une image plus fine est remise à la prochaine optimisation : les moteurs viennent d'être construits. | The search for a finer picture is left to the next optimization: the engines were just built. |
|   {:<8} {:>5} présentations, {:>5.1f} % perdues, retard p95 {:>6.1f} ms : {}  |   {:<8} {:>5} presentations, {:>5.1f} % lost, p95 delay {:>6.1f} ms: {}  |
| Exploration : la machine tient une hauteur de travail de {}p.  | Exploration: the machine holds a working height of {}p.  |
| le moteur TensorRT est installé mais le lecteur utilise {} : choisissez TensorRT dans Paramètres > Export et moteurs > Moteurs | TensorRT is installed but the player uses {}: pick TensorRT in Settings > Export and engines > Engines |
| moteur TensorRT non installé : installez-le dans Paramètres > Export et moteurs > Moteurs | TensorRT is not installed: install it in Settings > Export and engines > Engines |
| {} n'est pas une carte NVIDIA RTX 40 (Ada) ou le moteur a été construit sans ces noyaux | {} is not an NVIDIA RTX 40 (Ada) card, or the engine was built without these kernels |
| Upscale, interpolation et HDR pour les animes : la chaîne suit le profil de démarrage et les paramètres, réglables ici avant la lecture. | Upscaling, interpolation and HDR for anime: set the startup profile and settings here before playback. |
| Windows 11 24H2 requis (build {} installé) : mettre Windows à jour par Windows Update | Windows 11 24H2 required (build {} installed): update Windows through Windows Update |
| ROCm absent : installer « migraphx » depuis le dépôt ROCm 7 d'AMD (sudo apt install migraphx) | ROCm missing: install "migraphx" from AMD's ROCm 7 repository (sudo apt install migraphx) |
| ROCm 7 requis (ROCm 6 installé) : passer au dépôt ROCm 7 d'AMD, puis sudo apt install migraphx | ROCm 7 required (ROCm 6 installed): switch to AMD's ROCm 7 repository, then sudo apt install migraphx |
| pilote compute Intel absent (sudo apt install intel-opencl-icd libze-intel-gpu1) | Intel compute driver missing (sudo apt install intel-opencl-icd libze-intel-gpu1) |
| espace disque insuffisant : l'installation demande {} (dont {} à télécharger, {} une fois installé), il reste {} sur {} | not enough disk space: the install needs {} ({} to download, {} once installed), {} left on {} |

## Étape C3 : statistiques à l'écran, export, barre de progression, accueil

| Avant (français) | Après (anglais) | Après (français) |
|---|---|---|
| entrée {}x{} · construit en {:.0f} s | input {}x{} · built in {:.0f} s | entrée {}x{} · construit en {:.0f} s |
| Auto ×{} ({:.0f} i/s) · plafond écran ×{} | Auto ×{} ({:.0f} fps) · display limit ×{} | Auto ×{} ({:.0f} i/s) · plafond écran ×{} |
|  · prédit ×{} |  · predicted ×{} |  · prédit ×{} |
|  · scène lourde |  · heavy scene |  · scène lourde |
|  · écart service  |  · service gap  |  · écart service  |
| mémoire vidéo {:.0f} / {:.0f} Mio | video memory {:.0f} / {:.0f} MiB | mémoire vidéo {:.0f} / {:.0f} Mio |
| Vrai noir : coupé. | True black: off. | Vrai noir : coupé. |
| Vrai noir : la rampe ne monte pas, rien n'est changé. | True black: the ramp does not rise, nothing changed. | Vrai noir : la rampe ne monte pas, rien n'est changé. |
| audio absent : l'horloge de l'écran mène | audio missing: the display clock leads | audio absent : l'horloge de l'écran mène |
| image {} (σ {}) · rendu {} · sautées {}{} | frame {} (σ {}) · render {} · skipped {}{} | image {} (σ {}) · rendu {} · sautées {}{} |
|  · calée sur l'écran {:+.3f} % |  · locked to the display {:+.3f} % |  · calée sur l'écran {:+.3f} % |
| fluidité {}{} im/s · manquées {}{} · retard p95 {}{} | smoothness {}{} fps · missed {}{} · p95 delay {}{} | fluidité {}{} im/s · manquées {}{} · retard p95 {}{} |
|  (+ {} coupes et dessins tenus) |  (+ {} cuts and held drawings) |  (+ {} coupes et dessins tenus) |
| Vidéos | Videos | Vidéos |
| Tous les fichiers | All files | Tous les fichiers |
| Dernière optimisation : {} | Last optimization: {} | Dernière optimisation : {} |
| Préparation de RTX HDR | Preparing RTX HDR | Préparation de RTX HDR |
| {} ne calcule pas à un instant libre | {} cannot compute at a free instant | {} ne calcule pas à un instant libre |
| fréquence de l'écran inconnue | display refresh rate unknown | fréquence de l'écran inconnue |
| Export en cours · Échap pour annuler | Exporting · Esc to cancel | Export en cours · Échap pour annuler |
| {} images · {:.1f} i/s · écoulé {} · reste ~{} | {} frames · {:.1f} fps · elapsed {} · left ~{} | {} images · {:.1f} i/s · écoulé {} · reste ~{} |
| {}x{} à {} i/s · {} | {}x{} at {} fps · {} | {}x{} à {} i/s · {} |
| Vidéo ({}) | Video ({}) | Vidéo ({}) |
|  fusionné |  fused |  fusionné |
| sans agrandissement · {} | no upscaling · {} | sans agrandissement · {} |
| pas de RIFE | no RIFE | pas de RIFE |
| Léger | Light | Léger |
| Économe | Economy | Économe |

### Messages longs raccourcis

| Avant (français) | Après (anglais) |
|---|---|
| {} {} : {} paires, {} interpolées, {} dessins tenus, {} coupes, {} en retard | {} {}: {} pairs, {} interpolated, {} held drawings, {} cuts, {} late |
| Vrai noir : sans effet ici (grade coupé, ou sortie HDR sans route). | True black: no effect here (grading off, or HDR output without a path). |
| Vrai noir : 0 sous {:.4f} nit, pied jusqu'à {:.4f} nit (noir de la source à {:.4f} nit avant). | True black: 0 below {:.4f} nit, toe up to {:.4f} nit (source black was {:.4f} nit). |
| Ouvrez une vidéo : le lecteur prépare le mode Performance sur ses images. | Open a video: the player prepares Performance mode on its frames. |
| AJP figure dans « Ouvrir avec ». Pour en faire le lecteur des vidéos : « Faire de ce lecteur le lecteur par défaut » dans Paramètres > Général > Application, ou Paramètres > Applications > Applications par défaut > {}. | AJP is in "Open with". To make it the default: "Make this the default player" in Settings > General > Application, or Settings > Apps > Default apps > {}. |
| AJP figure dans « Ouvrir avec » ({}). Pour en faire le lecteur des vidéos : « Faire de ce lecteur le lecteur par défaut » dans Paramètres > Général > Application, ou xdg-mime default {}{} | AJP is in "Open with" ({}). To make it the default: "Make this the default player" in Settings > General > Application, or xdg-mime default {}{} |

## Ce qui reste en français

Les journaux développeur, l'aide `--help`, les messages d'erreur internes (refus de chargement de modèle, de graphe, de fichier de réglages) et les
textes des paquets de moteurs ne sont pas traduits : ils s'adressent aux développeurs et aux rapports de bogue. Tout texte de menu, d'infobulle,
de notification, d'assistant, de statistique à l'écran et de message de lecteur par défaut ou de mise à jour passe par `tr()`.
