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

## À venir

B2 : Image IA (Agrandissement, Fluidité). B3 : Audio. B4 : Export, Moteurs, Application, Configuration (benchmark). C : assistant, écran des moteurs,
lecteur par défaut, mises à jour, messages d'erreur visibles.
