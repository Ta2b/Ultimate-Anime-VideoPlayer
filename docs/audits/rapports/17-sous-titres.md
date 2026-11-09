# Rapport de l'audit 17. Sous-titres

Branche : `claude/project-thread-96lbq8`. Commit audité : `f02b8ae`. Date : `2026-11-09`.

## Verdict

Le module est propre, borné et bien testé sur ce qu'il fait (ASS, SRT, atlas, couleurs VSFilter), mais trois
manques touchent un utilisateur ordinaire : en 4K un seul panneau plein écran bordé suffit à rendre tous les
sous-titres flous jusqu'au prochain écran vide (A17-01, démontré), les sous-titres image (PGS, VobSub) d'un
remux sont ignorés sans un mot (A17-02), et après un seek ou une réouverture la réplique en cours manque
(A17-05).

## Cadrage retenu

Pas de questions posées, à la demande de Tatoub : audit au maximum. Cible retenue : NVIDIA sous Windows 11 et
CachyOS, écran 4K probable (agrandissement ×2 d'une source 1080p), lecture d'anime en remux Blu-ray et en
fansub ASS. Le VRR est hors sujet. RIFE 4.25, RIFE 4.22 lite et DistilDRBA comptent autant : rien dans cette
zone ne favorise l'un ou l'autre. Les sous-titres se composent après l'interpolation, au pts de chaque image de
sortie ; leur coût croît avec le facteur d'interpolation, pas avec le modèle (A17-08). Ce qui ralentit
DistilDRBA est à chercher ailleurs (audits 03, 09, 32).

Méthode : lecture complète des 30 fichiers et des 12 fichiers de test, puis des interfaces avec 13
(`pipeline.cpp`, `session.cpp`), 14 (`selection.cpp`, `source_loop.cpp`) et 18 (`bridges.cpp`, `wiring.cpp`).
Les défauts démontrés l'ont été par un banc local : `src/subs/*.cpp` compilés contre libass 0.17.1 (Ubuntu
24.04, sans le fork : `ass_set_render_thread_count` remplacé par une souche vide) et Catch2 3.4. Le code des
tests est en annexe ; 8 sur 9 échouent sur `f02b8ae`.

## Défauts

| ID | Gravité | Où | Défaut | Scénario | Correction | Preuve |
|---|---|---|---|---|---|---|
| A17-01 | Majeur | `src/subs/subtitle_renderer.cpp:199` et `:240-244` | MED-05. Quand un instant ne tient pas dans l'atlas, tout le rendu passe à ½ puis ¼ de la définition et n'en revient que si l'écran se vide ou si la cible change. Les dialogues qui suivent restent flous, étirés en bilinéaire. En 4K le seuil est bas : `kMaxAtlasSide` vaut 4096 (`subtitle_frame.h:36`), et un dessin plein écran bordé donne deux images de 3840×2160 (remplissage et bordure) que l'empaqueteur ne peut ni empiler (4321 > 4096) ni mettre côte à côte. | Plein écran 4K, OP de fansub : un carton titre en `\p1` bordé sur toute l'image, puis le karaoké continu de l'OP. L'écran ne se vide jamais pendant 90 s : tout le karaoké est rendu en 1920×1080 et étiré. Même chose pour un panneau plein écran au milieu d'une scène dialoguée. | À chaque contenu neuf (`change == 2`) avec `shrink_ > 1`, réessayer d'abord à pleine taille (un rendu de plus, seulement pendant la réduction). Mieux : un atlas à la taille permise par le GPU (16384 sur toute RTX) ou plusieurs atlas par image, la réduction ne servant plus qu'en dernier recours. | Démontré : test « MED-05 » en annexe, la largeur reste 1025 au lieu de 2050 29 s après la fin du dessin. |
| A17-02 | Majeur | `src/subs/subs_messages.h:42` (`TrackSource`), `src/subs/subtitle_frame.h:60-66` | Aucun sous-titre image : PGS, VobSub, DVB ne sont pas lus, ni mov_text ni WebVTT. Le contrat de `SubtitleFrame` (couverture R8 et une couleur par rectangle) ne peut pas porter une image à palette. Ces pistes sont écartées par `subtitle_readable` (`selection.cpp:79-84`) et absentes du menu (`pipeline.cpp:150`) : l'utilisateur n'en sait rien. Pire, leur seule présence empêche le repli sur un fichier externe (renvoi 13). | Remux Blu-ray français (Kazé, @Anime) ou américain (Aniplex USA) : la seule piste de sous-titres est en PGS. Lecture sans sous-titres, sans notice, sans entrée au menu. Si un `.ass` de fansub sans balise de langue est posé à côté, il n'est pas pris non plus. | Ajouter une source `BitmapStream` : décodage par `avcodec_decode_subtitle2` (FFmpeg sait déjà PGS, VobSub, DVB), rectangles RGBA dans un atlas RGBA8 à part, même file et même recyclage. Au minimum : lister ces pistes au menu, grisées, avec la notice « sous-titres image non pris en charge ». | Lu. Test à écrire dans `tests/unit/app/bridges_test.cpp` : `subtitle_track(TrackInfo{.kind = Subtitle, .codec = "hdmv_pgs_subtitle", .parameters = …}, 1)` doit rendre un `SelectTrack` ; aujourd'hui `TrackUnavailable`. |
| A17-03 | Mineur | `src/subs/text_encoding.cpp:212-216`, `:112-115`, `:149-164` | MED-11. Un indice de langue impose son encodage, et s'il échoue on passe directement au Windows-1252 sans essayer la forme des octets. Ni EUC-KR, ni EUC-JP : le coréen est décodé en GBK, qui l'accepte, et sort en sinogrammes. Un EUC-JP sans indice passe aussi en GBK : les kana sont justes, les kanji faux, une erreur qui ne se voit pas tout de suite. L'indice n'est que le dernier suffixe (`subtitle_track.cpp:57-61`) : « .fr.forced », « .chs&jpn », « .zh_TW » n'en donnent aucun, et « ko » n'est pas reconnu. | `Ep01.ko.srt` en EUC-KR : « 救崇窍技夸 » au lieu de « 안녕하세요 ». `Ep01.zh.srt` en Big5 : « §A¦n¡A ». `Ep01.ja.srt` en EUC-JP : « ¤³¤ó¤Ë ». | Indice raté : retomber sur `guessed()`, pas sur CP1252. Ajouter EUC-KR (949 / `CP949`) et EUC-JP (20932 / `EUC-JP`) ; les distinguer par leurs plages (EUC-KR : têtes B0-C8 ; EUC-JP : kana en A4/A5 suivis de A1-F3). Lire l'indice dans tous les suffixes, « ko », « kor », « zh_tw » compris. | Démontré : 5 tests « SRT » de l'annexe échouent avec les sorties citées. |
| A17-04 | Mineur | `src/subs/text_encoding.cpp:95-102`, `:202-211` | UTF-16 sans BOM non reconnu, demi-paires recopiées. Un UTF-16 LE sans BOM passe `valid_utf8` grâce à ses octets nuls (tout en ASCII) et part à libass avec des zéros, ou tombe dans la devinette. Une demi-paire est encodée telle quelle (ED A0 80), un UTF-8 que la RFC 3629 interdit et que le code refuse lui-même deux fonctions plus haut. | SRT exporté en « Unicode » sans BOM par un vieil éditeur : refusé (« horodatage attendu ») ou en charabia. Fichier UTF-16 coupé au milieu d'une paire : octets invalides donnés à libass. | Reconnaître l'UTF-16 sans BOM à ses zéros alternés (un octet sur deux nul sur les 512 premiers). Remplacer une demi-paire par U+FFFD. | Démontré : tests « UTF-16 LE sans BOM » et « demi-paire » de l'annexe. |
| A17-05 | Mineur | `src/subs/subs_messages.h:13-14` (contrat), `src/subs/subtitle_track.cpp:137-161` | Pas de pré-lecture des sous-titres au seek. La piste ne garde que les événements déjà reçus ; après un seek, la source repart de l'image clé (`source_loop.cpp:92-94`) et une réplique commencée avant elle n'arrive jamais. Le cas revient à chaque réouverture de la chaîne (nouveau thread subs, piste vide) et à chaque changement de piste (`pipeline.h:99-101` : seek à la position courante). | Seek au milieu d'une réplique, ou changement d'un réglage qui rouvre la chaîne : la réplique en cours disparaît pour ses 2 à 5 s restantes. Avec des images clés toutes les 10 s (x264 `keyint=240` à 23,976 i/s), jusqu'à 10 s de sous-titres perdus. | Comme le `subtitle_preroll` de mpv : au seek, la source lit les paquets de sous-titres à partir de `cible − N s` (N réglable, 10 s par défaut, ou la durée la plus longue vue dans les Cues). Le plus simple : un second contexte de démultiplexage qui ne lit que la piste de sous-titres (renvoi 14). | Lu. À confirmer : sur un MKV à fansub ASS, seek à mi-réplique (repérée dans Aegisub) ; la ligne n'apparaît pas. |
| A17-06 | Mineur | `src/subs/subtitle_renderer.cpp:221` | MED-09. Un délai négatif lit l'instant `pts + \|délai\|`, mais une piste intégrée n'a que les paquets que la source a déjà lus, soit quelques centaines de ms d'avance (profondeur de l'anneau et du décodage). Au-delà, chaque événement arrive à son heure d'origine, amputé du début. | MKV dont les sous-titres sont en retard d'une seconde : délai réglé à −1000 ms. Chaque ligne paraît 0,7 à 0,8 s après le moment voulu, et disparaît à l'heure. | La source garde la piste de sous-titres `\|délai\|` plus 1 s en avance (même mécanisme que A17-05). Un fichier externe n'est pas touché : il est lu en entier. | Lu (`session.cpp:299-301`, `bridges.cpp:126-135`). À confirmer : délai à −2000 ms sur une piste intégrée. |
| A17-07 | Mineur | `src/subs/subtitle_track.cpp:118-122` | Une cue SRT que `add_packet` refuse (plus de 24 h, plus de 1 Mo de texte) fait refuser tout le fichier, alors que `parse_srt` promet d'écarter une cue fautive et de lire le reste. | SRT de fansub avec une cue de crédits « 00:00:00,000 --> 30:00:00,000 » : aucun sous-titre, refus « paquet de sous-titres : … ». | Écarter la cue avec un avertissement, comme `skipped`, et garder le fichier. Ou borner la durée à `kMaxEventDurationUs`. | Démontré : test « une cue SRT de plus de 24 h » de l'annexe. |
| A17-08 | Mineur | `src/subs/subtitle_renderer.cpp:229-234`, `:279-297`, `:250-254` | Coût par image de sortie. Chaque image de l'anneau (après l'interpolation) demande un rendu : libass est appelé à chaque image tant qu'un sous-titre est visible. Un contenu animé (`\t`, `\move`, `\fad`, karaoké) donne `change == 2` à chaque appel : l'atlas est remis à zéro, recopié et retéléversé à chaque image de sortie. Quand un événement est en cours mais invisible (fondu transparent, `\clip` vide), la fenêtre de calme n'est pas posée et la boucle sur tous les événements tourne à chaque image. Le compteur `libass_calls()` existe mais n'est lu que par les tests. | 4K, 144 Hz, 23,976 i/s interpolé ×6 : un fondu plein écran de 1 s coûte 144 rendus libass et autant d'atlas de 8 Mo recopiés et téléversés (environ 1,2 Go/s). Si le thread subs prend du retard, `SubtitleQueue` sert la réponse précédente : l'animation saccade. Le coût est le même avec RIFE et DistilDRBA. | Exposer `libass_calls()` et le temps de rendu dans les statistiques. Pour l'atlas : ne reconstruire que la zone utilisée (déjà le cas) et ne pas remettre à zéro ce qui sera recouvert. Rendre au pts de la source pour les événements sans animation serait possible mais change le rendu : à mesurer d'abord. | À confirmer : exposer `libass_calls()` dans l'OSD de statistiques, lire l'OP d'un fansub karaoké en 4K à ×6, relever les appels par seconde et le temps du thread subs. |
| A17-09 | Mineur | `src/subs/ass_library.cpp:53-62`, `src/subs/subs_messages.h:54-57`, `src/subs/subs_worker.cpp:75` | Polices jointes : 64 Mo par police mais aucun plafond total. Chaque police existe en trois ou quatre copies pendant l'ouverture (FFmpeg, `attached_fonts`, le message, libass), deux ensuite. Un script à section `[Fonts]` ajoute ses polices à la bibliothèque à chaque `SelectTrack`, sans dédoublonnage (libass ajoute), et recrée le moteur. | Pack BD de fansub chinois avec 120 polices CJK de 15 Mo : environ 1,8 Go par copie, plusieurs Go au pic sur le thread de la fenêtre, `bad_alloc` possible. Alterner dix fois entre deux pistes externes à polices intégrées : dix copies de plus. | Plafond total (par exemple 512 Mo) avec une notice ; envoyer les polices par `shared_ptr<const vector>` plutôt que par copie ; ne recharger la section `[Fonts]` d'un script qu'une fois par bibliothèque. | Lu. À confirmer : relever la mémoire du processus à l'ouverture d'un MKV à 1 Go de polices. |
| A17-10 | Mineur | `src/subs/subs_thread.cpp:50` | `SubsWorker::create` (bibliothèque libass et `ass_set_fonts` avec le fournisseur système) tourne sur le thread qui appelle `start`, c'est-à-dire celui de la fenêtre pendant `Pipeline::open`, à chaque ouverture et réouverture de la chaîne. Avec fontconfig, le premier lancement sans cache peut construire ce cache. | Premier lancement sous CachyOS après l'installation de polices : la fenêtre se fige le temps de la construction du cache fontconfig. Sous Windows, coût de DirectWrite inconnu. | Créer le worker dans le thread subs et rendre le refus par le puits, ou garder une bibliothèque libass vivante d'une ouverture à l'autre. | À confirmer : chronométrer `SubsThread::start` dans le journal sous Windows et sous CachyOS, à froid (`fc-cache -r` puis lancement) et à chaud. |
| A17-11 | Mineur | `src/app/play/external_subtitles.cpp:123-124` | Les fichiers externes sont triés par nom puis coupés à 16 avant tout choix par langue. | Rip avec `Subs/Episode 01/` de 30 fichiers (« 2_English.srt » à « 30_Thai.srt ») : le tri par octets range « 10_ » à « 19_ » avant « 2_ », et la piste française peut tomber au-delà de 16, jamais proposée ni choisie par le préréglage. | Classer d'abord par langue voulue (`config.languages`), puis couper ; ou relever la borne. | Lu. |
| A17-12 | Mineur | `src/subs/subtitle_track.cpp:27`, `:44`, `:48`, `:54`, `:58`, `:130`, `src/subs/subtitle_renderer.cpp:113`, `src/app/play/external_subtitles.cpp:40`, `:77-78`, `:92`, `:99`, `:107`, `:109`, `:114`, `:123` | 16 `path::string()` contre la règle de `CLAUDE.md`. Sous Windows, sans danger tant que le manifeste UTF-8 s'applique ; hors manifeste (tests, outils), exception sur un nom japonais. `default_font.string()` est passé à libass qui attend de l'UTF-8. | Test unitaire Windows de `find_external_subtitles` sur « 進撃の巨人 01.mkv » hors manifeste. | `core::to_utf8`. | Lu. |
| A17-13 | Mineur | `tests/unit/subs/`, `tests/unit/app/external_subtitles_test.cpp` | Rien ne teste le retour à pleine taille après une réduction, les encodages coréen et EUC-JP, l'échec d'un indice, l'UTF-16 sans BOM, une cue SRT hors bornes, un délai négatif sur piste intégrée, la pré-lecture au seek, ni le plafond des polices. Le fuzz (`tests/fuzz/fuzz_parsers.cpp:40-44`) passe `to_utf8` sans indice : la branche des indices et iconv n'est jamais fuzzée. | | Reprendre les tests de l'annexe ; ajouter une cible de fuzz `to_utf8(data, hint)` avec un indice tiré des premiers octets. | Lu. |

## Défauts connus revérifiés

| ID | Statut | Note |
|---|---|---|
| MED-05 | Toujours présent | `subtitle_renderer.cpp:199`, `:240-244`. Démontré, et plus grave qu'annoncé en 4K : voir A17-01. |
| MED-09 | Toujours présent | `subtitle_renderer.cpp:221` ; la cause est la profondeur de lecture de la source. Voir A17-06. |
| MED-11 | Toujours présent | `text_encoding.cpp:212-216`. « zh » impose GBK, un Big5 nommé `.zh` sort en charabia ; pas d'EUC-KR ni d'EUC-JP ; UTF-16 sans BOM non reconnu ; demi-paires recopiées. Voir A17-03 et A17-04. Un Big5 sans indice est bien lu (test passé). |
| WIN-09 | Réfuté en grande partie | glibc iconv a `BIG5` et `CP950`, avec la même table. Sur les 13 000 et quelques codes à deux octets que CP950 accepte, `BIG5` de glibc n'en refuse aucun ; 249 diffèrent (C6A1 à C8FE, kana et symboles ETEN, rendus en zone privée), comme sous Windows. Le charabia vient des indices (A17-03), pas d'iconv. Pour lire les kana ETEN des fansubs taïwanais, essayer `BIG5-HKSCS` (Linux) avant `BIG5`. |
| MED-17 | Toujours présent (hors zone) | Le délai de sous-titres est gardé dans `settings` pour toute la séance (`settings.cpp:420`, `player_run.cpp:219`) : réglé pour l'épisode 1, il s'applique à l'épisode 2. Il n'est pas écrit dans `play.toml`. Renvoi 22. |
| Renvoi 13 (`external_subtitles.cpp:92`) | Toujours présent | Fait partie de A17-12. |

## Renvois

| Vers l'audit | Où | Constat |
|---|---|---|
| 13 | `src/app/play/pipeline.cpp:164-170` | Le repli sur le premier fichier externe n'a lieu que si le conteneur n'a aucune piste de sous-titres, lisible ou non : une piste PGS suffit à l'empêcher (A17-02). Tester `subtitle_readable` plutôt que `kind`. |
| 13 | `src/app/play/pipeline.cpp:183-191` | Chaque police est recopiée deux fois, et l'ouverture dort jusqu'à 1 s par police sur le thread de la fenêtre si la boîte est pleine (A17-09). |
| 13 | `src/app/play/pipeline.cpp:47` | `.SRT` reconnu, `.Srt` étiqueté « ass » : comparer en minuscules. |
| 14 | `src/media/source_loop.cpp:90-106` | Aucune pré-lecture des sous-titres au seek ; la source ne lit pas non plus la piste de sous-titres en avance pour un délai négatif (A17-05, A17-06). |
| 22 | `src/app/play/settings.cpp:420` | MED-17 : délai de sous-titres gardé d'un épisode à l'autre ; le remettre à zéro au changement de fichier, ou le retenir par fichier dans la reprise. |
| 24 | `src/app/play/osd_text.cpp:70` | L'OSD crée une seconde bibliothèque libass avec les polices système : deuxième initialisation de fontconfig ou DirectWrite (voir A17-10). |
| 35 | `tests/fuzz/fuzz_parsers.cpp:40-44` | `to_utf8` fuzzé sans indice : les branches iconv et `MultiByteToWideChar` ne le sont pas (A17-13). |

## Points positifs

- Bornes partout : taille des scripts, des paquets, des polices, des horodatages, des cibles, des demandes en
  vol et des atlas ; `kSubsInboxBounds` cohérent avec `kMaxAtlasesInFlight` et `kMaxSubtitleRequests`.
- Recyclage des atlas par `shared_ptr` et route de retour faible : aucun atlas ne fuit à l'arrêt, testé.
- Fenêtre de calme (P3.39) : aucun appel à libass entre deux répliques, ce qui compte avec l'interpolation.
- Couleurs VSFilter en mode « basic » fidèles à mpv, en-tête `YCbCr Matrix` respecté, testées.
- Placement en bandes noires, pixel non carré et échelle des polices traités et testés ; parité pixel à pixel
  avec libass.
- `SubtitleQueue` règle proprement le clignotement du 19 octobre et fond les réponses identiques.
- La boucle d'un dossier (`external_subtitles.cpp:98-121`) survit à un lien cassé : `increment(error)` remet
  l'erreur à zéro (vérifié).
- Le changement de piste relance un seek (`pipeline.h:99-101`) : les paquets de la nouvelle piste arrivent.

## Questions ouvertes

Aucune pour Tatoub. Le classement de A17-01 en Majeur suppose un écran 4K ; sur un écran 1440p, le même défaut
demande un contenu plus lourd et redevient Mineur. Le classement de A17-02 en Majeur suppose des remux Blu-ray à
sous-titres PGS seulement, fréquents dans les éditions françaises.

## Annexe : tests du banc local

Construits hors du dépôt : `src/subs/*.cpp`, `src/core/log*.cpp`, `src/core/thread_role.cpp` et
`tests/unit/subs/support.cpp`, avec `-std=c++23`, contre libass 0.17.1 et une souche
`ass_set_render_thread_count` vide. `samples.h` contient les mêmes phrases encodées par Python
(`"…".encode('euc_kr')`, `'big5'`, `'euc_jp'`, `'utf-16-le'`). Résultat sur `f02b8ae` : 9 cas, 1 passé
(Big5 sans indice), 8 échoués.

```cpp
TEST_CASE("A17 : SRT coréen EUC-KR sans indice", "[a17]") { CHECK(to_utf8(kEucKr) == kKo); }
TEST_CASE("A17 : SRT coréen EUC-KR, indice ko", "[a17]") { CHECK(to_utf8(kEucKr, "ko") == kKo); }
TEST_CASE("A17 : SRT Big5 nommé .zh", "[a17]") { CHECK(to_utf8(kBig5, "zh") == kTw); }
TEST_CASE("A17 : SRT Big5 sans indice", "[a17]") { CHECK(to_utf8(kBig5) == kTw); }  // passe
TEST_CASE("A17 : SRT japonais EUC-JP nommé .ja", "[a17]") { CHECK(to_utf8(kEucJp, "ja") == kJa); }
TEST_CASE("A17 : SRT japonais EUC-JP sans indice", "[a17]") { CHECK(to_utf8(kEucJp) == kJa); }
TEST_CASE("A17 : UTF-16 LE sans BOM", "[a17]") {
  CHECK(to_utf8(kUtf16LeNoBom).find(kJa) != std::string::npos);
}
TEST_CASE("A17 : demi-paire UTF-16 en UTF-8 invalide", "[a17]") {
  const std::string lone("\xFF\xFE\x00\xD8\x41\x00", 6);  // BOM, D800 seul, 'A'
  CHECK(to_utf8(lone).find("\xED\xA0\x80") == std::string::npos);
}

TEST_CASE("A17 / MED-05 : la réduction reste après la fin du dessin lourd", "[a17]") {
  // Script 100×100 de compat_test.cpp:190, style Default bordé (Outline 2), plus un dialogue continu :
  // "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,{\\pos(0,0)\\p1}m 0 0 l 100 0 100 100 0 100\n"
  // "Dialogue: 1,0:00:00.00,0:01:00.00,Default,,0,0,0,,{\\an2\\bord0\\fs6}Dialogue suivant\n"
  Scene scene(script);  // la Scene de compat_test.cpp
  const RenderTarget target{.width = 2050, .height = 2050};
  CHECK(scene.frame(0.5, target).width == 1025);   // réduit : voulu
  CHECK(scene.frame(30.0, target).width == 2050);  // échoue : 1025
}

TEST_CASE("A17 : une cue SRT de plus de 24 h fait refuser tout le fichier", "[a17]") {
  AssLibrary library = *AssLibrary::create();
  const std::string srt =
      "1\n00:00:01,000 --> 00:00:03,000\nPremière réplique\n\n"
      "2\n00:00:00,000 --> 30:00:00,000\nsubs by Groupe\n\n"
      "3\n00:00:05,000 --> 00:00:07,000\nTroisième réplique\n";
  CHECK(SubtitleTrack::from_srt(library, srt).has_value());  // échoue
}
```

Sorties obtenues : EUC-KR « 救崇窍技夸, 咯矾盒. » ; Big5 indice `zh` « §A¦n¡A§Ú­Ì¤µ¤Ñ » ; EUC-JP indice `ja`
« ¤³¤ó¤Ë¤Á¤Ï » ; EUC-JP sans indice « こんにちは、みなさん。海泣はいい欧丹ですね。 » (attendu « 今日はいい天気 ») ;
UTF-16 sans BOM : texte introuvable ; demi-paire : `ED A0 80` en tête de la sortie.
