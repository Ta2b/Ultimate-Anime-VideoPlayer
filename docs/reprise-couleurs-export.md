# Reprise : couleurs fausses des exports gradés (en pause, novembre 2026)

Chantier mis en pause à la demande de Tatoub (l'export compte peu, la lecture en temps réel prime). Rien n'est
vérifié : ni construit, ni testé. Ne pas fusionner sans avoir lancé le test ci-dessous.

## Symptôme

Export 4K HDR10 (FFV1 comme HEVC) de main e52f6d26 : fond gris-bleu (53, 65, 80) devenu brun-orangé (Y/Cb/Cr
10 bits 292/303/545), cheveux lavande devenus magenta. La capture --snapshot de la lecture est juste.

## Cause (lue dans les sources, pas encore mesurée)

libplacebo lance les hooks `PL_HOOK_OUTPUT` après `pl_shader_encode_color` (`pass_output_target`,
`src/renderer.c` de libplacebo v7.387). Les passes du grade sont sur OUTPUT. À l'écran la cible est en RVB,
l'encodage ne change rien ; dans `FrameExport::render` la cible est en YCbCr (P010, NV12 ou trois plans), et les
passes du grade traitaient Y, Cb, Cr comme R, V, B. La lecture n'est pas touchée.

## Correction sur cette branche

`src/app/play/placebo_export_grade.cpp` : `open_grade` déplacé là, avec une image RVB (`rgba16`, sinon
`rgba16f`) à la taille de sortie ; `render_graded` rend la mise à l'échelle, Anime4K et le grade dans cette
image, puis l'encode en YUV dans un second `pl_render_image` sans hook. Coût : une passe 4K de plus par image.

## Reste à faire

1. Construire (`linux-prefix`) et lancer `kirari_app_tests "Bout en bout : l'export HDR10 gradé garde la teinte
   de la source"` sous `xvfb-run` (`tests/unit/app/e2e_export_test.cpp`, caché `[.e2e]`).
2. Le voir échouer sans la correction (revenir à main pour `src/app/play`), passer avec. Ajuster les seuils
   (teinte à 20°) si le grade tourne un peu les couleurs.
3. Étapes de rules.yml (clang-format-20, tailles, getenv), i18n_test (nouveau journal français dans
   `placebo_export_grade.cpp` : à lister s'il le faut), puis PR auditée.
4. Seulement ensuite, sur le PC : export de 2 s, `ffmpeg -i x.mkv -vf scale=640:-2 -frames:v 1 x.png`,
   comparé à la source.
