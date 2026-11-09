// Les messages de FFmpeg dans le journal (A39-05) : sans cela ils partent sur stderr, perdus sous Windows
// (exécutable fenêtré) et absents du rapport. Seules les erreurs passent (blocs verts, HEVC abîmé, refus de
// NVDEC) ; une même ligne répétée est comptée, pas recopiée.
#pragma once

namespace ajp::media {

// À appeler une fois, après l'installation des puits du journal.
void route_ffmpeg_log_to_journal();

}  // namespace ajp::media
