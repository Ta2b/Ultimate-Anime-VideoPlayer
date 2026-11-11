#include "media/demuxer.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

extern "C" {
#include <libavcodec/codec_par.h>
#include <libavcodec/defs.h>
#include <libavutil/mathematics.h>
#include <libavutil/mem.h>
}

#include "clips.h"
#include "media/selection.h"

using namespace ajp::media;
using ajp::RefusalCode;

namespace {

Demuxer open_clip(const test::ClipSpec& spec) {
  auto demuxer = Demuxer::open(test::require_clip(spec));
  INFO((demuxer ? std::string() : demuxer.error().detail));
  REQUIRE(demuxer.has_value());
  return std::move(*demuxer);
}

std::string_view as_text(std::span<const std::byte> bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

std::vector<Packet> read_all(Demuxer& demuxer) {
  std::vector<Packet> packets;
  while (true) {
    auto packet = demuxer.read();
    REQUIRE(packet.has_value());
    if (!*packet) return packets;
    packets.push_back(std::move(**packet));
  }
}

}  // namespace

TEST_CASE("Demuxer : pistes avec langue, titre, codec et drapeaux", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::full_clip());
  const auto& tracks = demuxer.tracks();
  REQUIRE(tracks.size() == 6);

  const TrackInfo& video = tracks[0];
  CHECK(video.kind == TrackKind::Video);
  CHECK(video.codec == "ffv1");
  CHECK(video.width == 64);
  CHECK(video.height == 48);
  CHECK(video.frame_rate == Rational{25, 1});

  CHECK(tracks[1].kind == TrackKind::Audio);
  CHECK(tracks[1].codec == "pcm_s16le");
  CHECK(tracks[1].language == "jpn");
  CHECK(tracks[1].title == "Japonais");
  CHECK_FALSE(tracks[1].is_default);
  CHECK(tracks[1].sample_rate == 48000);
  CHECK(tracks[1].channels == 2);
  CHECK(tracks[2].language == "eng");
  CHECK(tracks[2].is_default);

  CHECK(tracks[3].kind == TrackKind::Subtitle);
  CHECK(tracks[3].codec == "ass");
  CHECK(tracks[3].title == "Complets");
  CHECK(tracks[3].is_default);
  CHECK_FALSE(tracks[3].forced);
  CHECK(as_text(tracks[3].parameters->extradata()).starts_with("[Script Info]"));
  CHECK(tracks[4].forced);

  const TrackInfo& font = tracks[5];
  CHECK(font.kind == TrackKind::Attachment);
  CHECK(font.codec == "ttf");
  CHECK(font.filename == "Police.ttf");
  CHECK(font.mime_type == "application/x-truetype-font");
  CHECK(is_font(font));
  CHECK(as_text(font.parameters->extradata()) == test::kFontBytes);
  CHECK_FALSE(is_font(tracks[3]));
}

TEST_CASE("Demuxer : chapitres et durée en microsecondes", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::full_clip());
  const auto& chapters = demuxer.chapters();
  REQUIRE(chapters.size() == 3);
  CHECK(chapters[0].start_us == 0);
  CHECK(chapters[0].end_us == 400'000);
  CHECK(chapters[1].title == "Partie A");
  CHECK(chapters[2].start_us == 800'000);
  CHECK(chapters[2].end_us == 1'200'000);
  CHECK(demuxer.duration_us() == 1'200'000);
  CHECK(demuxer.track(0) != nullptr);
  CHECK(demuxer.track(42) == nullptr);
}

TEST_CASE("Demuxer : chapitres répétés écartés, au plus mille gardés (A35-02)", "[media][demuxer]") {
  test::ClipSpec spec{.name = "chapitres-nombreux.mkv", .frames = 5};
  for (int i = 0; i < 1500; ++i) spec.chapters.push_back({.start_ms = i, .end_ms = i + 1, .title = "c"});
  spec.chapters.push_back({.start_ms = 3, .end_ms = 4, .title = "c"});  // un doublon
  Demuxer demuxer = open_clip(spec);
  CHECK(demuxer.chapters().size() == 1000);
}

TEST_CASE("Demuxer : une taille de vidéo hors de toute source est refusée à l'ouverture (A35-06)",
          "[media][demuxer]") {
  const auto opened =
      Demuxer::open(test::require_clip({.name = "trop-large.mkv", .width = 8320, .height = 16, .frames = 1}));
  REQUIRE_FALSE(opened);
  CHECK(opened.error().code == RefusalCode::FormatUnsupported);
}

TEST_CASE("Demuxer : paquets horodatés en microsecondes, piste par piste", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::full_clip());
  const std::vector<Packet> packets = read_all(demuxer);

  std::map<int, std::vector<const Packet*>> by_track;
  for (const Packet& p : packets) by_track[p.track].push_back(&p);
  REQUIRE(by_track[0].size() == 30);
  for (std::size_t n = 0; n < 30; ++n) {
    CHECK(by_track[0][n]->pts_us == static_cast<std::int64_t>(n) * test::kFrameUs);
    CHECK(by_track[0][n]->keyframe == (n % 10 == 0));  // un point clé toutes les 10 images
    CHECK_FALSE(by_track[0][n]->data().empty());
  }
  REQUIRE(by_track[1].size() == 60);  // 1,2 s de paquets de 20 ms
  CHECK(by_track[1][3]->pts_us == 60'000);
  CHECK(by_track[1][3]->duration_us == 20'000);
  CHECK(by_track[1][3]->data().size() == 3840);

  REQUIRE(by_track[3].size() == 2);
  CHECK(by_track[3][0]->pts_us == 100'000);
  CHECK(by_track[3][0]->duration_us == 500'000);
  CHECK(as_text(by_track[3][1]->data()).ends_with("Au revoir"));
  CHECK(by_track[4].size() == 1);
  CHECK(by_track.count(5) == 0);  // une pièce jointe n'a pas de paquet
}

TEST_CASE("Demuxer : keep_only saute les pistes non lues", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::full_clip());
  const std::vector<int> wanted{0, 3};
  REQUIRE(demuxer.keep_only(wanted));
  const std::vector<Packet> packets = read_all(demuxer);
  CHECK(std::ranges::all_of(packets, [](const Packet& p) { return p.track == 0 || p.track == 3; }));
  CHECK(std::ranges::count(packets, 3, &Packet::track) == 2);

  const std::vector<int> missing{9};
  auto refused = demuxer.keep_only(missing);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::TrackUnavailable);
}

TEST_CASE("Demuxer : choix des pistes du fichier par langues", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::full_clip());
  const TrackChoice ja = choose_tracks(demuxer.tracks(), {.audio = {"ja"}, .subtitles = {"fr"}});
  CHECK(ja.video == 0);
  CHECK(ja.audio == 1);
  CHECK(ja.subtitle == 3);  // complets avant panneaux
  const TrackChoice en = choose_tracks(demuxer.tracks(), {.audio = {"de"}, .subtitles = {"de"}});
  CHECK(en.audio == 2);     // défaut
  CHECK(en.subtitle == 3);  // défaut
}

TEST_CASE("Demuxer : un seek hors du fichier est refusé", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::full_clip());
  auto before = demuxer.seek(-1, 0, SeekTarget::PreviousKeyframe);
  REQUIRE_FALSE(before);
  CHECK(before.error().code == RefusalCode::SeekFailed);
  auto after = demuxer.seek(5'000'000, 0, SeekTarget::PreviousKeyframe);
  REQUIRE_FALSE(after);
  CHECK(after.error().code == RefusalCode::SeekFailed);
  auto wrong_track = demuxer.seek(0, 9, SeekTarget::PreviousKeyframe);
  REQUIRE_FALSE(wrong_track);
  CHECK(wrong_track.error().code == RefusalCode::TrackUnavailable);
}

TEST_CASE("Demuxer : seek sur le point clé précédent ou le plus proche", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::seek_clip());  // points clés aux images 0, 10, 20
  auto first_video_pts = [&] {
    while (true) {
      auto packet = demuxer.read();
      REQUIRE(packet.has_value());
      REQUIRE(packet->has_value());
      if ((*packet)->track == 0) return (*packet)->pts_us;
    }
  };
  REQUIRE(demuxer.seek(17 * test::kFrameUs, 0, SeekTarget::PreviousKeyframe));
  CHECK(first_video_pts() == 10 * test::kFrameUs);
  // L'index du MKV (Cues, chargées par le premier seek) donne le point clé d'avant : le fichier
  // n'est repositionné qu'une fois quand celui d'après gagne, deux fois sinon.
  std::uint32_t seeks = demuxer.file_seeks();
  REQUIRE(demuxer.seek(18 * test::kFrameUs, 0, SeekTarget::NearestKeyframe));
  CHECK(first_video_pts() == 20 * test::kFrameUs);
  CHECK(demuxer.file_seeks() - seeks == 1);
  seeks = demuxer.file_seeks();
  REQUIRE(demuxer.seek(12 * test::kFrameUs, 0, SeekTarget::NearestKeyframe));
  CHECK(first_video_pts() == 10 * test::kFrameUs);
  CHECK(demuxer.file_seeks() - seeks == 2);
  // À égale distance, le point clé d'avant, comme sans index.
  REQUIRE(demuxer.seek(15 * test::kFrameUs, 0, SeekTarget::NearestKeyframe));
  CHECK(first_video_pts() == 10 * test::kFrameUs);
}

TEST_CASE("Demuxer : le seek exact recule sous un point clé affiché après la cible (MP4, images B)",
          "[media][demuxer]") {
  // Index en dts : la cible 10 (pts 417 ms) tombe après le dts du point clé 12
  // (375 ms) mais avant son pts (500 ms) ; les images B 10 et 11 dépendent de
  // l'image 9. Le seek doit repartir du point clé 0.
  Demuxer demuxer = open_clip(test::bframes_clip());
  const std::int64_t target = av_rescale(10 * 1001, 1'000'000, 24000);
  REQUIRE(demuxer.seek(target, 0, SeekTarget::PreviousKeyframe));
  auto first = demuxer.read();
  REQUIRE(first.has_value());
  REQUIRE(first->has_value());
  CHECK((*first)->keyframe);
  CHECK((*first)->pts_us == 0);

  // Les paquets lus pour vérifier le point clé reviennent tous, dans l'ordre du fichier.
  std::vector<std::int64_t> after_seek{(*first)->dts_us};
  for (const Packet& p : read_all(demuxer)) after_seek.push_back(p.dts_us);
  Demuxer fresh = open_clip(test::bframes_clip());
  std::vector<std::int64_t> whole;
  for (const Packet& p : read_all(fresh)) whole.push_back(p.dts_us);
  CHECK(after_seek == whole);

  // Le seek rapide, lui, garde le point clé le plus proche.
  REQUIRE(demuxer.seek(target, 0, SeekTarget::NearestKeyframe));
  auto nearest = demuxer.read();
  REQUIRE(nearest.has_value());
  REQUIRE(nearest->has_value());
  CHECK((*nearest)->pts_us == av_rescale(12 * 1001, 1'000'000, 24000));
}

TEST_CASE("Demuxer : avant la première image vidéo, le seek rend le premier point clé", "[media][demuxer]") {
  // L'audio part de 0, la vidéo de l'image 5 : le premier point clé s'affiche
  // après la cible, et reculer encore ne trouve rien de plus tôt.
  test::ClipSpec spec = test::late_video_clip();
  spec.container = GENERATE(as<std::string>{}, "matroska", "mp4");
  spec.name = spec.container == "mp4" ? "late_video.mp4" : "late_video.mkv";
  Demuxer demuxer = open_clip(spec);
  for (const SeekTarget where : {SeekTarget::PreviousKeyframe, SeekTarget::NearestKeyframe}) {
    for (const std::int64_t target : {std::int64_t{0}, 2 * test::kFrameUs}) {
      INFO(spec.container << ", cible " << target << " µs");
      REQUIRE(demuxer.seek(target, 0, where));
      std::optional<Packet> video;
      while (!video) {
        auto packet = demuxer.read();
        REQUIRE(packet.has_value());
        REQUIRE(packet->has_value());
        if ((*packet)->track == 0) video = std::move(**packet);
      }
      CHECK(video->keyframe);
      CHECK(video->pts_us == 5 * test::kFrameUs);
    }
  }
}

TEST_CASE("Demuxer : un chemin non ASCII s'ouvre (UTF-8)", "[media][demuxer]") {
  test::ClipSpec spec = test::bframes_clip();
  spec.name = "アニメ 第1話.mp4";
  Demuxer demuxer = open_clip(spec);
  CHECK(demuxer.tracks().size() == 1);
}

TEST_CASE("is_font : type MIME de police, ou extension, avec nom et données", "[media][demuxer]") {
  AVCodecParameters* raw = avcodec_parameters_alloc();
  REQUIRE(raw != nullptr);
  raw->extradata = static_cast<std::uint8_t*>(av_mallocz(4 + AV_INPUT_BUFFER_PADDING_SIZE));
  raw->extradata_size = 4;
  auto with_data = CodecParameters::copy_of(*raw);
  avcodec_parameters_free(&raw);
  REQUIRE(with_data.has_value());
  const auto font = [&](std::string filename, std::string mime) {
    return is_font(TrackInfo{.kind = TrackKind::Attachment,
                             .filename = std::move(filename),
                             .mime_type = std::move(mime),
                             .parameters = *with_data});
  };
  CHECK(font("a.bin", "application/x-font"));  // liste de mpv, même « probablement faux »
  CHECK(font("a.bin", "font/collection"));
  CHECK(font("Police.OTC", "application/octet-stream"));  // MKV mal étiqueté
  CHECK_FALSE(font("notes.txt", "text/plain"));
  CHECK_FALSE(font("", "font/ttf"));  // libass exige un nom
  CHECK_FALSE(
      is_font(TrackInfo{.kind = TrackKind::Attachment, .filename = "a.ttf", .mime_type = "font/ttf"}));
  CHECK_FALSE(is_font(TrackInfo{.kind = TrackKind::Subtitle, .filename = "a.ttf", .parameters = *with_data}));
}

TEST_CASE("Demuxer : un fichier absent ou illisible est refusé", "[media][demuxer]") {
  auto missing = Demuxer::open("/nonexistent/clip.mkv");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::MediaUnreadable);
  CHECK(missing.error().detail.find("clip.mkv") != std::string::npos);
  // Le refus se lit en français : « introuvable », pas seulement le texte de FFmpeg.
  CHECK(missing.error().detail.find("introuvable") != std::string::npos);
}

TEST_CASE("Demuxer : une piste entrelacée à 29,97 sort à 23,976 par l'IVTC (A15-02)", "[media][demuxer]") {
  struct Case {
    const char* name;
    AVRational rate;
    AVFieldOrder order;
    bool telecine;
  };
  const Case c = GENERATE(Case{"tff_2997.mkv", {30000, 1001}, AV_FIELD_TT, true},
                          Case{"progressive_2997.mkv", {30000, 1001}, AV_FIELD_PROGRESSIVE, false},
                          Case{"tff_25.mkv", {25, 1}, AV_FIELD_TT, false});
  CAPTURE(c.name);
  test::ClipSpec spec = test::seek_clip();
  spec.name = c.name;
  spec.subtitles.clear();
  spec.frame_rate = c.rate;
  spec.field_order = c.order;
  Demuxer demuxer = open_clip(spec);
  const TrackInfo* video = demuxer.track(0);
  REQUIRE(video != nullptr);
  CHECK(video->inverse_telecine == c.telecine);
  if (c.telecine) {
    CHECK(video->frame_rate == Rational{24000, 1001});
  } else {
    CHECK(std::int64_t{video->frame_rate.num} * c.rate.den ==
          std::int64_t{video->frame_rate.den} * c.rate.num);
  }
}

#ifndef _WIN32
namespace {

struct ChangeDirectory {
  std::filesystem::path before = std::filesystem::current_path();
  explicit ChangeDirectory(const std::filesystem::path& dir) { std::filesystem::current_path(dir); }
  ~ChangeDirectory() { std::filesystem::current_path(before); }
};

}  // namespace

// B14-01, B35-06 : FFmpeg lit « Re: » comme un protocole ; le chemin est absolu et précédé de « file: ».
TEST_CASE("Demuxer : un chemin relatif contenant « : » s'ouvre", "[media][demuxer]") {
  const std::filesystem::path clip = test::require_clip(test::seek_clip());
  const std::filesystem::path dir = clip.parent_path() / "deux-points";
  std::filesystem::create_directories(dir);
  std::filesystem::copy_file(clip, dir / "Re:Zero - 01.mkv",
                             std::filesystem::copy_options::overwrite_existing);
  const ChangeDirectory here(dir);
  auto demuxer = Demuxer::open("Re:Zero - 01.mkv");
  INFO((demuxer ? std::string() : demuxer.error().detail));
  REQUIRE(demuxer.has_value());
  CHECK_FALSE(demuxer->tracks().empty());
  auto url = Demuxer::open("http://exemple.invalide/x.mkv");  // pas une URL : un nom de fichier introuvable
  CHECK_FALSE(url);
}
#endif

// B14-04, B35-04 : une piste vidéo annexe hors des bornes est écartée, le fichier reste lisible.
TEST_CASE("Demuxer : une piste vidéo annexe hors des bornes est écartée, pas le fichier",
          "[media][demuxer]") {
  test::ClipSpec spec{.name = "video-annexe.mkv", .frames = 5};
  spec.extra_videos.emplace_back(16384, 64);
  Demuxer demuxer = open_clip(spec);
  int videos = 0;
  for (const TrackInfo& track : demuxer.tracks()) videos += track.kind == TrackKind::Video ? 1 : 0;
  CHECK(videos == 1);
  CHECK(demuxer.track(0) != nullptr);
  CHECK(demuxer.track(1) == nullptr);
}

// B35-05 : titres coupés à 256 octets UTF-8 entiers, caractères de contrôle remplacés.
TEST_CASE("Demuxer : les titres de pistes sont bornés et nettoyés", "[media][demuxer]") {
  std::string long_title;
  for (int i = 0; i < 400; ++i) long_title += "é";  // 800 octets
  test::ClipSpec spec{.name = "titres-hostiles.mkv", .frames = 5};
  spec.audio.push_back({.language = "jpn", .title = long_title});
  spec.audio.push_back({.language = "eng", .title = "Ligne 1\nLigne 2\t##x"});
  Demuxer demuxer = open_clip(spec);
  std::vector<std::string> titles;
  for (const TrackInfo& track : demuxer.tracks())
    if (track.kind == TrackKind::Audio) titles.push_back(track.title);
  REQUIRE(titles.size() == 2);
  CHECK(titles[0].size() <= 256);
  CHECK(titles[0].size() % 2 == 0);  // pas au milieu d'un « é » (2 octets)
  CHECK(titles[1] == "Ligne 1 Ligne 2 ##x");
}

// B35-01 : seules les polices sont recopiées ; les lectures annexes n'en décrivent aucune.
TEST_CASE("Demuxer : les pièces jointes qui ne sont pas des polices ne sont pas recopiées",
          "[media][demuxer]") {
  test::ClipSpec spec{.name = "pieces-jointes.mkv", .frames = 5};
  spec.attachments.push_back({.filename = "Police.ttf", .mime_type = "font/ttf", .data = test::kFontBytes});
  spec.attachments.push_back(
      {.filename = "notes.txt", .mime_type = "text/plain", .data = std::string(4096, 'x')});
  const auto path = test::require_clip(spec);
  auto full = Demuxer::open(path);
  REQUIRE(full.has_value());
  int fonts = 0;
  int others = 0;
  for (const TrackInfo& track : full->tracks()) {
    if (track.kind != TrackKind::Attachment) continue;
    if (is_font(track)) {
      ++fonts;
    } else {
      ++others;
      CHECK(track.parameters == nullptr);
    }
  }
  CHECK(fonts == 1);
  CHECK(others == 1);
  auto bare = Demuxer::open(path, {.attachments = false});
  REQUIRE(bare.has_value());
  for (const TrackInfo& track : bare->tracks()) CHECK(track.kind != TrackKind::Attachment);
}

// B14-05 : un MKV dont l'en-tête annonce moins que le contenu (Duration réécrite à 300 ms sur 1,2 s).
TEST_CASE("Demuxer : un seek au-delà de la durée annoncée, mais dans le fichier, aboutit",
          "[media][demuxer]") {
  const std::filesystem::path clip = test::require_clip(test::seek_clip());
  std::ifstream in(clip, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::string marker("\x44\x89\x88", 3);  // Duration, un double de 8 octets
  const std::size_t at = bytes.find(marker);
  REQUIRE(at != std::string::npos);
  const double short_duration = 300.0;  // en ms : l'échelle de temps Matroska par défaut
  std::uint64_t bits = 0;
  std::memcpy(&bits, &short_duration, sizeof bits);
  for (int i = 0; i < 8; ++i)
    bytes[at + 3 + static_cast<std::size_t>(i)] = static_cast<char>(bits >> (56 - 8 * i));
  const std::filesystem::path patched = clip.parent_path() / "duree-courte.mkv";
  std::ofstream(patched, std::ios::binary | std::ios::trunc)
      .write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  Demuxer demuxer = [&] {
    auto opened = Demuxer::open(patched);
    REQUIRE(opened.has_value());
    return std::move(*opened);
  }();
  REQUIRE(demuxer.duration_us() == 300'000);
  CHECK(demuxer.seek(800'000, 0, SeekTarget::PreviousKeyframe).has_value());
  const auto beyond =
      demuxer.seek(60'000'000, 0, SeekTarget::PreviousKeyframe);  // au-delà de tout le fichier
  REQUIRE_FALSE(beyond);
  CHECK(beyond.error().code == RefusalCode::SeekFailed);
}

// B31-11 : une panne d'E/S de quelques secondes ne finit plus la vidéo (5 essais, 1,5 s, avant).
TEST_CASE("Demuxer : six erreurs d'E/S de suite sont encore retentées", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::seek_clip());
  demuxer.fail_reads({.after_packets = 3, .count = 6, .error = AVERROR(EIO)});
  std::size_t packets = 0;
  while (true) {
    auto packet = demuxer.read();
    REQUIRE(packet.has_value());
    if (!*packet) break;
    ++packets;
  }
  CHECK(packets > 6);
}

TEST_CASE("Demuxer : une panne d'E/S qui dépasse le budget met fin à la lecture", "[media][demuxer]") {
  Demuxer demuxer = open_clip(test::seek_clip());
  demuxer.fail_reads({.after_packets = 3, .count = 1000, .error = AVERROR(EIO), .io_budget_ms = 100});
  while (true) {
    auto packet = demuxer.read();
    if (!packet) {
      CHECK(packet.error().code == RefusalCode::IoFailed);
      CHECK(packet.error().recoverable);
      break;
    }
    REQUIRE(packet->has_value());
  }
}

// B14-07 : la luminance constante n'est pas une matrice linéaire ; elle n'est pas lue comme du NCL.
TEST_CASE("Demuxer : BT.2020 à luminance constante n'est pas lue comme NCL", "[media][demuxer]") {
  test::ClipSpec spec{.name = "bt2020-cl.mkv", .frames = 5};
  spec.color.matrix = AVCOL_SPC_BT2020_CL;
  Demuxer constant_luminance = open_clip(spec);
  CHECK_FALSE(constant_luminance.tracks()[0].color_matrix.has_value());
  spec.name = "bt2020-ncl.mkv";
  spec.color.matrix = AVCOL_SPC_BT2020_NCL;
  Demuxer non_constant = open_clip(spec);
  REQUIRE(non_constant.tracks()[0].color_matrix.has_value());
  CHECK(*non_constant.tracks()[0].color_matrix == ajp::ColorMatrix::Bt2020Ncl);
}
