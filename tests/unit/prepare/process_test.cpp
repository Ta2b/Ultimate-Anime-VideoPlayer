// Lanceur de processus, avec l'exécutable d'aide (helper.cpp) comme enfant.
#include "prepare/process.h"

#include <chrono>
#include <csignal>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;
using namespace std::chrono_literals;

namespace {

struct Run {
  std::expected<ProcessExit, ajp::Refusal> exit;
  std::vector<std::string> lines;
};

Run run_helper(std::vector<std::string> args, std::chrono::milliseconds timeout = 10s,
               const std::stop_token& stop = {}) {
  Run run{.exit = {}, .lines = {}};
  run.exit = run_process(
      {.program = test::helper_path(), .args = std::move(args), .timeout = timeout},
      [&](std::string_view line) { run.lines.emplace_back(line); }, stop);
  return run;
}

// Découpe de CommandLineToArgvW (hors argv[0]), pour vérifier windows_command_line.
std::vector<std::string> windows_split(std::string_view line) {
  std::vector<std::string> args;
  std::string current;
  bool quoted = false;
  bool any = false;
  for (std::size_t i = 0; i < line.size();) {
    const char c = line[i];
    if (c == '\\') {
      std::size_t n = 0;
      for (; i < line.size() && line[i] == '\\'; ++i) ++n;
      if (i < line.size() && line[i] == '"') {
        current.append(n / 2, '\\');
        if (n % 2 == 1) {
          current += '"';
          ++i;
        }
      } else {
        current.append(n, '\\');
      }
      any = true;
      continue;
    }
    if (c == '"') {
      quoted = !quoted;
      any = true;
    } else if ((c == ' ' || c == '\t') && !quoted) {
      if (any) args.push_back(current);
      current.clear();
      any = false;
    } else {
      current += c;
      any = true;
    }
    ++i;
  }
  if (any) args.push_back(current);
  return args;
}

}  // namespace

TEST_CASE("run_process : chaque ligne, puis le code de sortie", "[prepare][process]") {
  const Run run = run_helper({"lines", "3"});
  REQUIRE(run.exit);
  CHECK(run.exit->kind == ExitKind::Exited);
  CHECK(run.exit->code == 0);
  CHECK(run.lines == std::vector<std::string>{"ligne 1", "ligne 2", "ligne 3"});

  const Run failed = run_helper({"exit", "7"});
  REQUIRE(failed.exit);
  CHECK(failed.exit->kind == ExitKind::Exited);
  CHECK(failed.exit->code == 7);
}

TEST_CASE("run_process : stdout et stderr mêlés, \\r coupe les lignes", "[prepare][process]") {
  const Run streams = run_helper({"streams"});
  REQUIRE(streams.exit);
  CHECK(streams.lines == std::vector<std::string>{"sur stderr", "sur stdout"});
  const Run progress = run_helper({"progress"});
  REQUIRE(progress.exit);
  CHECK(progress.lines == std::vector<std::string>{"10%", "20%", "30%", "fin sans retour"});
}

TEST_CASE("run_process : les arguments arrivent intacts, sans shell", "[prepare][process]") {
  const std::vector<std::string> args{"un mot", "guillemet \" dedans", "é ü 日本", "$HOME", "*", ""};
  std::vector<std::string> with_mode{"args"};
  with_mode.insert(with_mode.end(), args.begin(), args.end());
  const Run run = run_helper(with_mode);
  REQUIRE(run.exit);
  REQUIRE(run.lines.size() == args.size());
  for (std::size_t i = 0; i < args.size(); ++i) CHECK(run.lines[i] == "[" + args[i] + "]");
}

TEST_CASE("run_process : un plantage de l'enfant devient un résultat", "[prepare][process]") {
  const Run run = run_helper({"crash"});
  REQUIRE(run.exit);
  CHECK(run.exit->kind == ExitKind::Crashed);
#ifndef _WIN32
  CHECK(run.exit->code == SIGABRT);
#endif
  CHECK(run.lines == std::vector<std::string>{"avant plantage"});
}

TEST_CASE("run_process : délai dépassé", "[prepare][process]") {
  const auto start = std::chrono::steady_clock::now();
  const Run run = run_helper({"hang"}, 300ms);
  REQUIRE(run.exit);
  CHECK(run.exit->kind == ExitKind::TimedOut);
  CHECK(run.exit->elapsed >= 300ms);
  CHECK(std::chrono::steady_clock::now() - start < 10s);
  CHECK(run.lines == std::vector<std::string>{"en attente"});
}

TEST_CASE("run_process : annulation pendant que l'enfant tourne", "[prepare][process]") {
  std::stop_source stop;
  std::jthread canceller([&] {
    std::this_thread::sleep_for(200ms);
    stop.request_stop();
  });
  const Run run = run_helper({"hang"}, 30s, stop.get_token());
  REQUIRE(run.exit);
  CHECK(run.exit->kind == ExitKind::Cancelled);
  CHECK(run.exit->elapsed < 10s);
}

TEST_CASE("run_process : refus d'un exécutable absent ou d'un délai nul", "[prepare][process]") {
  const test::TempDir dir;
  const auto missing = run_process({.program = dir / "absent", .args = {}, .timeout = 1s}, {}, {});
  CHECK(test::refusal(missing) == RefusalCode::BuilderFailed);
  const auto no_timeout = run_process({.program = test::helper_path(), .args = {}, .timeout = 0ms}, {}, {});
  CHECK(test::refusal(no_timeout) == RefusalCode::FieldOutOfRange);
}

#ifdef __linux__
TEST_CASE("run_process : l'annulation tue aussi les petits-enfants", "[prepare][process]") {
  std::stop_source stop;
  std::string grandchild;
  const auto exit = run_process(
      {.program = "/bin/sh", .args = {"-c", "sleep 30 & echo $!; wait"}, .timeout = 30s},
      [&](std::string_view line) {
        grandchild = line;
        stop.request_stop();
      },
      stop.get_token());
  REQUIRE(exit);
  CHECK(exit->kind == ExitKind::Cancelled);
  REQUIRE_FALSE(grandchild.empty());
  // Mort, ou zombie en attente de son nouveau parent : dans les deux cas il ne tourne plus.
  const auto proc = std::filesystem::path("/proc") / grandchild / "stat";
  bool running = true;
  for (int i = 0; i < 100 && running; ++i) {
    const std::string stat = test::read_file(proc);
    const auto state = stat.find(") ");
    running = !stat.empty() && state != std::string::npos && stat[state + 2] != 'Z';
    if (running) std::this_thread::sleep_for(20ms);
  }
  CHECK_FALSE(running);
}
#endif

TEST_CASE("LineSplitter : morceaux, fins de ligne mêlées, lignes trop longues", "[prepare][process]") {
  std::vector<std::string> lines;
  LineSplitter splitter([&](std::string_view line) { lines.emplace_back(line); }, 4);
  splitter.feed("ab");
  splitter.feed("\r\ncd\n\nabcdef");
  splitter.feed("g");
  splitter.finish();
  CHECK(lines == std::vector<std::string>{"ab", "cd", "abcd", "efg"});
}

TEST_CASE("windows_command_line : CommandLineToArgvW redonne chaque argument", "[prepare][process]") {
  const std::vector<std::string> args{
      "simple",   "deux mots", "", "a\"b", R"(C:\chemin\)", R"(C:\avec espace\)", R"(\\serveur\partage)",
      R"(a\\"b)", "tab\tici"};
  std::vector<std::string> argv{"trtexec.exe"};
  argv.insert(argv.end(), args.begin(), args.end());
  const std::string line = windows_command_line(argv);
  std::vector<std::string> parsed = windows_split(line);
  REQUIRE_FALSE(parsed.empty());
  parsed.erase(parsed.begin());
  CHECK(parsed == args);
  CHECK(windows_command_line({"a b", "c"}) == R"("a b" c)");
  CHECK(windows_command_line({R"(x\)", R"(y z\)"}) == R"(x\ "y z\\")");
}
