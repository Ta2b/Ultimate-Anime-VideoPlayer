#include "http_server.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace ajp::app::test {
namespace {

#if defined(_WIN32)
using Socket = SOCKET;
using Length = int;  // longueur des tampons de send et recv
constexpr Socket kInvalid = INVALID_SOCKET;
constexpr int kSendFlags = 0;
void close_socket(Socket s) {
  closesocket(s);
}
int poll_one(Socket s, int milliseconds) {
  WSAPOLLFD fd{.fd = s, .events = POLLRDNORM, .revents = 0};
  return WSAPoll(&fd, 1, milliseconds);
}
struct Network {
  Network() {
    WSADATA data{};
    WSAStartup(MAKEWORD(2, 2), &data);
  }
  ~Network() { WSACleanup(); }
};
#else
using Socket = int;
using Length = std::size_t;
constexpr Socket kInvalid = -1;
#if defined(MSG_NOSIGNAL)
constexpr int kSendFlags = MSG_NOSIGNAL;  // un client parti ne tue pas le test par SIGPIPE
#else
constexpr int kSendFlags = 0;
#endif
void close_socket(Socket s) {
  ::close(s);
}
int poll_one(Socket s, int milliseconds) {
  pollfd fd{.fd = s, .events = POLLIN, .revents = 0};
  return ::poll(&fd, 1, milliseconds);
}
struct Network {};
#endif

void ensure_network() {
  [[maybe_unused]] static const Network network;
}

Socket to_socket(std::intptr_t value) {
  return static_cast<Socket>(value);
}

// Socket qui écoute sur 127.0.0.1, port choisi par le système.
Socket listen_local(int& port) {
  ensure_network();
  const Socket s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s == kInvalid) throw std::runtime_error("socket");
#if defined(__APPLE__)
  const int one = 1;
  setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = 0;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(s, reinterpret_cast<const sockaddr*>(&address), static_cast<socklen_t>(sizeof address)) != 0 ||
      ::listen(s, 8) != 0) {
    close_socket(s);
    throw std::runtime_error("bind ou listen sur 127.0.0.1");
  }
  auto length = static_cast<socklen_t>(sizeof address);
  ::getsockname(s, reinterpret_cast<sockaddr*>(&address), &length);
  port = ntohs(address.sin_port);
  return s;
}

bool send_all(Socket s, std::string_view data) {
  while (!data.empty()) {
    const auto sent =
        ::send(s, data.data(), static_cast<Length>(std::min<std::size_t>(data.size(), 1 << 16)), kSendFlags);
    if (sent <= 0) return false;
    data.remove_prefix(static_cast<std::size_t>(sent));
  }
  return true;
}

std::string_view reason(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 206:
      return "Partial Content";
    case 404:
      return "Not Found";
    case 416:
      return "Range Not Satisfiable";
    case 503:
      return "Service Unavailable";
    default:
      return "Status";
  }
}

// Début demandé par « Range: bytes=N- », std::nullopt sans en-tête Range.
std::optional<std::size_t> range_start(std::string_view request, std::string& header) {
  std::string lower(request);
  std::ranges::transform(lower, lower.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const std::size_t at = lower.find("\r\nrange: bytes=");
  if (at == std::string::npos) return std::nullopt;
  const std::size_t start = at + std::string_view("\r\nrange: bytes=").size();
  header = std::string(request.substr(start, request.find("\r\n", start) - start));
  std::size_t offset = 0;
  std::from_chars(header.data(), header.data() + header.size(), offset);
  return offset;
}

}  // namespace

HttpServer::HttpServer() {
  listener_ = static_cast<std::intptr_t>(listen_local(port_));
  thread_ = std::jthread([this](const std::stop_token& stop) { run(stop); });
}

HttpServer::~HttpServer() {
  thread_.request_stop();
  thread_.join();
  close_socket(to_socket(listener_));
}

void HttpServer::serve(const std::string& path, Route route) {
  const std::scoped_lock lock(mutex_);
  routes_.insert_or_assign(path, Served{.route = std::move(route)});
}

std::string HttpServer::url(std::string_view path) const {
  return std::format("http://127.0.0.1:{}{}", port_, path);
}

int HttpServer::hits(const std::string& path) const {
  const std::scoped_lock lock(mutex_);
  const auto found = routes_.find(path);
  return found == routes_.end() ? 0 : found->second.hits;
}

std::vector<std::string> HttpServer::ranges(const std::string& path) const {
  const std::scoped_lock lock(mutex_);
  const auto found = routes_.find(path);
  return found == routes_.end() ? std::vector<std::string>{} : found->second.ranges;
}

int HttpServer::closed_port() {
  int port = 0;
  close_socket(listen_local(port));
  return port;
}

void HttpServer::run(const std::stop_token& stop) {
  const Socket listener = to_socket(listener_);
  while (!stop.stop_requested()) {
    if (poll_one(listener, 20) <= 0) continue;
    const Socket client = ::accept(listener, nullptr, nullptr);
    if (client == kInvalid) continue;
    answer(static_cast<std::intptr_t>(client), stop);
    close_socket(client);
  }
}

void HttpServer::answer(std::intptr_t handle, const std::stop_token& stop) {
  const Socket client = to_socket(handle);
  std::string request;
  std::array<char, 4096> buffer{};
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (request.find("\r\n\r\n") == std::string::npos) {
    if (stop.stop_requested() || std::chrono::steady_clock::now() > deadline) return;
    if (poll_one(client, 20) <= 0) continue;
    const auto received = ::recv(client, buffer.data(), static_cast<Length>(buffer.size()), 0);
    if (received <= 0) return;
    request.append(buffer.data(), static_cast<std::size_t>(received));
  }
  const std::size_t path_start = request.find(' ') + 1;
  const std::string path = request.substr(path_start, request.find(' ', path_start) - path_start);
  std::string range_header;
  const std::optional<std::size_t> asked = range_start(request, range_header);

  // Décisions de cette réponse, prises sous le verrou ; l'envoi se fait hors du verrou.
  int status = 404;
  std::string body;
  std::size_t offset = 0;
  std::size_t cut = std::numeric_limits<std::size_t>::max();
  std::chrono::milliseconds stall{0};
  {
    const std::scoped_lock lock(mutex_);
    const auto found = routes_.find(path);
    if (found != routes_.end()) {
      Served& served = found->second;
      Route& route = served.route;
      ++served.hits;
      served.ranges.push_back(range_header);
      if (route.failures > 0) {
        --route.failures;
        status = 503;
      } else if (route.status != 200) {
        status = route.status;
      } else {
        body = route.body;
        status = 200;
        if (route.ranges && asked) {
          offset = *asked;
          status = offset > body.size() ? 416 : 206;
        }
        if (route.cuts > 0) {
          --route.cuts;
          cut = route.cut_after;
        }
        if (route.stalls > 0) {
          --route.stalls;
          stall = route.stall;
        }
      }
    }
  }

  std::string head = std::format("HTTP/1.1 {} {}\r\nConnection: close\r\n", status, reason(status));
  if (status == 200 || status == 206) {
    head += std::format("Content-Length: {}\r\n", body.size() - offset);
    if (status == 206)
      head += std::format("Content-Range: bytes {}-{}/{}\r\n", offset, body.size() - 1, body.size());
  } else {
    if (status == 416) head += std::format("Content-Range: bytes */{}\r\n", body.size());
    head += "Content-Length: 0\r\n";
    body.clear();
  }
  head += "\r\n";
  if (!send_all(client, head)) return;

  // Silence : le client n'envoie plus rien, donc un socket lisible veut dire qu'il est parti.
  const auto quiet_until = std::chrono::steady_clock::now() + stall;
  while (std::chrono::steady_clock::now() < quiet_until) {
    if (stop.stop_requested()) return;
    if (poll_one(client, 10) > 0 && ::recv(client, buffer.data(), 1, 0) <= 0) return;
  }
  if (body.empty()) return;
  const std::string_view rest = std::string_view(body).substr(offset);
  (void)send_all(client, rest.substr(0, std::min(cut, rest.size())));
}

}  // namespace ajp::app::test
