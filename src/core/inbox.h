// Boîte de messages d'un thread : plusieurs producteurs, un seul consommateur.
//
// Chaque type de message a sa propre borne. Quand try_push échoue, le
// producteur n'attend pas ici : il attend dans sa propre boîte un crédit du
// consommateur. wait() est le seul point d'attente du thread propriétaire ; try_pop
// prend ce qui attend déjà sans attendre.
#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stop_token>
#include <type_traits>
#include <utility>
#include <variant>

namespace ajp {

template <class... Msg>
class Inbox {
 public:
  using Message = std::variant<Msg...>;
  using Bounds = std::array<std::size_t, sizeof...(Msg)>;

  explicit Inbox(const Bounds& bounds) : bounds_(bounds) {}

  Inbox(const Inbox&) = delete;
  Inbox& operator=(const Inbox&) = delete;

  // false si la borne de M est atteinte ; le message n'est alors pas pris.
  template <class M>
  bool try_push(M&& message) {
    constexpr std::size_t kind = index_of<std::remove_cvref_t<M>>();
    {
      std::scoped_lock lock(mutex_);
      if (counts_[kind] >= bounds_[kind]) return false;
      ++counts_[kind];
      queue_.emplace_back(std::in_place_index<kind>, std::forward<M>(message));
    }
    ready_.notify_one();
    return true;
  }

  // Attend le prochain message, dans l'ordre d'arrivée tous types confondus.
  // Renvoie std::nullopt quand l'arrêt du thread est demandé.
  std::optional<Message> wait(std::stop_token stop) {
    std::unique_lock lock(mutex_);
    if (!ready_.wait(lock, std::move(stop), [this] { return !queue_.empty(); })) return std::nullopt;
    return pop();
  }

  // Le prochain message s'il y en a déjà un, sans attendre ; std::nullopt sinon.
  std::optional<Message> try_pop() {
    std::scoped_lock lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    return pop();
  }

  // Nombre de messages de type M en attente.
  template <class M>
  std::size_t pending() const {
    std::scoped_lock lock(mutex_);
    return counts_[index_of<M>()];
  }

 private:
  // Sous mutex_, file non vide.
  Message pop() {
    Message message = std::move(queue_.front());
    queue_.pop_front();
    --counts_[message.index()];
    return message;
  }

  template <class M>
  static constexpr std::size_t index_of() {
    constexpr std::array matches{std::is_same_v<M, Msg>...};
    static_assert((std::is_same_v<M, Msg> + ...) == 1, "M doit figurer une seule fois dans Msg...");
    std::size_t i = 0;
    while (!matches[i]) ++i;
    return i;
  }

  mutable std::mutex mutex_;
  std::condition_variable_any ready_;
  std::deque<Message> queue_;
  Bounds bounds_;
  Bounds counts_{};
};

}  // namespace ajp
