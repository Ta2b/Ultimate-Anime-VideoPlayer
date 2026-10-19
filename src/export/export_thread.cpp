#include "export/export_thread.h"

#include <optional>
#include <stop_token>
#include <utility>
#include <variant>

namespace ajp::exporter {
namespace {

// La boîte vue comme une source d'images : une image reste valable jusqu'à l'appel suivant,
// qui la rend au producteur par un Credit.
class InboxSource final : public ExportFrameSource {
 public:
  InboxSource(ExportInbox& inbox, std::stop_token stop, std::function<void(Credit)> credit,
              std::optional<std::int64_t> total)
      : inbox_(inbox), stop_(std::move(stop)), credit_(std::move(credit)), total_(total) {}

  InboxSource(const InboxSource&) = delete;
  InboxSource& operator=(const InboxSource&) = delete;
  ~InboxSource() override { release(); }

  std::expected<std::optional<SourceFrame>, Refusal> next() override {
    release();
    auto message = inbox_.wait(stop_);
    if (!message) return std::unexpected(Refusal{RefusalCode::Cancelled, "thread export arrêté"});
    if (auto* frame = std::get_if<ExportFrameMsg>(&*message)) {
      current_ = std::move(*frame);
      return SourceFrame{current_->frame};
    }
    if (std::holds_alternative<ExportEnd>(*message)) return std::nullopt;
    return std::unexpected(Refusal{RefusalCode::Cancelled, "export annulé par ExportCancel"});
  }

  std::optional<std::int64_t> total_frames() const override { return total_; }

 private:
  void release() {
    if (!current_) return;
    current_.reset();
    credit_(Credit{1});
  }

  ExportInbox& inbox_;
  std::stop_token stop_;
  std::function<void(Credit)> credit_;
  std::optional<std::int64_t> total_;
  std::optional<ExportFrameMsg> current_;
};

}  // namespace

std::expected<std::unique_ptr<ExportThread>, Refusal> ExportThread::start(ExportJob job,
                                                                          ExportThreadHooks hooks,
                                                                          ThreadReport report) {
  if (!hooks.credit)
    return std::unexpected(Refusal{RefusalCode::FieldMissing, "thread export : rappel credit absent"});
  if (auto ok = validate_job(job); !ok) return std::unexpected(ok.error());

  std::unique_ptr<ExportThread> owner(new ExportThread());
  owner->inbox_ = std::make_shared<ExportInbox>(ExportInbox::Bounds{kFramesInFlight, 1, 1});
  auto body = [inbox = owner->inbox_, job = std::move(job), hooks = std::move(hooks)](
                  const std::stop_token& stop, const ThreadReport& thread_report) mutable {
    std::optional<std::int64_t> total = job.frames_total;
    InboxSource source(*inbox, stop, hooks.credit, total);
    job.source = &source;
    job.stop = stop;
    job.on_progress = hooks.progress;
    auto result = run_export(job);
    if (!result) {
      thread_report(result.error());
      return;
    }
    if (hooks.done) hooks.done(*result);
  };
  auto thread = start_thread("export", std::move(report), std::move(body));
  if (!thread) return std::unexpected(thread.error());
  owner->thread_ = std::move(*thread);
  return owner;
}

ExportThread::~ExportThread() {
  thread_.request_stop();
  join();
}

void ExportThread::join() {
  if (thread_.joinable()) thread_.join();
}

}  // namespace ajp::exporter
