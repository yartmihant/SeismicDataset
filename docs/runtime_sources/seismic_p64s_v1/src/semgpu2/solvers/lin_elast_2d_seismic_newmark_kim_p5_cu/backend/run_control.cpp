#include <runtime.hpp>

namespace SEMGPU2_SEISMIC_NAMESPACE {
void Completion::publish(const RunOutcome& outcome) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) return;
    outcome_ = outcome;
    ready_ = true;
    ready_condition_.notify_all();
}
void Completion::publish_error(const RunFailure& failure) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_) return;
    outcome_ = {failure.code, failure.tact, failure.current_time, failure.steps_count, true};
    message_ = failure.what();
    failed_ = true;
    ready_ = true;
    ready_condition_.notify_all();
}
RunOutcome Completion::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_condition_.wait(lock, [&] { return ready_; });
    if (failed_) throw RunFailure(outcome_.code, outcome_.tact, outcome_.current_time, message_);
    return outcome_;
}
void RunAuthority::enqueue(RunCommand kind, Code code) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) throw std::runtime_error("Run command was sent after terminal outcome");
    commands_.push_back({kind, code});
}
std::vector<Command> RunAuthority::drain_commands() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Command> result(commands_.begin(), commands_.end());
    commands_.clear();
    return result;
}
void RunAuthority::close_commands() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    commands_.clear();
}
void RunAuthority::handler_failure() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!closed_) commands_.push_back({RunCommand::ABORT, static_cast<Code>(SGRunCodeEnum::HANDLER_FAILURE)});
}
void RunAuthority::device_failure(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_failure_.empty()) device_failure_ = message.empty() ? "Asynchronous CUDA receiver failed" : message;
    if (!closed_) commands_.push_back({RunCommand::ABORT, static_cast<Code>(SGRunCodeEnum::DEVICE_FAILURE)});
}
std::string RunAuthority::device_failure() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return device_failure_;
}
}  // namespace SEMGPU2_SEISMIC_NAMESPACE
