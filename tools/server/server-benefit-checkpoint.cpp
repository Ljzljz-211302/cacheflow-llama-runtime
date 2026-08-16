#include "server-benefit-checkpoint.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace {
constexpr uintmax_t maximum_checkpoint_bytes = 4U * 1024U * 1024U;

class server_benefit_file_checkpoint_store final : public server_benefit_checkpoint_store {
public:
    explicit server_benefit_file_checkpoint_store(std::filesystem::path path) : path_(std::move(path)) {}

    server_benefit_checkpoint_load_result load() override {
        std::error_code ec;
        if (!std::filesystem::exists(path_, ec)) {
            if (ec) return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR, {}, ec.message()};
            return {};
        }
        const auto size = std::filesystem::file_size(path_, ec);
        if (ec) return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR, {}, ec.message()};
        if (size > maximum_checkpoint_bytes) {
            return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR, {}, "checkpoint exceeds 4 MiB limit"};
        }
        std::ifstream input(path_, std::ios::binary);
        if (!input) {
            return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR, {}, "cannot open checkpoint for reading"};
        }
        std::string payload((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (!input.good() && !input.eof()) {
            return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR, {}, "checkpoint read failed"};
        }
        return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_RESTORED, std::move(payload), {}};
    }

    bool save(const std::string & payload, std::string & error) override {
        if (payload.size() > maximum_checkpoint_bytes) {
            error = "checkpoint exceeds 4 MiB limit";
            return false;
        }
        std::error_code ec;
        const auto parent = path_.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                error = "cannot create checkpoint directory: " + ec.message();
                return false;
            }
        }
        const std::filesystem::path temporary = path_.string() + ".tmp";
#ifdef _WIN32
        std::FILE * file = _wfopen(temporary.c_str(), L"wb");
#else
        std::FILE * file = std::fopen(temporary.c_str(), "wb");
#endif
        if (file == nullptr) {
            error = "cannot open temporary checkpoint: " + std::string(std::strerror(errno));
            return false;
        }
        bool ok = std::fwrite(payload.data(), 1, payload.size(), file) == payload.size();
        if (ok) ok = std::fflush(file) == 0;
#ifdef _WIN32
        if (ok) ok = _commit(_fileno(file)) == 0;
#else
        if (ok) ok = ::fsync(fileno(file)) == 0;
#endif
        if (std::fclose(file) != 0) ok = false;
        if (!ok) {
            error = "cannot durably write temporary checkpoint: " + std::string(std::strerror(errno));
            std::filesystem::remove(temporary, ec);
            return false;
        }
#ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), path_.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            error = "cannot atomically replace checkpoint: Win32 error " + std::to_string(GetLastError());
            std::filesystem::remove(temporary, ec);
            return false;
        }
#else
        if (std::rename(temporary.c_str(), path_.c_str()) != 0) {
            error = "cannot atomically replace checkpoint: " + std::string(std::strerror(errno));
            std::filesystem::remove(temporary, ec);
            return false;
        }
        const std::filesystem::path directory = parent.empty() ? std::filesystem::path(".") : parent;
        int directory_flags = O_RDONLY;
#ifdef O_DIRECTORY
        directory_flags |= O_DIRECTORY;
#endif
        const int directory_fd = ::open(directory.c_str(), directory_flags);
        if (directory_fd < 0 || ::fsync(directory_fd) != 0) {
            error = "checkpoint committed but directory sync failed: " + std::string(std::strerror(errno));
            if (directory_fd >= 0) ::close(directory_fd);
            return false;
        }
        ::close(directory_fd);
#endif
        return true;
    }

private:
    std::filesystem::path path_;
};
}

std::unique_ptr<server_benefit_checkpoint_store> server_benefit_create_file_checkpoint_store(
        const std::filesystem::path & path) {
    return std::make_unique<server_benefit_file_checkpoint_store>(path);
}

server_benefit_checkpoint::server_benefit_checkpoint(
        std::unique_ptr<server_benefit_checkpoint_store> store) : store_(std::move(store)) {
    if (!store_) throw std::invalid_argument("benefit checkpoint store must not be null");
    worker_ = std::thread(&server_benefit_checkpoint::run, this);
}

server_benefit_checkpoint::~server_benefit_checkpoint() {
    flush();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_one();
    if (worker_.joinable()) worker_.join();
}

server_benefit_checkpoint_load_result server_benefit_checkpoint::load() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_ || writing_ || enqueued_ != 0) {
        return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR, {},
                "checkpoint load is only valid before the first enqueue"};
    }
    try {
        return store_->load();
    } catch (const std::exception & exception) {
        return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR,
                {}, exception.what()};
    } catch (...) {
        return {server_benefit_checkpoint_load_status::SERVER_BENEFIT_CHECKPOINT_LOAD_STATUS_IO_ERROR,
                {}, "unknown checkpoint load failure"};
    }
}

void server_benefit_checkpoint::enqueue(std::string payload) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_) coalesced_++;
        pending_payload_ = std::move(payload);
        pending_sequence_ = ++enqueued_;
        pending_ = true;
    }
    condition_.notify_one();
}

bool server_benefit_checkpoint::flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    const uint64_t target = enqueued_;
    condition_.notify_one();
    flushed_.wait(lock, [&]() { return completed_sequence_ >= target; });
    return save_failures_ == 0;
}

server_benefit_checkpoint_io_snapshot server_benefit_checkpoint::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {enqueued_, coalesced_, saves_completed_, save_failures_, pending_ || writing_ ? 1U : 0U};
}

void server_benefit_checkpoint::run() {
    for (;;) {
        std::string payload;
        uint64_t sequence = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&]() { return pending_ || stopping_; });
            if (!pending_ && stopping_) return;
            payload = std::move(pending_payload_);
            sequence = pending_sequence_;
            pending_ = false;
            writing_ = true;
        }
        std::string error;
        bool saved = false;
        try {
            saved = store_->save(payload, error);
        } catch (const std::exception & exception) {
            error = exception.what();
        } catch (...) {
            error = "unknown checkpoint save failure";
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            writing_ = false;
            completed_sequence_ = sequence;
            if (saved) saves_completed_++;
            else save_failures_++;
        }
        flushed_.notify_all();
    }
}
