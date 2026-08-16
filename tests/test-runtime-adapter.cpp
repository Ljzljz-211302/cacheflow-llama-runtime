#include "server-runtime-adapter.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

int main() {
    server_deterministic_runtime runtime({ { 1, -2, 0 }, 1000 });
    llama_batch empty{};
    const auto oom = runtime.decode(nullptr, empty);
    require(oom.code == 1 && !oom.committed, "injected KV OOM is uncommitted");
    require(oom.elapsed_us >= 500, "fixed decode latency is observable");
    const auto compute = runtime.decode(nullptr, empty);
    require(compute.code == -2 && !compute.committed, "partial compute failure is uncommitted");
    const auto success = runtime.decode(nullptr, empty);
    require(success.code == 0 && success.committed, "successful decode commits");
    require(runtime.calls() == 3, "deterministic call accounting");
    return 0;
}
