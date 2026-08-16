#ifdef NDEBUG
#undef NDEBUG
#endif

#include "server-inference-iteration.h"

#include <cassert>

int main() {
    server_inference_iteration iteration;
    assert(!iteration.begin_execute());
    assert(iteration.prepare());
    assert(!iteration.prepare());
    assert(iteration.plan({2, 6, 8}));
    const auto frozen = iteration.plan_summary();
    assert(iteration.begin_execute());
    assert(iteration.record_executed(3));
    assert(!iteration.begin_commit());
    assert(iteration.record_executed(5));
    assert(iteration.begin_commit());
    assert(iteration.plan_summary().decode_tokens == frozen.decode_tokens);
    assert(iteration.commit());
    assert(iteration.validate().empty());

    server_inference_iteration failed;
    assert(failed.prepare() && failed.plan({1, 3, 4}) && failed.begin_execute());
    assert(failed.record_executed(2));
    failed.abort("injected decode failure");
    assert(failed.is_aborted() && !failed.begin_commit());
    assert(failed.executed_tokens() == 2 && failed.validate().empty());

    server_inference_iteration overflow;
    assert(overflow.prepare() && overflow.plan({0, 1, 1}) && overflow.begin_execute());
    assert(!overflow.record_executed(2));
    return 0;
}
