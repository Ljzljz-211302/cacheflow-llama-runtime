include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
      server-inference-scheduler.cpp
      server-inference-scheduler.h
      server-inference-iteration.cpp
      server-inference-iteration.h
      server-inference-engine.cpp
      server-inference-engine.h
    server-kv-capacity-planner.cpp
    server-kv-capacity-planner.h
    server-kv-block-manager.cpp
    server-kv-block-manager.h
    server-kv-runtime.cpp
    server-kv-runtime.h
      server-kv-block-backend.cpp
      server-kv-block-backend.h
      server-kv-swap-store.cpp
      server-kv-swap-store.h
    server-speculation-controller.cpp
    server-speculation-controller.h
    server-runtime-adapter.cpp
    server-runtime-adapter.h
    server-model-cost-model.cpp
    server-model-cost-model.h
    server-kv-action-policy.cpp
    server-kv-action-policy.h
    server-benefit-policy.cpp
    server-benefit-policy.h
    server-benefit-checkpoint.cpp
    server-benefit-checkpoint.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

if (GGML_CUDA)
    target_sources(${TARGET} PRIVATE server-kv-block-cuda.cu)
    target_compile_definitions(${TARGET} PUBLIC GGML_USE_CUDA)
    target_link_libraries(${TARGET} PUBLIC CUDA::cudart)
endif()

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
