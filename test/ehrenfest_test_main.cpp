#include <string>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#if defined(LOGGING)
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#endif

#if defined(LOGGING)
#include "custom_sink.hpp"
#endif

#include "helper_cuda.h"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    const std::string path_to_log_file = "./";
#else
    const std::string path_to_log_file = "/tmp/";
#endif

int main(int argc,
         char* const argv[]) {
#if defined(LOGGING)
    // Initialise logger
    auto worker = g3::LogWorker::createLogWorker();
    auto default_handle = worker->addDefaultLogger(argv[0], path_to_log_file);
    auto output_handle = worker->addSink(std2::make_unique<CustomSink>(),
                                         &CustomSink::ReceiveLogMessage);
    g3::initializeLogging(worker.get());
    std::future<std::string> log_file_name = default_handle->
                                             call(&g3::FileSink::fileName);
    std::cout << "\n All logging output will be written to: "
              << log_file_name.get() << "\n" << std::endl;
    // g3::only_change_at_initialization::setLogLevel(DEBUG, false);
#endif
    printf("*********************************\n");
    printf("*                               *\n");
    printf("*   WELCOME TO EHRENFEST TEST   *\n");
    printf("*                               *\n");
    printf("*********************************\n");

#if defined(LOGGING)
    LOGF(INFO, "\nRunnning on your local CUDA device.");
#endif
    // findCudaDevice(argc,
    //                argv);

    int result = Catch::Session().run(argc,
                                      argv);

#if defined(LOGGING)
    g3::internal::shutDownLogging();
#endif

  return result;
}
