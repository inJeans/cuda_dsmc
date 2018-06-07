/** \file
 *  \brief Main code
 *
 *  More detailed description
 *  Copyright 2017 Christopher Watkins
 */

/** \var const std::strin kPathToLogFile
    \brief String with path to log file.
    
    More.
*/
/** \var const std::strin kLogfilename
    \brief String with path to log file.
    
    More.
*/

#include <stdio.h>
#include <g3log/g3log.hpp>
#include <g3log/logworker.hpp>
#include <g3log/std2_make_unique.hpp>

#include <string>

#include "cuda_dsmc/CustomSink.hpp"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    mkdir("./tmp", 0700);
    const std::string kPathToLogFile = "./tmp/";
#else
    const std::string kPathToLogFile = "/tmp/";
#endif
const std::string kLogfilename = "cuda_dsmc";


/** \brief Main process for the DSMC code.
 *
 *  \param argc testing words
 *  \param argv 
 *  \exception not yet.
 *  \return int
 */
int main(int argc, char const *argv[]) {
    auto worker = g3::LogWorker::createLogWorker();
    auto logfileHandle = worker->addDefaultLogger(kLogfilename,
                                                  kPathToLogFile);

    // logger is initialized
    g3::initializeLogging(worker.get());

    auto stdoutHandle = worker->addSink(std2::make_unique<CustomSink>(),
                                        &CustomSink::ReceiveLogMessage);

    LOGF(INFO, "****************************");
    LOGF(INFO, "*                          *");
    LOGF(INFO, "*   WELCOME TO CUDA DSMC   *");
    LOGF(INFO, "*                          *");
    LOGF(INFO, "****************************");

    printf("PASSED\n");

    return 0;
}
