/** \file
 *  \brief Main code Test
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

#include "CustomSink.hpp"

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__))
    mkdir("./tmp", 0700);
    const std::string kPathToLogFile = "./tmp/";
#else
    const std::string kPathToLogFile = "/tmp/";
#endif
const std::string kLogfilename = "cuda_dsmc";


/** \fn int main(int argc, char const *argv[])
 *  \brief Calls the function to fill a `zomplex2` array of aligned spins 
 *  on the host or device with a mean projection of 1.
 *  \param num_atoms Number of atoms in the thermal gas.
 *  \param params (TODO).
 *  \param *pos Pointer to a `double3` host or device array of length `num_atoms`.
 *  \param *zomplex2 Pointer to a `zomplex2` host or device array of length `num_atoms`.
 *  \exception not yet.
 *  \return void
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
