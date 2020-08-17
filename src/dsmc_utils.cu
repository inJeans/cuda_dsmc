/** \file
 *  \brief Utility functions
 *
 *  All the common utility functions I need for doing stuff like copying,
 *  collating results from multiple devices/streams/threads, saving arrays
 *  Things of that nature
 *
 *  Copyright 2017 Christopher Watkins
 */

#include "cuda_dsmc/dsmc_utils.cuh"

/** \brief Generate unique hash given a hostname
 *
 *  \param string Hostname character string to be hashed.
 *  \exception not yet.
 *  \return Hashed hostname
 */
static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

/** \brief Get and parse hostname
 *
 *  \param hostname Pointer to array for storing the hostname.
 *  \param maxlen Number of elements in the hostname string.
 *  \exception not yet.
 *  \return None
 */
static void getHostName(char* hostname,
                        int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }

  return;
}

/** \brief Evenly divide elements amongst parallel unit
 *
 *  \param num_arrays Number of ranks in the MPI world.
 *  \param num_elements Pointer to the global number of elements.
 *  \exception not yet.
 *  \return The rank local number of elements
 */
int getLocalDeviceId() {
    int myRank, nRanks, localRank = 0;
    
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    //calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPI_CHECK(MPI_Allgather(MPI_IN_PLACE,
                            0,
                            MPI_DATATYPE_NULL,
                            hostHashs,
                            sizeof(uint64_t),
                            MPI_BYTE,
                            MPI_COMM_WORLD));
    for (int p=0; p<nRanks; p++) {
        if (p == myRank) break;
        if (hostHashs[p] == hostHashs[myRank]) localRank++;
    }

    return localRank;
}
