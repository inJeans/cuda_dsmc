# Start with the CUDA image from nvidia
FROM injeans/cuda-devel:latest
MAINTAINER Chris Watkins <christopher.watkins@me.com>

# Install wget and build-essential
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    cmake \
    unzip \
    git

# Download and install g3log
RUN wget https://github.com/KjellKod/g3log/archive/v1.1.tar.gz && \
    tar xvfz v1.1.tar.gz && \
    cd g3log-1.1 && \
    mkdir build && \
    cd ./build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_DYNAMIC_LOGGING_LEVELS=ON .. && \
    make && \
    cp -r ../src/g3log /usr/local/include && \
    cp ./lib* /usr/local/lib

# Download and install pcg_rand
RUN git clone https://github.com/imneme/pcg-c.git && \
    cd pcg-c && \
    make && \
    make test && \
    cp src/libpcg_random.a /usr/local/lib/ && \
    cp include/pcg_variants.h /usr/local/include/


# Download and install testu01
RUN wget http://www.iro.umontreal.ca/~simardr/testu01/TestU01.zip && \
    unzip TestU01.zip && \
    cd TestU01* && \
    ./configure && \
    make && \
    make install

# Add the testu01 installation to the environment paths
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/lib:$LIBRARY_PATH
ENV C_INCLUDE_PATH /usr/local/include:$C_INCLUDE_PATH