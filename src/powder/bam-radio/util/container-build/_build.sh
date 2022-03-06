#!/bin/bash

#
# only call this script through symlinks! :)
#
# ./build_uhd.sh -- builds UHD in a container
# ./build_gnuradio.sh -- builds gnuradio in a container
# ./build_bamradio.sh -- builds bamradio in a container
# ./new_container.sh -- make a new container from a branch
#

set -e
NAME=$(basename "$0")

#
# usage
#
function print_usage {
    if [[ "$NAME" = "build_uhd.sh" ]]; then
	echo "Usage: $0 <REPO> <BRANCH> <CONTAINER>"
	echo ""
	echo "Build UHD (branch BRANCH) from REPO in container CONTAINER."
	echo "Example: $0 \"../repos/uhd\" \"UHD-3.9.LTS\" \"bam-base\""
    elif [[ "$NAME" = "build_gnuradio.sh" ]]; then
	echo "Usage: $0 <REPO> <BRANCH> <CONTAINER>"
	echo ""
	echo "Build gnuradio (branch BRANCH) from REPO in container CONTAINER."
	echo "Example: $0 \"../repos/gnuradio\" \"origin/sgl/next++\" \"bam-base\""
    elif [[ "$NAME" = "build_bamradio.sh" ]]; then
	echo "Usage: $0 <BRANCH> <CONTAINER>"
	echo ""
	echo "Build bamradio (branch BRANCH) from the bam-radio repo in container CONTAINER."
	echo "Example: $0 \"../repos/bamradio\" \"master\" \"bam-base\""
    elif [[ "$NAME" = "build_base.sh" ]]; then
	echo "Usage: $0 <NEW-BASE-NAME>"
	echo ""
	echo "Build base image <NEW-BASE-NAME>"
    elif [[ "$NAME" = "new_container.sh" ]]; then
	echo "Usage: $0 <BRANCH> <BASE-ID> <IMAGE-PREFIX>"
	echo ""
	echo -n "create a new bam-radio container image named IMAGE-PREFIX-<commit hash>"
	echo " from the branch BRANCH of the bam-radio repo."
	echo "Use the base image BASE-ID"
    fi
}

#
# arguments & container
#
if [[ "$NAME" = "build_base.sh" ]]; then
        if [[ "$#" != 1 ]]; then
	print_usage
	exit 1
    fi
    NEW_BASE_NAME="$1"
    CONTAINER="e"$(openssl rand -hex 15)
elif [[ "$NAME" = "new_container.sh" ]]; then
    if [[ "$#" != 3 ]]; then
	print_usage
	exit 1
    fi
    BRANCH="$1"
    BASENAME="$2"
    IMG_PREFIX="$3"
    CONTAINER="e"$(openssl rand -hex 15)
elif [[ "$NAME" = "build_bamradio.sh" ]]; then
    if [[ "$#" != 2 ]]; then
	print_usage
	exit 1
    fi
    BRANCH="$1"
    CONTAINER="$2"
    if [[ $(lxc info "$CONTAINER"  | awk '/Status:/ {print $2}') != "Running" ]]; then
	echo "Container $CONTAINER is not running or non-existent."
	exit 1
    fi
else
    if [[ "$#" != 3 ]]; then
	print_usage
	exit 1
    fi
    REPO=$(readlink -f "$1")
    BRANCH="$2"
    CONTAINER="$3"
    if [[ $(lxc info "$CONTAINER"  | awk '/Status:/ {print $2}') != "Running" ]]; then
	echo "Container $CONTAINER is not running or non-existent."
	exit 1
    fi
fi

#
# run a command in the container. note that
#   run_in_container mkdir -p /path/to/dir
# works, but
#   run_in_container "mkdir -p /path/to/dir"
# does not
#
function run_in_container {
    lxc exec "$CONTAINER" -- "$@"
}

#
# push a file or directory to the container. this will create
# directories in the container if they do not exist.
#
function push_to_container {
    if [[ -d "$1" ]]; then
	lxc exec "$CONTAINER" -- mkdir -p "$2"
	tar -c -C "$1" . | lxc exec "$CONTAINER" -- tar xvf - -C "$2"
    else
	lxc exec "$CONTAINER" -- mkdir -p $(dirname "$2")
	lxc file push "$1" "$CONTAINER$2"
    fi
}

if [[ "$NAME" = "build_uhd.sh" ]]; then
    #
    # build UHD in the container
    #
    TMPD=$(mktemp -d)
    git clone "$REPO" "$TMPD"
    (cd "$TMPD" && \
	    git checkout "$BRANCH")
    push_to_container "$TMPD" "/build/uhd"
    push_to_container "../repos/x300-Optimal-ADC-DAC-settings-for-Colosseum-SRNs_UHD-3.9.x.patch" "/build/uhd/ni.patch"
    run_in_container git -C "/build/uhd" apply "/build/uhd/ni.patch"
    run_in_container mkdir -p "/build/uhd/build"
    run_in_container bash -c 'cd /build/uhd/build &&
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DENABLE_USB=OFF \
-DENABLE_TESTS=OFF \
../host'
    run_in_container bash -c 'cd /build/uhd/build && make -j30 && make install && ldconfig'
    rm -rf "$TMPD"
    
elif [[ "$NAME" = "build_gnuradio.sh" ]]; then
    #
    # build GNU radio in the container
    #
    TMPD=$(mktemp -d)
    git clone "$REPO" "$TMPD"
    (cd "$TMPD" && \
	    git checkout "$BRANCH" && \
    	    git submodule init && \
	    git submodule update)
    push_to_container "$TMPD" "/build/gnuradio"
    run_in_container mkdir -p /build/gnuradio/build
    run_in_container bash -c 'cd /build/gnuradio/build &&
cmake \
-DCMAKE_BUILD_TYPE=Release \
-DENABLE_DEFAULT=OFF \
-DENABLE_GR_QTGUI=OFF \
-DENABLE_GRC=OFF \
-DENABLE_PYTHON=OFF \
-DENABLE_SPHINX=OFF \
-DENABLE_TESTING=OFF \
-DENABLE_INTERNAL_VOLK=ON \
-DENABLE_VOLK=ON \
-DENABLE_DOXYGEN=OFF \
-DENABLE_GNURADIO_RUNTIME=ON \
-DENABLE_GR_BLOCKS=ON \
-DENABLE_GR_FEC=OFF \
-DENABLE_GR_FFT=ON \
-DENABLE_GR_FILTER=ON \
-DENABLE_GR_ANALOG=ON \
-DENABLE_GR_DIGITAL=ON \
-DENABLE_GR_CHANNELS=OFF \
-DENABLE_GR_TRELLIS=OFF \
-DENABLE_GR_UHD=ON \
-DENABLE_GR_UTILS=OFF \
-DENABLE_GR_WAVELET=OFF \
-DENABLE_GR_ZEROMQ=ON \
-DENABLE_GR_CTRLPORT=OFF \
..'
    run_in_container bash -c 'cd /build/gnuradio/build && make -j30 && make install && ldconfig'
    rm -rf "$TMPD"

elif [[ "$NAME" = "build_bamradio.sh" ]]; then
    #
    # build bam radio in the container
    #
    # little bit of a hack here
    TMP1=$(mktemp -d)
    TMP2=$(mktemp -d)
    git clone ssh://git@cloudradio.ecn.purdue.edu/bam-wireless/bam-radio.git "$TMP1" --mirror
    git clone "$TMP1" "$TMP2"
    (cd "$TMP2" && \
	    git checkout origin/"$BRANCH" && \
    	    git submodule init && \
	    git submodule update)
    push_to_container "$TMP2" "/root/bam-radio"
    run_in_container mkdir -p /root/bam-radio/build
    run_in_container bash -c 'cd /root/bam-radio/build && cmake -DCMAKE_BUILD_TYPE=Release ../controller && make -j30'
    run_in_container ln -s /root/bam-radio/build/src/bamradio /root/bamradio
    # run the llr_gen script and copy the llr tables to the correct place
    mods=( 2 4 5 6 7 8 )
    for order in "${mods[@]}"; do
	echo "Generating LLR table of order $order..."
	run_in_container bam-radio/build/src/llr_gen \
			 --precision=6 \
			 --npoints=20 \
			 --min=0 \
			 --max=40 \
			 --order=$order
    done
    COMMIT=$(cd "$TMP2" && git rev-parse HEAD)
    echo "$COMMIT" | lxc exec "$CONTAINER" -- dd of=/root/ver
    rm -rf "$TMP1"    
    rm -rf "$TMP2"

elif [[ "$NAME" = "build_base.sh" ]]; then
    #
    # build the base image. they are all based on the trusty image
    #
    UBUNTU_IMG="069b95ed3a60"
    lxc launch -p default "$UBUNTU_IMG" "$CONTAINER"
    sleep 10s
    lxc stop "$CONTAINER"
    sleep 10s
    lxc start "$CONTAINER"
    # wait until we have connection
    while true; do
	if run_in_container ping -c 1 www.google.com; then
	    break
	fi
    done
    # install dependencies from ubuntu sources
    run_in_container add-apt-repository ppa:ubuntu-toolchain-r/test --yes
    run_in_container bash -c 'export debian_frontend=noninteractive; \
apt update --yes && \
apt upgrade --yes'
        run_in_container bash -c 'export DEBIAN_FRONTEND=noninteractive; \
apt install --yes \
gcc-6 \
g++-6 \
make \
libtool \
automake \
autoconf \
git \
emacs24-nox \
unzip \
gdb \
socat \
liblog4cpp5-dev \
libcppunit-dev \
python-dev \
python-mako \
python-cheetah \
python3-pip \
hping3 \
htop \
libsqlite3-dev sqlite3 \
pkg-config'
    # CUDA
    push_to_container "../repos/cuda-repo-ubuntu1604-9-1-local_9.1.85-1_amd64.deb" "/build/cuda-9.1.deb"
    push_to_container "../repos/cuda-repo-ubuntu1604-9-1-local-cublas-performance-update-1_1.0-1_amd64.deb" "/build/cuda-9.1p1.deb"
    run_in_container dpkg -i "/build/cuda-9.1.deb"
    run_in_container apt-key add "/var/cuda-repo-9-1-local/7fa2af80.pub"
    run_in_container dpkg -i "/build/cuda-9.1p1.deb"
    run_in_container bash -c 'export debian_frontend=noninteractive; apt update --yes'
    run_in_container bash -c 'export DEBIAN_FRONTEND=noninteractive; apt install --yes --no-install-recommends cuda-9-1'
    run_in_container rm "/build/cuda-9.1.deb"
    run_in_container rm "/build/cuda-9.1p1.deb"
    # make sure the correct compiler is selected
    run_in_container update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 100 \
		     --slave /usr/bin/g++ g++ /usr/bin/g++-6 \
		     --slave /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/x86_64-linux-gnu-gcc-6 \
		     --slave /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/x86_64-linux-gnu-g++-6 \
		     --slave /usr/bin/x86_64-linux-gnu-gcc-ar x86_64-linux-gnu-gcc-ar /usr/bin/x86_64-linux-gnu-gcc-ar-6 \
		     --slave /usr/bin/x86_64-linux-gnu-gcc-nm x86_64-linux-gnu-gcc-nm /usr/bin/x86_64-linux-gnu-gcc-nm-6 \
		     --slave /usr/bin/x86_64-linux-gnu-gcc-ranlib x86_64-linux-gnu-gcc-ranlib /usr/bin/x86_64-linux-gnu-gcc-ranlib-6			 
    # run_in_container update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50 --slave /usr/bin/g++ g++ /usr/bin/g++-4.8
    # compile the rest of the deps
    # cmake
    run_in_container mkdir -p /build/cmake
    cat ../repos/cmake-3.9.0.tar.gz | lxc exec "$CONTAINER" -- tar xvzf - -C /build/cmake --strip-components=1
    run_in_container bash -c 'cd /build/cmake && ./bootstrap &&  make -j30 && make install && ldconfig'
    # boost
    run_in_container mkdir -p /build/boost
    cat ../repos/boost_1_64_0.tar.gz | lxc exec "$CONTAINER" -- tar xvzf - -C /build/boost --strip-components=1
    run_in_container bash -c 'cd /build/boost && ./bootstrap.sh && ./b2 install -j 30'
    # protobuf
    TMPD=$(mktemp -d)
    git clone ../repos/protobuf "$TMPD"
    (cd "$TMPD" && \
	    git submodule init && \
	    git submodule update && \
	    git checkout tags/v3.3.2)
    push_to_container "$TMPD" /build/protobuf
    run_in_container bash -c 'cd /build/protobuf && \
./autogen.sh && \
./configure && make -j30 && make check -j30 && make install && ldconfig'
    rm -rf "$TMPD"
    push_to_container ../repos/FindZeroMQ.cmake /usr/local/share/cmake-3.9/Modules/FindZeroMQ.cmake
    # zeromq
    run_in_container mkdir -p /build/zmq
    cat ../repos/zeromq-4.2.1.tar.gz | lxc exec "$CONTAINER" -- tar xvzf - -C /build/zmq --strip-components=1
    run_in_container bash -c 'mkdir -p /build/zmq/build && cd /build/zmq/build && \
cmake .. && make -j30 && make install && ldconfig'
    run_in_container mkdir -p /build/cppzmq
    push_to_container ../repos/cppzmq /build/cppzmq
    run_in_container bash -c 'mkdir -p /build/cppzmq/build && cd /build/cppzmq/build && cmake .. && make -j30 && make install && ldconfig'
    # FFTW
    run_in_container mkdir -p /build/fftw
    cat ../repos/fftw-3.3.6-pl2.tar.gz | lxc exec "$CONTAINER" -- tar xvzf - -C /build/fftw --strip-components=1
    run_in_container bash -c 'cd /build/fftw && ./configure \
--enable-shared \
--enable-float \
--enable-threads \
--enable-sse2 \
--enable-avx \
--enable-avx2 && \
make -j30 && make install && ldconfig'
    # GPSD
    run_in_container bash -c 'export DEBIAN_FRONTEND=noninteractive; apt-get install scons libncurses5-dev --yes'
    run_in_container mkdir -p /build/gpsd
    cat ../repos/gpsd-3.16.tar.gz | lxc exec "$CONTAINER" -- tar xvzf - -C /build/gpsd --strip-components=1
    run_in_container bash -c 'cd /build/gpsd && scons && scons check && scons udev-install && ldconfig'
    # UHD
    (/build/build_uhd.sh ../repos/uhd "UHD-3.9.LTS" "$CONTAINER")
    # GNU radio
    (/build/build_gnuradio.sh ../repos/gnuradio.git "origin/sgl/next++" "$CONTAINER")
    # ColosseumCLI
    run_in_container mkdir -p /build/ccli
    cat ../repos/colosseumcli-2.2.3.tar.gz | lxc exec "$CONTAINER" -- tar xvzf - -C /build/ccli --strip-components=1
    run_in_container pip3 install --upgrade setuptools
    run_in_container bash -c 'cd /build/ccli && python3 setup.py build && python3 setup.py install && ldconfig'
    # system settings
    # run_in_container sysctl -w net.ipv4.ip_forward=1
    run_in_container sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/g' /etc/sysctl.conf
    echo -e "auto tun0\niface tun0 inet manual\nup ip route add 192.168.0.0/16 dev tun0" \
	| lxc exec "$CONTAINER" -- bash -c 'cat - >> /etc/network/interfaces'
    run_in_container sed -i \
		     -e 's/PasswordAuthentication no/PasswordAuthentication yes/g' \
		     -e 's/PermitRootLogin prohibit-password/PermitRootLogin yes/g' \
		     -e 's/PermitEmptyPasswords no/PermitEmptyPasswords yes/g' \
		     /etc/ssh/sshd_config
    echo -e "root:*" | lxc exec "$CONTAINER" -- chpasswd -e
    # now export the container as an image
    run_in_container rm -rf /build
    run_in_container bash -c 'export DEBIAN_FRONTEND=noninteractive; apt-get clean --yes'
    true && \
	lxc stop "$CONTAINER" && \
	lxc publish "$CONTAINER" --public --verbose --alias "$NEW_BASE_NAME" && \
	lxc delete "$CONTAINER"
    # we are not exporting the image, there is not much point to
    # that. the next step is to take a bam-radio ref and compile it in
    # the container, then export that.

elif [[ "$NAME" = "new_container.sh" ]]; then
    #
    # put together a new container from the base
    #
    # new container
    lxc launch "$BASENAME" "$CONTAINER"
    # build the bamradio distribution
    (/build/build_bamradio.sh "$BRANCH" "$CONTAINER")
    # add the radio api files
    run_in_container mkdir -p /root/radio_api
    TMP1=$(mktemp -d)
    TMP2=$(mktemp -d)
    git clone ssh://git@cloudradio.ecn.purdue.edu/bam-wireless/radio_api.git "$TMP1" --mirror
    git clone "$TMP1" "$TMP2"
    (cd "$TMP2" && \
	    git submodule init && \
	    git submodule update)
    # push config.conf
    run_in_container mkdir -p /root/.gnuradio
    push_to_container "../repos/config.conf" "/root/.gnuradio/config.conf"
    push_to_container "$TMP2" /root/radio_api
    run_in_container cp /root/radio_api/bamradio.conf /etc/init
    run_in_container cp /root/radio_api/bampcap.conf /etc/init
    rm -rf "$TMP1"
    rm -rf "$TMP2"
    # clean up after yourself
    run_in_container rm -rf /root/radio_api/.git
    # export the image
    BAMRADIO_VER=$(run_in_container cat /root/ver | cut -b-12)
    IMGNAME="$IMG_PREFIX"-"$BAMRADIO_VER"
    true && \
	lxc stop "$CONTAINER" && \
	lxc publish "$CONTAINER" --public --verbose --alias "$IMGNAME" && \
	lxc image export "$IMGNAME" /build/"$IMGNAME" && \
	lxc delete "$CONTAINER" && \
	lxc image delete "$IMGNAME" && \
	chmod 755 "$IMGNAME".tar.gz
fi

# done here
