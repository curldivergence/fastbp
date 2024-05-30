#!/bin/bash

BUILD_TYPE=${1:-release}

cd fastbp
make clean

if [ "$BUILD_TYPE" = "debug" ]; then
    echo "Building in debug mode"
    make debug
else
    echo "Building in release mode"
    make
fi

cd ..
