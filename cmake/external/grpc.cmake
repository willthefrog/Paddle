# Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

IF(NOT WITH_DISTRIBUTE)
    return()
ENDIF()

IF(WITH_SYSTEM_GRPC)
  FIND_PACKAGE(PkgConfig REQUIRED)
  PKG_CHECK_MODULES(GPR REQUIRED gpr)
  PKG_CHECK_MODULES(GRPC REQUIRED grpc)
  PKG_CHECK_MODULES(GRPC_UNSECURE REQUIRED grpc_unsecure)
  PKG_CHECK_MODULES(GRPC++ REQUIRED grpc++)
  PKG_CHECK_MODULES(GRPC++_UNSECURE REQUIRED grpc++_unsecure)

  ADD_LIBRARY(extern_grpc INTERFACE IMPORTED GLOBAL)
  ADD_LIBRARY(gpr INTERFACE IMPORTED GLOBAL)
  ADD_LIBRARY(grpc INTERFACE IMPORTED GLOBAL)
  ADD_LIBRARY(grpc_unsecure INTERFACE IMPORTED GLOBAL)
  ADD_LIBRARY(grpc++ INTERFACE IMPORTED GLOBAL)
  ADD_LIBRARY(grpc++_unsecure INTERFACE IMPORTED GLOBAL)

  SET_PROPERTY(TARGET extern_grpc PROPERTY IMPORTED_LIBNAME grpc)
  SET_PROPERTY(TARGET gpr PROPERTY IMPORTED_LIBNAME gpr)
  SET_PROPERTY(TARGET grpc PROPERTY IMPORTED_LIBNAME grpc)
  SET_PROPERTY(TARGET grpc_unsecure PROPERTY IMPORTED_LIBNAME grpc_unsecure)
  SET_PROPERTY(TARGET grpc++ PROPERTY IMPORTED_LIBNAME grpc++)
  SET_PROPERTY(TARGET grpc++_unsecure PROPERTY IMPORTED_LIBNAME grpc++_unsecure)

  FIND_PROGRAM(GRPC_CPP_PLUGIN NAMES grpc_cpp_plugin)
  SET(GRPC_INCLUDE_DIR ${GRPC_INCLUDE_DIRS} CACHE PATH "grpc include directory." FORCE)

  ADD_DEPENDENCIES(extern_grpc zlib protobuf)
  RETURN()
ENDIF()

include (ExternalProject)

SET(GRPC_SOURCES_DIR ${THIRD_PARTY_PATH}/grpc)
SET(GRPC_INSTALL_DIR ${THIRD_PARTY_PATH}/install/grpc)
SET(GRPC_INCLUDE_DIR "${GRPC_INSTALL_DIR}/include/" CACHE PATH "grpc include directory." FORCE)
SET(GRPC_CPP_PLUGIN "${GRPC_INSTALL_DIR}/bin/grpc_cpp_plugin" CACHE FILEPATH "GRPC_CPP_PLUGIN" FORCE)

include(ProcessorCount)
ProcessorCount(NUM_OF_PROCESSOR)

IF(APPLE)
  SET(BUILD_CMD make -n HAS_SYSTEM_PROTOBUF=false -s -j ${NUM_OF_PROCESSOR} static grpc_cpp_plugin | sed "s/-Werror//g" | sh)
  SET(GRPC_INSTALL_CMD make prefix=${GRPC_INSTALL_DIR} install) 
ELSE()
  SET(GRPC_CFLAGS "-Wno-error -std=c11 ${CLFAGS}")
  SET(GRPC_CXXFLAGS "-Wno-error -std=c++11 ${CXXFLAGS}")
  SET(BUILD_CMD make CFLAGS=${GRPC_CFLAGS} CXXFLAGS=${GRPC_CXXFLAGS} HAS_SYSTEM_PROTOBUF=false -s -j ${NUM_OF_PROCESSOR} static grpc_cpp_plugin) 
  SET(GRPC_INSTALL_CMD make prefix=${GRPC_INSTALL_DIR} install CFLAGS=${GRPC_CFLAGS} CXXFLAGS=${GRPC_CXXFLAGS})
ENDIF()

# FIXME(wuyi): do not build zlib cares protobuf twice, find a way to build grpc with them
ExternalProject_Add(
    extern_grpc
    DEPENDS protobuf zlib
    # NOTE(wuyi):
    # this package is generated by following steps:
    # 1. git clone -b v1.8.x https://github.com/grpc/grpc.git
    # 2. git submodule update --init
    # 3. keep only zlib, cares, protobuf, boringssl under "third_party",
    #    checkout and clean other dirs under third_party
    # 4. remove .git, and package the directory.
    URL          http://paddlepaddledeps.bj.bcebos.com/grpc-v1.10.x_paddle.tar.gz
    URL_MD5      f5442d137ddccee252e194b1bc90f98c
    PREFIX          ${GRPC_SOURCES_DIR}
    UPDATE_COMMAND  ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE 1
    # NOTE(yuyang18):
    # Disable -Werror, otherwise the compile will fail in MacOS.
    # It seems that we cannot configure that by make command.
    # Just dry run make command and remove `-Werror`, then use a shell to run make commands
    BUILD_COMMAND  ${BUILD_CMD}
    INSTALL_COMMAND ${GRPC_INSTALL_CMD}
)

ADD_LIBRARY(grpc++_unsecure STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET grpc++_unsecure PROPERTY IMPORTED_LOCATION
             "${GRPC_INSTALL_DIR}/lib/libgrpc++_unsecure.a")

ADD_LIBRARY(grpc++ STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET grpc++ PROPERTY IMPORTED_LOCATION
            "${GRPC_INSTALL_DIR}/lib/libgrpc++.a")
ADD_LIBRARY(gpr STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET gpr PROPERTY IMPORTED_LOCATION
            "${GRPC_INSTALL_DIR}/lib/libgpr.a")

ADD_LIBRARY(grpc_unsecure STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET grpc_unsecure PROPERTY IMPORTED_LOCATION
            "${GRPC_INSTALL_DIR}/lib/libgrpc_unsecure.a")

include_directories(${GRPC_INCLUDE_DIR})
ADD_DEPENDENCIES(grpc++_unsecure extern_grpc)
