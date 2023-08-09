# Install script for directory: /media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/smile/prog/cudaAtScaleLib/gpa/third_party/deploy")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/opencv4/haarcascades" TYPE FILE FILES
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_eye.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_frontalcatface.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_frontalcatface_extended.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_fullbody.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_lefteye_2splits.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_license_plate_rus_16stages.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_lowerbody.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_profileface.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_righteye_2splits.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_russian_plate_number.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_smile.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/haarcascades/haarcascade_upperbody.xml"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/opencv4/lbpcascades" TYPE FILE FILES
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/lbpcascades/lbpcascade_frontalcatface.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/lbpcascades/lbpcascade_frontalface.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/lbpcascades/lbpcascade_profileface.xml"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/opencv/data/lbpcascades/lbpcascade_silverware.xml"
    )
endif()

