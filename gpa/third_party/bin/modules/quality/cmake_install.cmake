# Install script for directory: /home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality

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
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so.4.8.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so.408"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/smile/prog/cudaAtScaleLib/gpa/third_party/deploy/lib:/usr/local/cuda/lib64")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY OPTIONAL FILES
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/bin/lib/libopencv_quality.so.4.8.0"
    "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/bin/lib/libopencv_quality.so.408"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so.4.8.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so.408"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/usr/local/cuda/lib64:/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/bin/lib:"
           NEW_RPATH "/home/smile/prog/cudaAtScaleLib/gpa/third_party/deploy/lib:/usr/local/cuda/lib64")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so"
         RPATH "/home/smile/prog/cudaAtScaleLib/gpa/third_party/deploy/lib:/usr/local/cuda/lib64")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/bin/lib/libopencv_quality.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so"
         OLD_RPATH "/usr/local/cuda/lib64:/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/gpa/third_party/bin/lib:"
         NEW_RPATH "/home/smile/prog/cudaAtScaleLib/gpa/third_party/deploy/lib:/usr/local/cuda/lib64")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_quality.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/quality" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality/quality_utils.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/quality" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality/qualitybase.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/quality" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality/qualitybrisque.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/quality" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality/qualitygmsd.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/quality" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality/qualitymse.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/quality" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality/qualitypsnr.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/quality" TYPE FILE OPTIONAL FILES "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/include/opencv2/quality/qualityssim.hpp")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/opencv4/quality" TYPE FILE FILES
    "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/samples/brisque_model_live.yml"
    "/home/smile/prog/cudaAtScaleLib/gpa/third_party/opencv_contrib/modules/quality/samples/brisque_range_live.yml"
    )
endif()

