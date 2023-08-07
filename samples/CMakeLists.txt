#add_dependencies(cudaAtScaleLib gpa)


file(GLOB SOURCES "../samples_gpa_lib/*")


if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../samples_gpa_lib/third_party)
    set(OCV_SAMPLE_DATA "")
    set(MANDATORY_PATH true)
    set(MAKE_PATH_MADATORY @)
else()
    set(DEPLOY_DIR_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../samples_gpa_lib/third_party/deploy")
    set(OpenCV_DIR ${DEPLOY_DIR_PATH}/lib/cmake/opencv4)

    set(OCV_SAMPLE_DATA ${CMAKE_CURRENT_SOURCE_DIR}/../samples_gpa_lib/third_party/opencv/samples/data)
    set(MANDATORY_PATH false)
    set(MAKE_PATH_MADATORY )

endif()
    find_package(OpenCV 4.8.0 REQUIRED)
    find_package(CUDA REQUIRED)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/internal/test.utils.hpp.in ${CMAKE_CURRENT_SOURCE_DIR}/internal/test.utils.hpp.in.config)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/internal/test.utils.hpp.in.config ${CMAKE_CURRENT_SOURCE_DIR}/sources/test.utils.hpp COPYONLY)



    #NEIF -> NPPI EXTRA IMAGE FILTERING, i.e. the lib compile in the first project.
    set(LIB_NEIF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/cudaAtScale/include)
    set(LIB_NEIF_LIB_DIR ${CMAKE_BINARY_DIR}/cudaAtScale/lib)
#    set(LIB_NEIF_LIB -lnppi_extra_image_filtering -lcuda_gpabf)

    set(LIB_GPA_INCLUDE_DIR ${CMAKE_BINARY_DIR}/cudaAtScale/samples/gpa/include)
    set(LIB_GPA_LIB_DIR ${CMAKE_BINARY_DIR}/cudaAtScale/samples/gpa/lib)
    set(LIB_GPA_LIB -lgpa)

    message("OCV_LIBS: ${OpenCV_LIBS}")
    message("NEIF: ${LIB_NEIF_LIB}")

    file(GLOB DEMO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/sources/*.cpp")
    foreach(sample ${DEMO_FILES})
        get_filename_component(sample_name ${sample} NAME_WE)
        add_executable(${sample_name} ${sample})

        target_include_directories(${sample_name} PRIVATE ${OpenCV_INCLUDE_DIRS} ${LIB_NEIF_INCLUDE_DIR} ${LIB_GPA_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
        target_link_directories(${sample_name} PRIVATE ${OpenCV_INCLUDE_DIRS}/../../lib  ${LIB_NEIF_LIB_DIR} ${LIB_GPA_LIB_DIR})
        target_link_libraries(${sample_name} PRIVATE ${OpenCV_LIBS} ${LIB_NEIF_LIB} ${LIB_GPA_LIB})

        add_custom_command(TARGET ${sample_name}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "${sample} was built successfully."
        )

        # Install the executable to the bin directory
#        install(TARGETS ${sample_name} RUNTIME DESTINATION bin)

#        target_include_directories(${sample_name} PRIVATE ${CMAKE_BINARY_DIR}/cudaAtScale/samples/lib_gpa/include)
#        target_link_directories(${sample_name} PRIVATE ${OpenCV_INCLUDE_DIRS}/../../lib)
#        target_link_libraries(${sample_name} PRIVATE -lgpa)

    endforeach()
