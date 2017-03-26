cmake_minimum_required(VERSION 2.8)

project(opencv_cpu_vs_gpu)

############################################################################################################################
# Project configuration type
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile") 
set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING "List of supported configurations.") 
mark_as_advanced(CMAKE_CONFIGURATION_TYPES) 
if(NOT CMAKE_BUILD_TYPE) 
	message("Defaulting to release build.") 
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: ${CMAKE_CONFIGURATION_TYPES}." FORCE) 
endif()

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_DEBUG_POSTFIX "_d")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
############################################################################################################################

############################################################################################################################
# OpenCV
find_package( OpenCV REQUIRED )
############################################################################################################################

############################################################################################################################
# INCLUDES
SET( ${PROJECT_NAME}_INC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
)
	
include_directories(${${PROJECT_NAME}_INC})
############################################################################################################################

############################################################################################################################
# SOURCES
file(GLOB_RECURSE ${PROJECT_NAME}_SRC *.cpp)
# HEADERS
file(GLOB_RECURSE ${PROJECT_NAME}_HDR *.h *.hpp)
############################################################################################################################

############################################################################################################################
# FLAGS
if (UNIX)
    ADD_DEFINITIONS(
        -std=c++11
    )
endif (UNIX)
############################################################################################################################

############################################################################################################################
# EXECUTABLE

add_executable(
    ${PROJECT_NAME}
    ${${PROJECT_NAME}_HDR}
    ${${PROJECT_NAME}_SRC}
)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)
############################################################################################################################
