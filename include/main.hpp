#include <stdlib.h>
#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>


void printUsage();
void testCpu();
void testGpu();
void testGpuZeroCopy();
void saveResImages();
