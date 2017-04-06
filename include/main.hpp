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
void testGpuMemManaged();
void saveResImages();

class Timer
{
public:
    Timer(){};

    void tic(); ///< Start timer
    double toc(); ///< Returns msec elapsed from last tic

private:
    std::chrono::high_resolution_clock::time_point mStart;
};
