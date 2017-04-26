#include <main.hpp>

#include <string>

using namespace std;
using namespace chrono;

typedef struct _data
{
    string srcFile;
    cv::Mat source;
    int count;

    double totMemUpCpu_msec;
    double totResizeCpu_msec;
    double totBlurCpu_msec;
    double totCvtCpu_msec;
    double totCannyCpu_msec;
    double totCpu_msec;

    double totMemUpGpu_msec;
    double totResizeGpu_msec;
    double totBlurGpu_msec;
    double totCvtGpu_msec;
    double totCannyGpu_msec;
    double totGpu_msec;

    double totMemUpGpuZc_msec;
    double totResizeGpuZc_msec;
    double totBlurGpuZc_msec;
    double totCvtGpuZc_msec;
    double totCannyGpuZc_msec;
    double totGpuZc_msec;

    double totMemGpuMan_msec;
    double totResizeGpuMan_msec;
    double totBlurGpuMan_msec;
    double totCvtGpuMan_msec;
    double totCannyGpuMan_msec;
    double totGpuMan_msec;
} Data;

Data globalData;

int main( int argc, const char* argv[] )
{
    cout << " opencv_cpu_vs_gpu " << endl;
    cout << "===================" << endl << endl;


    if( argc==1 )
    {
        printUsage();

        return EXIT_SUCCESS;
    }

    if( argc>=2 )
    {
        globalData.srcFile = argv[1];
        globalData.source = cv::imread( globalData.srcFile );

        if( globalData.source.empty() )
        {
            cout << "Error reading file \"" << globalData.srcFile << "\"" << endl << endl;

            printUsage();

            return EXIT_FAILURE;
        }
    }

    globalData.count = 100;

    if( argc==3 )
    {
        string countStr = argv[2];
        try
        {
            globalData.count = stoi( countStr );
        }
        catch(...)
        {
            cout << "Invalid 'iterations' parameter" << endl << endl;
            printUsage();

            return EXIT_FAILURE;
        }
    }

    cout << "The test will be performed by averaging the timing of " << globalData.count << " iterations." << endl << endl;

    saveResImages();

    testCpu();

    if( cv::gpu::getCudaEnabledDeviceCount()<=0 )
    {
        cout << endl << "No CUDA enabled GPU found. Performances comparison is not available. " << endl << endl;

        return EXIT_SUCCESS;
    }

    testGpu();

    if( !cv::gpu::CudaMem::canMapHostMemory() )
    {
        cout << endl << "GPU device has not shared memory with CPU, so no 'ZEROCOPY' test can be performed. " << endl << endl;

        return EXIT_SUCCESS;
    }

    testGpuZeroCopy();

    testGpuMemManaged();

    return EXIT_SUCCESS;
}

void testCpu()
{
    cout << "Testing performances of CPU..." << endl << endl;

    globalData.totResizeCpu_msec=0;
    globalData.totBlurCpu_msec=0;
    globalData.totCvtCpu_msec=0;
    globalData.totCannyCpu_msec=0;
    globalData.totCpu_msec=0;

    cv::Mat result;

    for( int i=0; i<globalData.count; i++)
    {
        Timer procTmr,elabTmr;

        procTmr.tic();

        elabTmr.tic();
        cv::Mat elab;
        globalData.source.copyTo(elab);
        globalData.totMemUpCpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::Mat resized;
        cv::resize( elab, resized, cv::Size(1280,720), CV_INTER_AREA );
        globalData.totResizeCpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::Mat gray;
        cv::cvtColor( resized, gray, CV_RGB2GRAY );
        globalData.totCvtCpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::Mat blurred;
        cv::blur( gray, blurred, cv::Size(3,3) );
        globalData.totBlurCpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::Canny( blurred, result, 150, 100 );
        globalData.totCannyCpu_msec += elabTmr.toc();

        globalData.totCpu_msec += procTmr.toc();
    }

    cout << " Memory \t " << "Total: " << globalData.totMemUpCpu_msec << " msec \t Mean: " <<
            globalData.totMemUpCpu_msec/globalData.count << " msec" << endl;

    cout << " Resize \t " << "Total: " << globalData.totResizeCpu_msec << " msec \t Mean: " <<
            globalData.totResizeCpu_msec/globalData.count << " msec" << endl;

    cout << " RGB2Gray \t " << "Total: " << globalData.totCvtCpu_msec << " msec \t Mean: " <<
            globalData.totCvtCpu_msec/globalData.count << " msec" << endl;

    cout << " Blur \t\t " << "Total: " << globalData.totBlurCpu_msec << " msec \t Mean: " <<
            globalData.totBlurCpu_msec/globalData.count << " msec" << endl;

    cout << " Canny \t\t " << "Total: " << globalData.totCannyCpu_msec << " msec \t Mean: " <<
            globalData.totCannyCpu_msec/globalData.count << " msec" << endl;

    cout << "---------------------------------------------------------------" << endl;
    cout << "Process \t " << "Total: " << globalData.totCpu_msec << " msec \t Mean: " <<
            globalData.totCpu_msec/globalData.count << " msec" << endl << endl;
}

void testGpu()
{
    cout << "Testing performances GPU with memory copy..." << endl << endl;

    globalData.totMemUpGpu_msec=0;
    globalData.totResizeGpu_msec=0;
    globalData.totBlurGpu_msec=0;
    globalData.totCvtGpu_msec=0;
    globalData.totCannyGpu_msec=0;
    globalData.totGpu_msec=0;

    cv::gpu::GpuMat result;

    for( int i=0; i<globalData.count; i++)
    {
        Timer procTmr,elabTmr;

        procTmr.tic();

        elabTmr.tic();
        cv::gpu::GpuMat elab;
        elab.upload( globalData.source );
        globalData.totMemUpGpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat resized;
        cv::gpu::resize( elab, resized, cv::Size(1280,720), CV_INTER_AREA );
        globalData.totResizeGpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat gray;
        cv::gpu::cvtColor( resized, gray, CV_BGR2GRAY );
        globalData.totCvtGpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat blurred;
        cv::gpu::blur( gray, blurred, cv::Size(3,3) );
        globalData.totBlurGpu_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::Canny( blurred, result, 150, 100 );
        globalData.totCannyGpu_msec += elabTmr.toc();

        globalData.totGpu_msec += procTmr.toc();
    }

    cout << " Memory \t " << "Total: " << globalData.totMemUpGpu_msec << " msec \t Mean: " <<
            globalData.totMemUpGpu_msec/globalData.count << " msec" << endl;

    cout << " Resize \t " << "Total: " << globalData.totResizeGpu_msec << " msec \t Mean: " <<
            globalData.totResizeGpu_msec/globalData.count << " msec" << endl;

    cout << " RGB2Gray \t " << "Total: " << globalData.totCvtGpu_msec << " msec \t Mean: " <<
            globalData.totCvtGpu_msec/globalData.count << " msec" << endl;

    cout << " Blur \t\t " << "Total: " << globalData.totBlurGpu_msec << " msec \t Mean: " <<
            globalData.totBlurGpu_msec/globalData.count << " msec" << endl;

    cout << " Canny \t\t " << "Total: " << globalData.totCannyGpu_msec << " msec \t Mean: " <<
            globalData.totCannyGpu_msec/globalData.count << " msec" << endl;

    cout << "---------------------------------------------------------------" << endl;
    cout << "Process \t " << "Total: " << globalData.totGpu_msec << " msec \t Mean: " <<
            globalData.totGpu_msec/globalData.count << " msec" << endl << endl;
}

void testGpuZeroCopy()
{
    cout << "Testing performances of GPU with ZEROCOPY..." << endl << endl;

    globalData.totMemUpGpuZc_msec=0;
    globalData.totResizeGpuZc_msec=0;
    globalData.totBlurGpuZc_msec=0;
    globalData.totCvtGpuZc_msec=0;
    globalData.totCannyGpuZc_msec=0;
    globalData.totGpuZc_msec=0;

    cv::gpu::GpuMat result;

    for( int i=0; i<globalData.count; i++)
    {
        Timer procTmr,elabTmr;

        procTmr.tic();

        elabTmr.tic();
        cv::gpu::CudaMem elabMem( globalData.source, cv::gpu::CudaMem::ALLOC_ZEROCOPY );
        cv::gpu::GpuMat elab;
        elab = elabMem.createGpuMatHeader();
        globalData.totMemUpGpuZc_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat resized;
        cv::gpu::resize( elab, resized, cv::Size(1280,720), CV_INTER_AREA );
        globalData.totResizeGpuZc_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat gray;
        cv::gpu::cvtColor( resized, gray, CV_BGR2GRAY );
        globalData.totCvtGpuZc_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat blurred;
        cv::gpu::blur( gray, blurred, cv::Size(3,3) );
        globalData.totBlurGpuZc_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::Canny( blurred, result, 150, 100 );
        globalData.totCannyGpuZc_msec += elabTmr.toc();

        globalData.totGpuZc_msec += procTmr.toc();
    }

    /*cv::Mat res;
    result.download(res);
    cv::imshow( "GPU ZeroCopy",res );
    cv::waitKey();*/

    cout << " Memory \t " << "Total: " << globalData.totMemUpGpuZc_msec << " msec \t Mean: " <<
            globalData.totMemUpGpuZc_msec/globalData.count << " msec" << endl;

    cout << " Resize \t " << "Total: " << globalData.totResizeGpuZc_msec << " msec \t Mean: " <<
            globalData.totResizeGpuZc_msec/globalData.count << " msec" << endl;

    cout << " RGB2Gray \t " << "Total: " << globalData.totCvtGpuZc_msec << " msec \t Mean: " <<
            globalData.totCvtGpuZc_msec/globalData.count << " msec" << endl;

    cout << " Blur \t\t " << "Total: " << globalData.totBlurGpuZc_msec << " msec \t Mean: " <<
            globalData.totBlurGpuZc_msec/globalData.count << " msec" << endl;

    cout << " Canny \t\t " << "Total: " << globalData.totCannyGpuZc_msec << " msec \t Mean: " <<
            globalData.totCannyGpuZc_msec/globalData.count << " msec" << endl;

    cout << "---------------------------------------------------------------" << endl;
    cout << "Process \t " << "Total: " << globalData.totGpuZc_msec << " msec \t Mean: " <<
            globalData.totGpuZc_msec/globalData.count << " msec" << endl << endl;
}

#include <cuda_runtime.h>

void testGpuMemManaged()
{
    cout << "Testing performances of GPU with Memory Managed..." << endl << endl;

    globalData.totMemGpuMan_msec=0;
    globalData.totResizeGpuMan_msec=0;
    globalData.totBlurGpuMan_msec=0;
    globalData.totCvtGpuMan_msec=0;
    globalData.totCannyGpuMan_msec=0;
    globalData.totGpuMan_msec=0;

    cv::gpu::GpuMat result;

    for( int i=0; i<globalData.count; i++)
    {
        Timer procTmr,elabTmr;

        procTmr.tic();

        elabTmr.tic();
        int w = globalData.source.cols;
        int h = globalData.source.rows;
        int ch = globalData.source.channels();
        int size = w*h*ch;

        uint8_t* mem;
        //cudaSetDeviceFlags(cudaDeviceMapHost); //  This flag must be set in order to allocate pinned host memory that is accessible to the device
        cudaMallocManaged( &mem, sizeof(uint8_t)*size );
        memcpy( mem, globalData.source.data, sizeof(uchar)*size );
        cv::gpu::GpuMat elab( globalData.source.size(), CV_8UC3, mem );
        globalData.totMemGpuMan_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat resized;
        cv::gpu::resize( elab, resized, cv::Size(1280,720), CV_INTER_AREA );
        globalData.totResizeGpuMan_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat gray;
        cv::gpu::cvtColor( resized, gray, CV_BGR2GRAY );
        globalData.totCvtGpuMan_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::GpuMat blurred;
        cv::gpu::blur( gray, blurred, cv::Size(3,3) );
        globalData.totBlurGpuMan_msec += elabTmr.toc();

        elabTmr.tic();
        cv::gpu::Canny( blurred, result, 150, 100 );
        globalData.totCannyGpuMan_msec += elabTmr.toc();

        cudaFree( mem );

        globalData.totGpuMan_msec += procTmr.toc();
    }

    /*cv::Mat res;
    result.download(res);
    cv::imshow( "GPU ZeroCopy",res );
    cv::waitKey();*/

    cout << " Memory \t " << "Total: " << globalData.totMemGpuMan_msec << " msec \t Mean: " <<
            globalData.totMemGpuMan_msec/globalData.count << " msec" << endl;

    cout << " Resize \t " << "Total: " << globalData.totResizeGpuMan_msec << " msec \t Mean: " <<
            globalData.totResizeGpuMan_msec/globalData.count << " msec" << endl;

    cout << " RGB2Gray \t " << "Total: " << globalData.totCvtGpuMan_msec << " msec \t Mean: " <<
            globalData.totCvtGpuMan_msec/globalData.count << " msec" << endl;

    cout << " Blur \t\t " << "Total: " << globalData.totBlurGpuMan_msec << " msec \t Mean: " <<
            globalData.totBlurGpuMan_msec/globalData.count << " msec" << endl;

    cout << " Canny \t\t " << "Total: " << globalData.totCannyGpuMan_msec << " msec \t Mean: " <<
            globalData.totCannyGpuMan_msec/globalData.count << " msec" << endl;

    cout << "---------------------------------------------------------------" << endl;
    cout << "Process \t " << "Total: " << globalData.totGpuMan_msec << " msec \t Mean: " <<
            globalData.totGpuMan_msec/globalData.count << " msec" << endl << endl;
}

void saveResImages()
{
    cv::Mat elab;

    cv::resize( globalData.source, elab, cv::Size(1280,720) );
    cv::imwrite( "00_resized.jpg", elab);

    cv::cvtColor( elab, elab, CV_RGB2GRAY );
    cv::imwrite( "01_gray.jpg", elab);

    cv::blur( elab, elab, cv::Size(3,3) );
    cv::imwrite( "02_blurred.jpg", elab);

    cv::Canny( elab, elab, 150, 100 );
    cv::imwrite( "03_canny.jpg", elab);
}

void printUsage()
{
    cout << "Usage: " << endl;
    cout << "\t opencv_cpu_vs_gpu <file_to_test> [iterations {100}]" << endl << endl;
}

void Timer::tic()
{
    mStart = std::chrono::high_resolution_clock::now();
}

double Timer::toc()
{
    auto end = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(end - mStart);

    double elapsed = static_cast<double>(dur.count())/1000;

    return elapsed;
}
