#include <main.hpp>

#include <string>

using namespace std;
using namespace chrono;

typedef struct _data
{
    string srcFile;
    cv::Mat source;
    int count;

    double totResizeCpu_msec;
    double totBlurCpu_msec;
    double totCvtCpu_msec;
    double totCannyCpu_msec;
    double totCpu_msec;

    double totResizeGpu_msec;
    double totBlurGpu_msec;
    double totCvtGpu_msec;
    double totCannyGpu_msec;
    double totGpu_msec;
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

    return EXIT_SUCCESS;
}

void testCpu()
{
    cout << "Testing CPU performances..." << endl << endl;

    globalData.totResizeCpu_msec=0;
    globalData.totBlurCpu_msec=0;
    globalData.totCvtCpu_msec=0;
    globalData.totCannyCpu_msec=0;
    globalData.totCpu_msec=0;

    for( int i=0; i<globalData.count; i++)
    {
        auto proc_start = high_resolution_clock::now();

        cv::Mat elab;

        auto start = high_resolution_clock::now();
        cv::resize( globalData.source, elab, cv::Size(1280,720) );
        auto end = high_resolution_clock::now();
        auto dur = duration_cast<microseconds>(end - start);
        globalData.totResizeCpu_msec += static_cast<double>(dur.count())/1000;

        start = high_resolution_clock::now();
        cv::blur( elab, elab, cv::Size(3,3) );
        end = high_resolution_clock::now();
        dur = duration_cast<microseconds>(end - start);
        globalData.totBlurCpu_msec += static_cast<double>(dur.count())/1000;

        start = high_resolution_clock::now();
        cv::cvtColor( elab, elab, CV_RGB2GRAY );
        end = high_resolution_clock::now();
        dur = duration_cast<microseconds>(end - start);
        globalData.totCvtCpu_msec += static_cast<double>(dur.count())/1000;

        start = high_resolution_clock::now();
        cv::Canny( elab, elab, 150, 100 );
        end = high_resolution_clock::now();
        dur = duration_cast<microseconds>(end - start);
        globalData.totCannyCpu_msec += static_cast<double>(dur.count())/1000;

        auto proc_end = high_resolution_clock::now();
        auto proc_dur = duration_cast<microseconds>(proc_end - proc_start);
        globalData.totCpu_msec += static_cast<double>(proc_dur.count())/1000;
    }

    cout << " Resize \t " << "Total: " << globalData.totResizeCpu_msec << " msec \t Mean: " <<
            globalData.totResizeCpu_msec/globalData.count << " msec" << endl;

    cout << " Blur \t " << "Total: " << globalData.totBlurCpu_msec << " msec \t Mean: " <<
            globalData.totBlurCpu_msec/globalData.count << " msec" << endl;

    cout << " RGB2Gray \t " << "Total: " << globalData.totCvtCpu_msec << " msec \t Mean: " <<
            globalData.totCvtCpu_msec/globalData.count << " msec" << endl;

    cout << " Canny \t " << "Total: " << globalData.totCannyCpu_msec << " msec \t Mean: " <<
            globalData.totCannyCpu_msec/globalData.count << " msec" << endl;

    cout << "---------------------------------------------------------------" << endl;
    cout << "Process \t " << "Total: " << globalData.totCpu_msec << " msec \t Mean: " <<
            globalData.totCpu_msec/globalData.count << " msec" << endl;
}

void testGpu()
{
    cout << "Testing GPU performances..." << endl << endl;

    globalData.totResizeGpu_msec=0;
    globalData.totBlurGpu_msec=0;
    globalData.totCvtGpu_msec=0;
    globalData.totCannyGpu_msec=0;
    globalData.totGpu_msec=0;

    for( int i=0; i<globalData.count; i++)
    {
        auto proc_start = high_resolution_clock::now();


        cv::gpu::GpuMat elab;
        elab.download( globalData.source );

        auto start = high_resolution_clock::now();
        cv::gpu::resize( elab, elab, cv::Size(1280,720) );
        auto end = high_resolution_clock::now();
        auto dur = duration_cast<microseconds>(end - start);
        globalData.totResizeGpu_msec += static_cast<double>(dur.count())/1000;

        start = high_resolution_clock::now();
        cv::gpu::blur( elab, elab, cv::Size(3,3) );
        end = high_resolution_clock::now();
        dur = duration_cast<microseconds>(end - start);
        globalData.totBlurGpu_msec += static_cast<double>(dur.count())/1000;

        start = high_resolution_clock::now();
        cv::gpu::cvtColor( elab, elab, CV_RGB2GRAY );
        end = high_resolution_clock::now();
        dur = duration_cast<microseconds>(end - start);
        globalData.totCvtGpu_msec += static_cast<double>(dur.count())/1000;

        start = high_resolution_clock::now();
        cv::gpu::Canny( elab, elab, 150, 100 );
        end = high_resolution_clock::now();
        dur = duration_cast<microseconds>(end - start);
        globalData.totCannyGpu_msec += static_cast<double>(dur.count())/1000;

        auto proc_end = high_resolution_clock::now();
        auto proc_dur = duration_cast<microseconds>(proc_end - proc_start);
        globalData.totGpu_msec += static_cast<double>(proc_dur.count())/1000;
    }

    cout << " Resize \t " << "Total: " << globalData.totResizeGpu_msec << " msec \t Mean: " <<
            globalData.totResizeGpu_msec/globalData.count << " msec" << endl;

    cout << " Blur \t " << "Total: " << globalData.totBlurGpu_msec << " msec \t Mean: " <<
            globalData.totBlurGpu_msec/globalData.count << " msec" << endl;

    cout << " RGB2Gray \t " << "Total: " << globalData.totCvtGpu_msec << " msec \t Mean: " <<
            globalData.totCvtGpu_msec/globalData.count << " msec" << endl;

    cout << " Canny \t " << "Total: " << globalData.totCannyGpu_msec << " msec \t Mean: " <<
            globalData.totCannyGpu_msec/globalData.count << " msec" << endl;

    cout << "---------------------------------------------------------------" << endl;
    cout << "Process \t " << "Total: " << globalData.totGpu_msec << " msec \t Mean: " <<
            globalData.totGpu_msec/globalData.count << " msec" << endl;
}

void testGpuZeroCopy()
{}

void saveResImages()
{
    cv::Mat elab;

    cv::resize( globalData.source, elab, cv::Size(1280,720) );
    cv::imwrite( "00_resized.jpg", elab);

    cv::blur( elab, elab, cv::Size(3,3) );
    cv::imwrite( "01_blurred.jpg", elab);

    cv::cvtColor( elab, elab, CV_RGB2GRAY );
    cv::imwrite( "02_gray.jpg", elab);

    cv::Canny( elab, elab, 150, 100 );
    cv::imwrite( "03_canny.jpg", elab);
}

void printUsage()
{
    cout << "Usage: " << endl;
    cout << "\t opencv_cpu_vs_gpu <file_to_test> [iterations {100}]" << endl << endl;
}
