#include "gpa.hpp"
#include "test.utils.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace test;

int main(int argc, char* argv[])
{

    auto [folder, N, flag, sigma_range, sigma_spatial_or_box_width, epsilon] = parseInputsDemo2(argc, argv);

    if(epsilon <= std::numeric_limits<double>::epsilon())
    {
        std::cout<<getCommandLineKeysDemo2() <<std::endl;
        return EXIT_SUCCESS;
    }


    std::vector filenames = getNImagesRandomly(folder, N);

    int i=-1;
    for(const auto& filename : filenames)
    {
        ++i;

        Mat img_host = imread(filename, IMREAD_UNCHANGED);

        if((img_host.depth() != CV_8U) && (img_host.depth() != CV_16U) && (img_host.depth() != CV_16S) && (img_host.depth() != CV_32F))
        {
            std::cout<<"This example only support 8 bits per pixel, unsigned, 16 bits per pixels signed and unsigned, and single precision floating points, image type. Please retry with an image that respect these constraints"<<std::endl;
            return EXIT_SUCCESS;
        }

        if(img_host.empty())
            continue;

        Mat dst;
        UMat usrc;
        GpuMat img_device;
        Stream stream;

        imshow("Source " + std::to_string(i), img_host);


        // First generate the reference.

        dst = test_fun<false, false>(img_host, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon, i);

        // Prepare Test with OpenCL (TAPI).

        img_host.copyTo(usrc);

        test_fun<false, false>(usrc, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon, i);

        // Prepare Test with CUDA.

        img_device.upload(img_host, stream);

        test_fun<false, false>(img_device, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon, i);

        // Prepare Test with NPPI.

        img_device.upload(img_host, stream);

        test_fun<true, false>(img_device, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon, i);

        // Prepare Test with CAS.

        img_device.upload(img_host, stream);

        test_fun<false, true>(img_device, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon);

    }

    waitKey(-1);


    return EXIT_SUCCESS;
}

//#include <cstdlib>
//int main()
//{
//    return EXIT_SUCCESS;
//}
