#include "gpa.hpp"
#include "test.utils.hpp"



using namespace cv;
using namespace test;

int main(int argc, char* argv[])
{

    auto [folder, filename, flag, sigma_range, sigma_spatial_or_box_width, epsilon] = parseInputsDemo1(argc, argv);

    if(epsilon <= std::numeric_limits<double>::epsilon())
    {
        std::cout<<getCommandLineKeysDemo1() <<std::endl;
        return EXIT_SUCCESS;
    }

    if(!cv::utils::fs::exists(folder))
        CV_Error(cv::Error::StsError, "The specified folder does not exists!");

    if(!cv::utils::fs::exists(filename) && cv::utils::fs::exists(cv::utils::fs::join(folder, filename)))
        filename = cv::utils::fs::join(folder, filename);

    if(!cv::utils::fs::exists(filename) )
        CV_Error(cv::Error::StsError, "The specified filename does not exists!");

    cv::Mat img_host = imread(filename, IMREAD_UNCHANGED);

    if((img_host.depth() != CV_8U) && (img_host.depth() != CV_16U) && (img_host.depth() != CV_16S) && (img_host.depth() != CV_32F))
    {
        std::cout<<"This example only support 8 bits per pixel, unsigned, 16 bits per pixels signed and unsigned, and single precision floating points, image type. Please retry with an image that respect these constraints"<<std::endl;
        return EXIT_SUCCESS;
    }

    if(!img_host.empty())
        std::cout<<"Size: "<<img_host.size()<<" depth: "<<img_host.depth()<<" channels: "<<img_host.channels()<<std::endl;

    cv::Mat dst;
    cv::UMat usrc;
    cv::cuda::GpuMat img_device;
    cv::cuda::Stream stream;

    cv::imshow("Source", img_host);

    if(img_host.empty())
        return EXIT_SUCCESS;

    // First generate the reference.

    dst = test_fun<false, false>(img_host, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon);

    // Prepare Test with OpenCL (TAPI).

    img_host.copyTo(usrc);

    test_fun<false, false>(usrc, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon);

    // Prepare Test with CUDA.

    img_device.upload(img_host, stream);

    test_fun<false, false>(img_device, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon);

    // Prepare Test with NPPI.

    img_device.upload(img_host, stream);

    test_fun<true, false>(img_device, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon);

    // Prepare Test with CAS.

    img_device.upload(img_host, stream);

    test_fun<false, true>(img_device, dst, sigma_range, sigma_spatial_or_box_width, flag, epsilon);

    cv::waitKey(-1);


    return EXIT_SUCCESS;
}
