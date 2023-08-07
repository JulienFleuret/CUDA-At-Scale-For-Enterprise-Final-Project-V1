#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/quality.hpp>
#include <opencv2/cvconfig.hpp>
#include "gpa.hpp"

#include <type_traits>

#ifdef HAVE_CUDA
struct has_opencv_cuda : std::true_type
#else
struct has_opencv_cuda : std::false_type
#endif

namespace test
{

namespace
{

template<class T, bool use_nppi>
struct test_fun_helper_t
{
    template<class U=T,std::enable_if_t<std::is_same<U, Mat>::value, bool> = true>
    static void op(const T& img, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, T& dst, const double& eps)
    {
        cv::BilateralFilterGPA(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
    }

    template<class U=T,std::enable_if_t<std::is_same<U, UMat>::value, bool> = true>
    static void op(const T& img, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, T& dst, const double& eps)
    {
        cv::BilateralFilterGPA(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
    }

    template<class U=T,std::enable_if_t<std::is_same<U, GpuMat>::value && !use_nppi, bool> = true>
    static void op(const T& img, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, T& dst, const double& eps)
    {
	if constexpr(has_opencv_cuda())
	{
	        cv::cuda::BilateralFilterGPA(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
        }

    }

    template<class U=T,std::enable_if_t<std::is_same<U, GpuMat>::value && use_nppi, bool> = true>
    static void op(const T& img, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, T& dst, const double& eps)
    {
        nppi::BilateralFilterGPA(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
    }
};

} // anonymous

template<bool use_nppi, class T>
T test_fun(const T& img, const cv::Mat& img_ref, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, const double& eps)
{
    cv::Ptr<cv::quality::QualityMSE> metric;

    if(!img_ref.empty())
    {
        metric = cv::quality::QualityMSE::create(img_ref);
    }

    T dst;

    if constexpr (use_nppi)
    {
        dst.create(img.size(), img.type());
    }

    TickMeter tic;

    tic.start();
    test_fun_helper_t<T, use_nppi>::op(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
    tic.stop();



    if constexpr(std::is_same<T, Mat>())
    {
        std::cout<<"Computation time OpenCV CPU: "<<" : "<<tic.getTimeMilli()<<" ms"<<std::endl;
        std::cout<<"Error With The Reference: None Because It is The Reference."<<std::endl;
        imshow("OpenCV CPU", dst);
    }
    else if constexpr(std::is_same<T, UMat>())
    {
        Mat tmp;

        dst.copyTo(tmp);
        std::cout<<"Computation time OpenCV OpenCL: "<<tic.getTimeMilli()<<" ms"<<std::endl;
        std::cout<<"Error With The Reference: "<<metric->compute(tmp)<<std::endl;
        imshow("OpenCV OpenCL", tmp);
    }
    else
    {
        Mat tmp(dst);

        if constexpr (use_nppi)
        {
            std::cout<<"Computation time NPPI: "<<tic.getTimeMilli()<<" ms"<<std::endl;
        }
        else if constexpr(has_opencv_cuda())
        {
            std::cout<<"Computation time OpenCV CUDA: "<<tic.getTimeMilli()<<" ms"<<std::endl;
        }

        std::cout<<"Error With The Reference: "<<metric->compute(tmp)<<std::endl;

        if constexpr (use_nppi)
        {
            imshow("NPPI", tmp);
        }
        else
        {
            imshow("OpenCV CUDA", tmp);
        }
    }

    return dst;
}

constexpr const char* getDefaultPath()
{
    return "/home/smile/prog/cudaAtScaleLib/samples/third_party/opencv/samples/data";
}

constexpr bool shouldPathBeMandatory()
{
    return false;
}

constexpr const char* getCommandLineKeysDemo1()
{
    return "{help h usage ? | | print this message}{path | | folder to where to select the images}{filename | | image to process (by default an image will be randomly selected)}{flag |gaussian| gaussian or box}{sigma_range |40 | range kernel standard deviation}{sigma_space |4. | spatial kernel standard deviation. Used if flag is set to 'gaussian'}{box_width |40 | box filter width. Used if flag is set to 'box'}{epsilon |1e-3| desired accuracy.}";	
}

constexpr const char* getCommandLineKeysDemo2()
{
    return "{help h usage ? | | print this message}{path | | folder to where to select the images}{N |10 | number of images to proces}{flag |gaussian| gaussian or box}{sigma_range |40 | range kernel standard deviation}{sigma_space |4. | spatial kernel standard deviation. Used if flag is set to 'gaussian'}{box_width |40 | box filter width. Used if flag is set to 'box'}{epsilon |1e-3| desired accuracy.}";	
}

cv::String getAnImageRandomly(const cv::String& path)
{
    using namespace cv::utils::fs;

    cv::RNG_MT19937 rng(std::time(nullptr));

    std::vector<std::string> filenames;

    std::vector extensions = {"*.jpg", "*.png"};

    for(const auto& extension : extensions)
    {
        std::vector<cv::String> tmp;

        glob(path, extension, tmp);

        filenames.insert(filenames.end(), tmp.begin(), tmp.end());
    }

    int idx = rng.uniform(0, static_cast<int>(filenames.size()));

    return filenames.at(idx);
}

} // test
