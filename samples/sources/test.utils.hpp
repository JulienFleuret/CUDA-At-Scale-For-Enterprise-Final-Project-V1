#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/quality.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cvconfig.h>
#include "gpa.hpp"

#include <type_traits>
#include <unordered_set>
#include <vector>
#include <tuple>

#ifdef HAVE_CUDA
struct has_opencv_cuda : std::true_type {};
#else
struct has_opencv_cuda : std::false_type {};
#endif

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#pragma once

using namespace cv;
using namespace cuda;

namespace test
{

namespace
{

template<class T, bool use_nppi, bool use_cas>
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

    template<class U=T,std::enable_if_t<std::is_same<U, GpuMat>::value && !use_nppi && !use_cas, bool> = true>
    static void op(const T& img, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, T& dst, const double& eps)
    {
	if constexpr(has_opencv_cuda())
	{
	        cv::cuda::BilateralFilterGPA(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
        }

    }

    template<class U=T,std::enable_if_t<std::is_same<U, GpuMat>::value && use_nppi && !use_cas, bool> = true>
    static void op(const T& img, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, T& dst, const double& eps)
    {   
        nppi::BilateralFilterGPA(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
    }
    
    template<class U=T,std::enable_if_t<std::is_same<U, GpuMat>::value && !use_nppi && use_cas, bool> = true>
    static void op(const T& img, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, T& dst, const double& eps)
    {   
    	cas::GPA_FLAG cas_flag = flag == GPA_FLAG::BOX ? cas::GPA_FLAG::BOX : cas::GPA_FLAG::GAUSSIAN;
    
        cas::BilateralFilterGPA(img, sigma_range, sigma_spatial_or_box_width, cas_flag, dst, eps);
    }
};

} // anonymous

template<bool use_nppi, bool use_cas, class T>
T test_fun(const T& img, const Mat& img_ref, const double& sigma_range, const double& sigma_spatial_or_box_width, const GPA_FLAG& flag, const double& eps, const int& idx = -1)
{
    Ptr<quality::QualityMSE> metric;

    if(!img_ref.empty())
    {
        metric = quality::QualityMSE::create(img_ref);
    }

    T dst;

    if constexpr (use_nppi)
    {
        dst.create(img.size(), img.type());
    }

    TickMeter tic;

    tic.start();
    test_fun_helper_t<T, use_nppi, use_cas>::op(img, sigma_range, sigma_spatial_or_box_width, flag, dst, eps);
    tic.stop();



    if constexpr(std::is_same<T, Mat>())
    {
        std::cout<<"Computation time OpenCV CPU: "<<" : "<<tic.getTimeMilli()<<" ms"<<std::endl;
        std::cout<<"Error With The Reference: None Because It is The Reference."<<std::endl;
        imshow(idx < 0 ? "OpenCV CPU" : "OpenCV CPU " + std::to_string(idx), dst);
    }
    else if constexpr(std::is_same<T, UMat>())
    {
        Mat tmp;

        dst.copyTo(tmp);
        std::cout<<"Computation time OpenCV OpenCL: "<<tic.getTimeMilli()<<" ms"<<std::endl;
        std::cout<<"Error With The Reference: "<<metric->compute(tmp)<<std::endl;
        imshow(idx < 0 ? "OpenCV OpenCL" :  "OpenCV OpenCL " + std::to_string(idx), tmp);
    }
    else
    {
        Mat tmp(dst);

        if constexpr (use_nppi)
        {
            std::cout<<"Computation time NPPI: "<<tic.getTimeMilli()<<" ms"<<std::endl;
        }
        else if constexpr(use_cas)
        {
            std::cout<<"Computation time Custom Cuda Kernel: "<<tic.getTimeMilli()<<" ms"<<std::endl;
        }
        else if constexpr(has_opencv_cuda())
        {
            std::cout<<"Computation time OpenCV CUDA: "<<tic.getTimeMilli()<<" ms"<<std::endl;
        }

        std::cout<<"Error With The Reference: "<<metric->compute(tmp)<<std::endl;

        if constexpr (use_nppi)
        {
            imshow(idx < 0 ? "NPPI" : "NPPI " + std::to_string(idx), tmp);
        }
        else if constexpr(use_cas)
        {
            imshow(idx < 0 ? "CAS" : "CAS " + std::to_string(idx), tmp);
        }
        else
        {
            imshow(idx < 0 ? "OpenCV CUDA" : "OpenCV CUDA " + std::to_string(idx), tmp);
        }
    }

    return dst;
}

constexpr const char* getDefaultPath()
{
    return "/media/smile/3FA6E701592EA94E/prog/cudaAtScaleLib/samples/../samples_gpa_lib/third_party/opencv/samples/data";
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



String getAnImageRandomly(const String& path)
{
    using namespace utils::fs;

    RNG_MT19937 rng(std::time(nullptr));

    std::vector<std::string> filenames;

    std::vector extensions = {"*.jpg", "*.png"};

    for(const auto& extension : extensions)
    {
        std::vector<String> tmp;

        glob(path, extension, tmp);

        filenames.insert(filenames.end(), tmp.begin(), tmp.end());
    }

    int idx = rng.uniform(0, static_cast<int>(filenames.size()));

    return filenames.at(idx);
}

std::vector<String> getNImagesRandomly(const String& path, const int& N)
{

    using namespace utils::fs;

    std::vector<std::string> filenames;

    std::vector extensions = {"*.jpg", "*.png"};

    for(const auto& extension : extensions)
    {
        std::vector<String> tmp;

        glob(path, extension, tmp);

        filenames.insert(filenames.end(), tmp.begin(), tmp.end());
    }

    if(static_cast<int>(filenames.size()) < N)
        return filenames;


    RNG_MT19937 rng(std::time(nullptr));

    std::unordered_set<String> tmp;

    while(static_cast<int>(tmp.size()) != N)
    {
        int idx = rng.uniform(0, static_cast<int>(filenames.size()));
        tmp.insert(filenames.at(idx));
    }
    	
    return std::vector<String>(tmp.begin(), tmp.end());
}



namespace
{


std::tuple<String, String, int, GPA_FLAG, double, double, double> parseInputs(const int& argc, char*const argv[], const char* keys)
{
    cv::CommandLineParser parser(argc, argv, keys);
	
    if(parser.has("h") || parser.has("help") || parser.has("usage"))
        return std::make_tuple("", "", -1, GPA_FLAG::BOX, 0., 0., 0.);
	
    String path, filename;
    int N(-1);
    double sigma_range, sigma_spatial_or_box_width, epsilon;
    GPA_FLAG flag = GPA_FLAG::GAUSSIAN;
	
    bool is_demo1(true);

    try 
    {
        parser.has("filename");
    }
    catch (const cv::Exception& err)
    {
        (void)err;
        is_demo1 = false;
    }

    path = parser.get<String>("path");
    
    if(path.empty())
        path = getDefaultPath();
    
    if(is_demo1)
    {
	filename = parser.get<String>("filename");
        if(filename.empty())
	    filename = getAnImageRandomly(path);
    }
    else
        N = parser.get<int>("N");

    if(parser.get<String>("flag") != "gaussian")
    {
        flag = GPA_FLAG::BOX;
        sigma_spatial_or_box_width = parser.get<double>("box_width");
    }
    else
    {
        sigma_spatial_or_box_width = parser.get<double>("sigma_space");
    }
    
    sigma_range = parser.get<double>("sigma_range");
    
    epsilon = parser.get<double>("epsilon");
    
    return std::make_tuple(path, filename, N, flag, sigma_range, sigma_spatial_or_box_width, epsilon);
} 

} // anonymous

std::tuple<String, String, GPA_FLAG, double, double, double> parseInputsDemo1(const int& argc, char*const argv[])
{
    auto [path, filename, N, flag, sigma_range, sigma_spatial_or_box_width, epsilon] = parseInputs(argc, argv, getCommandLineKeysDemo1());

    (void)N;

    return std::make_tuple(path, filename, flag, sigma_range, sigma_spatial_or_box_width, epsilon);
}

std::tuple<String, int, GPA_FLAG, double, double, double> parseInputsDemo2(const int& argc, char*const argv[])
{
    auto [path, filename, N, flag, sigma_range, sigma_spatial_or_box_width, epsilon] = parseInputs(argc, argv, getCommandLineKeysDemo2());

    (void)filename;

    return std::make_tuple(path, N, flag, sigma_range, sigma_spatial_or_box_width, epsilon);
}



} // test

#endif
