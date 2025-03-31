#pragma once
#include <opencv2/opencv.hpp>
#include "Tools.hpp"
namespace lvcv
{
class CMorph
{
//Member variables
private:
	cv::Size m_szKernel;
	cv::Mat m_kernel_cross;
	cv::Mat m_kernel_rect;
	cv::Mat m_kernel_ellipse;

//Constructor and destructor
public:
	CMorph()
	{
		m_szKernel = cv::Size(3, 3);
		m_kernel_cross = cv::getStructuringElement(cv::MORPH_CROSS, m_szKernel);
		m_kernel_rect = cv::getStructuringElement(cv::MORPH_RECT, m_szKernel);
		m_kernel_ellipse = cv::getStructuringElement(cv::MORPH_ELLIPSE, m_szKernel);
	}
	virtual ~CMorph()
	{

	}
//Member function
protected:

public:
	virtual int erodeImage(cv::Mat& imSrc, cv::Mat &imDst, int iIterations = 1)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (iIterations < 1)
			return -2;
		//Erode the image
		if (imSrc.channels() == 1)
			cv::erode(imSrc, imDst, m_kernel_cross, cv::Point(-1, -1), iIterations);
		else if (imSrc.channels() == 3)
		{
			//Convert color space from BGR to HSV
			cv::Mat imHsv;
			if (lvcv::CTools::convertBGR2HSV(imSrc, imHsv) < 0)
				return -1;
			//Split the hsv image to vIms
			std::vector<cv::Mat> vIms;
			cv::split(imHsv, vIms);
			//Erode the value channel image
			cv::erode(vIms[2], vIms[2], m_kernel_cross, cv::Point(-1, -1), iIterations);
			//Merge the vIms
			cv::merge(vIms, imDst);
			//Convert color sapce from HSV to BGR
			if (lvcv::CTools::convertHSV2BGR(imDst, imDst) < 0)
				return -1;
		}
		return 1;
	}

	virtual int dilateImage(cv::Mat& imSrc,cv::Mat&imDst, int iIterations = 1)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (iIterations < 1)
			return -2;
		//Dilate the image
		if (imSrc.channels() == 1)
			cv::dilate(imSrc,imDst, m_kernel_cross, cv::Point(-1, -1), iIterations);
		else if (imSrc.channels() == 3)
		{
			//Convert color space from BGR to HSV
			lvcv::CTools::convertBGR2HSV(imSrc, imDst);
			//Split the hsv image to vIms
			std::vector<cv::Mat> vIms;
			cv::split(imDst, vIms);
			//Dilate the value image in vIms
			cv::dilate(vIms[2], vIms[2], m_kernel_cross, cv::Point(-1, -1), iIterations);
			//Merge the vIms
			cv::merge(vIms, imDst);
			//Convert color space from HSV to BGR
			if (lvcv::CTools::convertHSV2BGR(imDst, imDst) < 0)
				return -1;
		}
		return 1;
	}

	virtual int openImage(cv::Mat& imSrc, cv::Mat& imDst, int iIterations = 1)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (iIterations < 1)
			return 1;
		//Open the input image
		if (imSrc.channels() == 1)
		{
			cv::morphologyEx(imSrc, imDst, cv::MORPH_OPEN, m_kernel_cross, 
				cv::Point(-1, -1), iIterations);
		}
		else if(imSrc.channels()==3)
		{
			//Convert color space from BGR to HSV
			lvcv::CTools::convertBGR2HSV(imSrc, imDst);
			//Split the hsv image vIms
			std::vector<cv::Mat> vIms;
			cv::split(imDst, vIms);
			//Open the value channel image 
			cv::morphologyEx(vIms[2], vIms[2], cv::MORPH_OPEN, m_kernel_cross, 
				cv::Point(-1, -1), iIterations);
			//Merge the vIms 
			cv::merge(vIms, imDst);
			//Convert color space from HSV to BGR
			lvcv::CTools::convertHSV2BGR(imDst, imDst);
		}
		return 1;
	}

	virtual int closeImage(cv::Mat& imSrc, cv::Mat& imDst, int iIterations = 1)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (iIterations < 0)
			return -1;
		//close the image
		if (imSrc.channels() == 1)
		{
			cv::morphologyEx(imSrc, imDst, cv::MORPH_CLOSE, m_kernel_cross, 
				cv::Point(-1, -1), iIterations);
		}
		else if (imSrc.channels() == 3)
		{
			//Convert the color space from BGR to HSV
			lvcv::CTools::convertBGR2HSV(imSrc, imDst);
			//Split the hsv image to vIms
			std::vector<cv::Mat> vIms;
			cv::split(imDst, vIms);
			//Close the value channel image 
			cv::morphologyEx(vIms[2], vIms[2], cv::MORPH_CLOSE, m_kernel_cross, 
				cv::Point(-1, -1), iIterations);
			//Merge the vIms
			cv::merge(vIms, imDst);
			//Convert the color space from HSV to BGR
			lvcv::CTools::convertHSV2BGR(imDst, imDst);
		}
		return 1;
	}

	virtual int detectEdge(cv::Mat& imSrc, cv::Mat& imEdge, int iIterations = 1)
	{
		//Judge the input parameters

		//Erode and dilate the value channel image
		cv::Mat imErode;
		cv::Mat imDilate;
		if (1 == imSrc.channels())
		{
			cv::erode(imSrc, imErode, m_kernel_cross, cv::Point(-1, -1), iIterations);
			cv::dilate(imSrc, imDilate, m_kernel_cross, cv::Point(-1, -1), iIterations);
		}
		else if (3 == imSrc.channels())
		{
			//Convert the color space from BGR to HSV
			cv::Mat imHsv;
			lvcv::CTools::convertBGR2HSV(imSrc, imHsv);
			//Extract the value channel image from the hsv image
			cv::Mat imValue;
			cv::extractChannel(imHsv,imValue, 2);
			//Erode the value channel image
			cv::erode(imValue, imErode, m_kernel_cross, cv::Point(-1, -1), iIterations);
			//Dilate the value channel image
			cv::dilate(imValue, imDilate, m_kernel_cross, cv::Point(-1, -1), iIterations);
		}
		//Compute the edge image
		cv::subtract(imDilate, imErode, imEdge);
		cv::threshold(imEdge, imEdge, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
		return 1;
	}

	virtual int detectEdgeMorphEx(cv::Mat& imSrc, cv::Mat& imEdge, int iIterations = 1)
	{
		//Judge the input parameters

		//Detect the edge from the 
		if (1 == imSrc.channels())
		{
			cv::morphologyEx(imSrc, imEdge, cv::MORPH_GRADIENT, m_kernel_cross,
				cv::Point(-1, -1), iIterations);
		}
		else if (3 == imSrc.channels())
		{
			//Convert the color space from BGR to HSV
			cv::Mat imHsv;
			lvcv::CTools::convertBGR2HSV(imSrc, imHsv);
			//Extract the value channel image fromm the hsv image
			cv::Mat imValue;
			cv::extractChannel(imHsv, imValue, 2);
			//Detect the edge from the value channel image
			cv::morphologyEx(imValue, imEdge, cv::MORPH_GRADIENT, m_kernel_cross, 
				cv::Point(-1, -1), iIterations);

		}
		return 1;
	}
//Static function
public:
};
}