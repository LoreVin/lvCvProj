#pragma once
#include <opencv2/opencv.hpp>
#include "Tools.hpp"
namespace lvcv
{

class CHistogram
{
//Member variable
private:
	int m_channels[3]{ 0,0,0 };
	int m_histSizes[3]{ 256,256,256 };
	cv::Mat m_imMask;
	float m_range[2]{ 0, 256.0 };
	float m_rangeHue[2]{ 0,180.0 };
	const float* m_Ranges[3]{ m_range,m_range,m_range };
	const float* m_RangeHue[1]{ m_rangeHue };

//Constructor and Destructor
public:
	CHistogram() {}
	virtual ~CHistogram() {}

//Member function
public:

	virtual int computeHistogram(const cv::Mat& imSrc, cv::Mat& imHist)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;

		//Calculate the histogram of the gray image or color
		if (1 == imSrc.channels())
			cv::calcHist(&imSrc, 1, &m_channels[0], m_imMask,
				imHist, 1, &m_histSizes[0], &m_Ranges[0]);
		else
		{
			cv::Mat imCur;
			cv::Mat imHistTmp;
			std::vector<cv::Mat> vImHist;
			for (int i = 0; i < imSrc.channels(); ++i)
			{
				cv::extractChannel(imSrc, imCur, i);
				cv::calcHist(&imCur, 1, &m_channels[i], m_imMask,
					imHistTmp, 1, &m_histSizes[i], &m_Ranges[i]);
				vImHist.push_back(imHistTmp);
				imHistTmp.release();
			}
			cv::hconcat(vImHist, imHist);
		}
		return 1;
	}

	virtual int computeHueHistogram(const cv::Mat& imColor, cv::Mat& imHist)
	{
		//Judge input parameters
		if (imColor.empty())
			return -1;
		//Extract hue channel image from the image
		cv::Mat imHue;
		if (imColor.channels() == 1)
			imColor.copyTo(imHue);
		cv::Mat imHsv;
		cv::cvtColor(imColor, imHsv, cv::COLOR_BGR2HSV);
		cv::extractChannel(imHsv, imHue, 0);
		if (imHue.empty())
			return -2;
		//Calculate the hist of the hue image
		if (!imHist.empty())
			imHist.release();
		cv::calcHist(&imHue, 1, &m_channels[0], m_imMask, imHist, 1, &m_histSizes[0], m_RangeHue);
		if (imHist.empty())
			return -3;
		return 1;

	}

	virtual int computeBackproject(const cv::Mat& imSrc, cv::Mat& imHistTarget, cv::Mat& imBackproject)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		//Compute the backproject of the input image
		if (1 == imSrc.channels())
			cv::calcBackProject(&imSrc, 1, &m_channels[0], imHistTarget, imBackproject, &m_Ranges[0]);
		else if (3 == imSrc.channels())
			cv::calcBackProject(&imSrc, 1, m_channels, imHistTarget, imBackproject, m_Ranges);
		if (imBackproject.empty())
			return -2;
		return 1;
	}

	virtual int computeHueBackproject(const cv::Mat& imColor, cv::Mat& imHistHueTarget, cv::Mat& imBack)
	{
		//Judge the input parameters
		if (imColor.empty())
			return -1;
		if (imHistHueTarget.empty())
			return -2;
		if (imColor.channels() <= 2)
			return -3;
		//Extract hue image from the corresponding image
		cv::Mat imHue;
		cv::Mat imHsv;
		cv::cvtColor(imColor, imHsv, cv::COLOR_BGR2HSV);
		cv::extractChannel(imHsv, imHue, 0);
		if (imHue.empty())
			return -4;
		//Compute backproject of the hue image
		cv::calcBackProject(&imHue, 1, &m_channels[0], imHistHueTarget, imBack, m_RangeHue);
		if (imBack.empty())
			return -5;
		return 1;
	}
	virtual int equalizeImage(const cv::Mat& imInput, cv::Mat& imEqulized)
	{
		//Judge the input parameters
		if (imInput.empty())
			return -1;
		//Equalize the image
		if (imInput.channels() > 1)
		{
			//Convert BGR to HSV space
			cv::Mat imHsv;
			cv::cvtColor(imInput, imHsv, cv::COLOR_BGR2HSV);
			if (imHsv.empty())
				return -1;
			//Split HSV to a image vector
			std::vector<cv::Mat> vIms;
			cv::split(imHsv, vIms);
			//Equalize the value channel image 
			cv::equalizeHist(vIms[2], vIms[2]);
			//Merge the image vector
			cv::merge(vIms, imEqulized);
			//Convert color space from hsv to bgr
			cv::cvtColor(imEqulized, imEqulized, cv::COLOR_HSV2BGR);
		}
		else
		{
			cv::equalizeHist(imInput, imEqulized);
		}
		return 1;
	}

	virtual int compareHist(const cv::Mat& imSrc01, const cv::Mat& imSrc02, double& dRes)
	{
		//Judge the input parameters
		if (imSrc01.empty() || imSrc02.empty())
			return -1;
		if (imSrc01.channels() != imSrc02.channels())
			return -2;
		//Convert the color image to gray
		cv::Mat imGray01, imGray02;
		if (CTools::convertToGray(imSrc01, imGray01) < 0)
			return -1;
		if (CTools::convertToGray(imSrc02, imGray02) < 0)
			return -1;
		//Calculate the hist of the input images
		cv::Mat imHist01, imHist02;
		computeHistogram(imGray01, imHist01);
		computeHistogram(imGray02, imHist02);
		//Normalize the hist of the input images
		cv::normalize(imHist01, imHist01, 1, cv::NORM_MINMAX);
		cv::normalize(imHist02, imHist02, 1, cv::NORM_MINMAX);
		//Compare the hist of the two images
		dRes = cv::compareHist(imHist01, imHist02, cv::HISTCMP_INTERSECT);
		return 1;
	}

	virtual int compareHueHist(const cv::Mat& imSrc01, const cv::Mat& imSrc02, double& dRes, 
		const int&iMinSta=30)
	{
		//Judge the input parameters
		if (imSrc01.empty() || imSrc02.empty())
			return -1;
		if (imSrc01.channels() < 3 || imSrc02.channels() < 3)
			return -2;
		//Convert color space from BGR to HSV
		cv::Mat imHsv01, imHsv02;
		CTools::convertBGR2HSV(imSrc01, imHsv01);
		CTools::convertBGR2HSV(imSrc02, imHsv02);
		//Extract hue and saturation channels from the hsv image
		cv::Mat imHue01, imHue02;
		cv::Mat imSta01, imSta02;
		cv::extractChannel(imHsv01, imHue01, 0);
		cv::extractChannel(imHsv02, imHue02, 0);
		cv::extractChannel(imHsv01, imSta01, 1);
		cv::extractChannel(imHsv02, imSta02, 1);
		imHsv01.release();
		imHsv02.release();
		//Construct mask according to saturation channel image
		cv::Mat imMask01, imMask02;
		if (iMinSta > 0)
		{
			cv::threshold(imSta01, imMask01, iMinSta, 255, cv::THRESH_BINARY);
			cv::threshold(imSta02, imMask02, iMinSta, 255, cv::THRESH_BINARY);
		}
		//Calculate the hist of the two hue images
		cv::Mat imHist01, imHist02;
		computeHistogram(imHue01, imHist01);
		computeHistogram(imHue02, imHist02);
		//Normalize the hist
		cv::normalize(imHist01, imHist01, 1, cv::NORM_MINMAX);
		cv::normalize(imHist02, imHist02, 1, cv::NORM_MINMAX);
		//Compare the hist
		cv::compareHist(imHist01, imHist02, cv::HISTCMP_INTERSECT);
		return 1;
	}

	virtual int convertHist2Image(const cv::Mat& imHist, cv::Mat& imDst)
	{
		//Judge the input parameters
		if (imHist.empty())
			return -1;
		if (imHist.rows != 256 || imHist.cols != 1)
			return -2;
		//Find the maximum and minimum values from the imHist
		double dMin=0.0, dMax=0.0;
		cv::minMaxLoc(imHist, &dMin, &dMax, 0, 0);
		//Allocate space for the output image
		imDst.create((int)dMax, 256, CV_8UC1);
		imDst.setTo(0);
		//Loop hist for assigning values to output image
		const float* dataHist = nullptr;
		uchar* dataDst = nullptr;
		cv::Point pt01=cv::Point(0,0), pt02(0,0);
		cv::Scalar color(255);
		for (int i = 0; i < imHist.rows; ++i)
		{
			dataHist = imHist.ptr<float>(i);
			pt01.x = pt02.x = i;
			pt02.y = (int)*dataHist;
			cv::line(imDst, pt01, pt02, color, 1);
		}
		cv::flip(imDst, imDst, 0);
		return 1;
	}
		
//Static function
public:

};//end of class


}//end of namespace lvcv
