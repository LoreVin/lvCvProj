#pragma once
#include <opencv2/opencv.hpp>
#include <io.h>
namespace lvcv
{

class CTools
{
//Member variable
private:

//Constructor and Destructor
public:
	CTools(){}
	virtual ~ CTools() {}

//Member function
public:

//Static function
public:
	static int convertToGray(const cv::Mat& imSrc, cv::Mat& imGray)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		//Convert color to gray image
		if (1 == imSrc.channels())
			imSrc.copyTo(imGray);
		else if (3 == imSrc.channels())
			cv::cvtColor(imSrc, imGray, cv::COLOR_BGR2GRAY);
		if (imGray.empty())
			return -2;
		return 1;
	}

	static int convertGray2BGR(const cv::Mat& imGray, cv::Mat& imBGR)
	{
		if (imGray.channels() > 1)
			return -1;
		cv::cvtColor(imGray, imBGR, cv::COLOR_GRAY2BGR);
		return 1;
	}
	static int convertBGR2HSV(const cv::Mat& imBGR, cv::Mat& imHsv)
	{
		//Judge the input parameters
		if (imBGR.empty())
			return -1;
		if (1 == imBGR.channels())
			return -1;
		//Convert color space from BGR to HSV
		cv::cvtColor(imBGR, imHsv, cv::COLOR_BGR2HSV);
		return 1;
	}

	static int convertHSV2BGR(const cv::Mat& imHsv, cv::Mat& imBGR)
	{
		//Judge the input parameters
		if (imHsv.empty() || 1==imHsv.channels())
			return -1;
		//Convert color space from HSV to BGR
		cv::cvtColor(imHsv, imBGR, cv::COLOR_HSV2BGR);
		if (imBGR.empty())
			return -1;
		return 1;
	}

	static int drawKps(cv::Mat& imSrc, std::vector<cv::KeyPoint>& vKps)
	{
		//Judge parameters
		if (imSrc.empty())
			return -1;
		if (vKps.empty())
			return -2;
		//Draw keypoints in the image
		cv::Scalar color(0,255, 0);
		for(cv::KeyPoint&kp: vKps)
		{
			cv::circle(imSrc, kp.pt, 3, color, 1);
		}
		return 1;
	}

	template<typename T>
	static int drawKps(cv::Mat& imSrc, std::vector<cv::Point_<T>>& vPnts)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (vPnts.empty())
			return -2;
		//Draw keypoints in the image
		cv::Scalar color{ 0, 255, 0 };
		for (cv::Point_<T>& pt : vPnts)
			cv::circle(imSrc, pt,3, color, 1);
		return 1;
	}

	static int drawMatchingRes(const cv::Mat& imSrc01,
		const cv::Mat& imSrc02,
		const std::vector<cv::KeyPoint>& vKps01,
		const std::vector<cv::KeyPoint>& vKps02,
		const std::vector<cv::DMatch>& vMatches,
		cv::Mat &imDrawResult)
	{
		//Judge the input parameters

		//Construct result image(imDrawResult)
		int iRows=0, iCols =0;
		iRows = std::max(imSrc01.rows, imSrc02.rows);
		iCols = imSrc01.cols + imSrc02.cols;
		imDrawResult.create(iRows, iCols, imSrc01.type());
		cv::Rect rect(0, 0, imSrc01.cols, imSrc01.rows);
		imSrc01.copyTo(imDrawResult(rect));
		rect.x = imSrc01.cols;
		rect.y = 0; 
		rect.width = imSrc02.cols;
		rect.height = imSrc02.rows;
		imSrc02.copyTo(imDrawResult(rect));
		//Draw keypoints and lines in the result image
		cv::Point pt01, pt02;
		cv::Scalar color(0, 255, 0);
		for (const cv::DMatch& match : vMatches)
		{
			pt01 = vKps01[match.queryIdx].pt;
			pt02.x = imSrc01.cols + (int)vKps02[match.trainIdx].pt.x;
			pt02.y = (int)vKps02[match.trainIdx].pt.y;
			cv::circle(imDrawResult, pt01,3, color, 1);
			cv::circle(imDrawResult, pt02,3, color, 1);
			cv::line(imDrawResult, pt01, pt02, color, 1);

		}
		return 1;
	}

	static int getFilenamesFromDir(const char* strRootPath,
		std::vector<std::string>&vStrFilenames)
	{
		if (nullptr == strRootPath)
			return -1;
		if (!vStrFilenames.empty())
			vStrFilenames.clear();
		intptr_t handle; //file descriptor
		std::string strTmp;
		strTmp.assign(strRootPath).append("\\*");
		_finddata_t fileInfo;
		handle = _findfirst(strTmp.c_str(), &fileInfo);
		if (handle ==-1)
		{
			std::cerr << "Can not find file!" << std::endl;
			return -1;
		}
		while (_findnext(handle, &fileInfo)==0)
		{
			if (!(fileInfo.attrib & _A_SUBDIR))
			{
				strTmp.assign(strRootPath).append("\\");
				strTmp.append(fileInfo.name);
				vStrFilenames.push_back(strTmp);
			}
		}
		return 1;
	}

	static bool existedFileOrDirectory(const std::string& strTmp)
	{
		if (strTmp.empty())
			return false;
		return (_access(strTmp.c_str(), 0) == 0);
	}

};
}