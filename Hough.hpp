#pragma once 
#include <opencv2/opencv.hpp>
namespace lvcv
{

class CHough
{
//Member variables
private:
	cv::Size m_szBlur;
	cv::Mat m_imTmp;
	cv::Scalar m_color;
//Constructor and destructor
public:
	CHough() {
		m_szBlur = cv::Size(3, 3);
		m_color = cv::Scalar(0, 255, 0);
	}
	virtual ~CHough() {}

//Member function
protected:

public:
	virtual int detectLines(const cv::Mat& imSrc,
		std::vector<cv::Vec4i>& vLines, 
		cv::Mat* imDraw = nullptr)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (!vLines.empty())
			return -2;
		
		//Pre-process the image
		cv::GaussianBlur(imSrc, m_imTmp, m_szBlur,0);
		//Canny the input image
		cv::Mat imEdge;
		cv::Canny(m_imTmp, imEdge, 100, 300);
		//Detect the lines by hough algorithm
		double dRho = 1.0;
		double dTheta = CV_PI / 180.0;
		int iMinVote = 50;
		cv::HoughLinesP(imEdge, vLines, dRho, dTheta, iMinVote, 100.0, 20.0);
		if (nullptr != imDraw)
		{
			imSrc.copyTo(*imDraw);
			cv::Point pt01, pt02;
			for (int i = 0; i < vLines.size(); ++i)
			{
				pt01.x = vLines[i][0];
				pt01.y = vLines[i][1];
				pt02.x = vLines[i][2];
				pt02.y = vLines[i][3];
				cv::line(*imDraw, pt01, pt02, m_color, 1);
			}
		}
		return 1;
	}

	virtual int detectCircles(const cv::Mat& imSrc,
		std::vector<cv::Vec3f>& vCircles,
		cv::Mat* imDraw = nullptr)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (!vCircles.empty())
			vCircles.clear();
		//Pre-process the input image
		cv::Mat imGray;
		lvcv::CTools::convertToGray(imSrc, imGray);
		cv::GaussianBlur(imGray, imGray, m_szBlur, 0);
		//Detect circles from the image
		double dp = 1.0;
		double dMinDis = 50.0;
		double dMaxThresh = 160;
		double dMinVote = 60;
		int iMaxRadius =100;
		int iMinRadius = 10;
		cv::HoughCircles(imGray, vCircles, cv::HOUGH_GRADIENT,
			dp, dMinDis, dMaxThresh,dMinVote, iMinRadius, iMaxRadius);
		//Draw circles in the corresponding image
		if (nullptr != imDraw)
		{
			imSrc.copyTo(*imDraw);
			cv::Point PCenter;
			for (int i = 0; i < vCircles.size(); ++i)
			{
				PCenter.x = vCircles[i][0];
				PCenter.y = vCircles[i][1];
				cv::circle(*imDraw, PCenter, vCircles[i][2], m_color, 1);
			}
		}
		return 1;
	}
//Static function
public:

};
}