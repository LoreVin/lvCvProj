#pragma once
#include <opencv2/opencv.hpp>
namespace lvcv
{
class CImageProcessor
{
//Constructor and destructor
public:
	CImageProcessor() = default;
	virtual ~CImageProcessor() = default;
protected:
	virtual bool judgeParams(const cv::Mat& imSrc)
	{
		if (imSrc.empty())
			return false;
		return true;
	}

public:
	virtual int processImage(const cv::Mat& imSrc, cv::Mat& imDst) = 0;
};

class CImageCanny : virtual public CImageProcessor
{
private:
	int m_iThreshLow;
	int m_iThreshHigh;

public:
	CImageCanny() :m_iThreshLow(80), m_iThreshHigh(240) {};

	CImageCanny(const int &iThreshLow = 80,const int &iThreshHigh = 240) :
		m_iThreshLow(iThreshLow), 
		m_iThreshHigh(iThreshHigh)
	{
		if (setThresh(iThreshLow, iThreshHigh) < 0)
			std::cerr << "Failed parameters threshLow or threshHigh" << std::endl;

	}
	virtual ~CImageCanny() = default;
	
public:
	virtual int setThresh(const int& iThreshLow = 80, const int& iThreshHigh = 240)
	{
		if (iThreshLow >= iThreshHigh)
			return -1;
		if (iThreshLow < 0 || iThreshHigh <= 0)
			return -2;
		m_iThreshLow = iThreshLow;
		m_iThreshHigh = iThreshHigh;
		return 1;
	}

	virtual int processImage(const cv::Mat& imSrc, cv::Mat& imDst)override
	{
		if (!judgeParams(imSrc))
			return -1;
		cv::Canny(imSrc, imDst, m_iThreshLow, m_iThreshHigh);
		if (imDst.empty())
			return -2;
		return 1;
	}


};

class CObjectTracker :virtual public CImageProcessor
{
private:
	cv::Rect m_box;
	cv::Scalar m_color;
	cv::Ptr<cv::Tracker> m_objTracker;
public:
	CObjectTracker() = default;
	virtual ~CObjectTracker() = default;
	CObjectTracker(const cv::Rect& box = cv::Rect())
	{
		m_box = box;
		m_color = cv::Scalar(0, 255, 0);
	}
	virtual ~CObjectTracker(){}
public:
	virtual int setTracker(cv::Ptr<cv::Tracker> objTracker)
	{
		if (nullptr == objTracker)
			return -1;
		m_objTracker = objTracker;
		return 1;
	}
	virtual int processImage(const cv::Mat& imSrc, cv::Mat& imDst)override
	{
		if (!judgeParams(imSrc))
			return -1;
		m_objTracker->init(imSrc, m_box);
		if (!m_objTracker->update(imSrc, m_box))
			return -2;
		imSrc.copyTo(imDst);
		cv::rectangle(imDst, m_box, m_color, 1);
		return 1;
	}

};
}