#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
namespace lvcv
{
class CDetectorFeature
{
//Member variables
public:
	cv::Ptr<cv::FeatureDetector> m_objDetector;

//Constructor and destructor
public:
	CDetectorFeature() {}
	virtual ~CDetectorFeature() {}

//Member function
protected:
public:
	virtual int processImage(cv::Mat& imSrc, cv::Mat& imDst, const double& dVal = 0) {
	
		return 1;
	}

	virtual int detectAndCompute(cv::Mat& imGray, std::vector<cv::KeyPoint>& vKps, cv::Mat& des)
	{
		//Judge the inputparameters
		if (imGray.empty() || imGray.channels() > 1)
			return -1;
		if (!vKps.empty())
			vKps.clear();
		//Detect keypoints and compute the descriptors
		if (m_objDetector==nullptr)
			return -1;
		m_objDetector->detectAndCompute(imGray, cv::Mat(),vKps,des);
		if (vKps.empty())
			return -1;
		if (des.empty())
			return -2;
		return 1;
	}

//Static function
public:
	//Detect keypoints by FAST algorithm
	static int detectKpsByFAST(const cv::Mat& imGray,
		std::vector<cv::KeyPoint> &vKps,
		int iThresh = 30)
	{
		//Judge the input parameters
		if (imGray.empty() || imGray.channels() > 1)
			return -1;
		if (!vKps.empty())
			return -2;
		//Call FAST algorithm to detect keypoints in the image 
		cv::FAST(imGray, vKps, iThresh, true);
		if (vKps.empty())
			return -1;
		return 1;
	}

	static int detectKpsByGoodharris(const cv::Mat& imGray,
		std::vector<cv::Point2f>& vPnts,
		int iMaxCorners=500)
	{
		//Judge the input parameters
		if (imGray.empty())
			return -1;
		if (!vPnts.empty())
			vPnts.clear();
		//Detect features from the image
		double dQualityLevel = 0.05;
		double dMinDis = 10;
		cv::goodFeaturesToTrack(imGray, vPnts, iMaxCorners, dQualityLevel, dMinDis);
		if (vPnts.empty())
			return -1;
		return 1;
	}

};
class CDetector_ORB : virtual public CDetectorFeature
{

public:
	CDetector_ORB(int iFeatures=500,float fScaleFactor=1.2f,int iLevels=5) 
	{
		m_objDetector = cv::ORB::create(iFeatures, fScaleFactor, iLevels);
	}
	virtual ~CDetector_ORB() {
		m_objDetector = nullptr;
	}
};
class CDetector_BRISK : virtual public CDetectorFeature
{
public:
	CDetector_BRISK(int iThresh=30,int iOctaves=3,float fPatternScale=1.0f)
	{
		m_objDetector = cv::BRISK::create(iThresh, iOctaves, fPatternScale);
	}
	virtual ~CDetector_BRISK() {
		m_objDetector = nullptr;
	}
};
class CDetector_SIFT : virtual public CDetectorFeature
{
public:
	CDetector_SIFT(int iFeatures=0, int iOctaveLayers=3)
	{
		m_objDetector = cv::SIFT::create(iFeatures, iOctaveLayers);
	}
	virtual ~CDetector_SIFT() {
		m_objDetector = nullptr;
	}
};


}