#pragma once
#include <opencv2/opencv.hpp>
#include "Tools.hpp"
#include "DetectorFeature.hpp"
#include "MatcherFeature.hpp"
namespace lvcv
{

class CReconstructor3D
{ 
private:
	cv::Mat m_deta_R;
	cv::Mat m_deta_t;
	cv::Mat m_map01;
	cv::Mat m_map02;
	std::vector<cv::Point2f> m_vPointf01;
	std::vector<cv::Point2f> m_vPointf02;
	std::vector<uchar> m_vInlier;
	cv::Mat m_Essential;
	//Camera intrinsic parameters
	bool m_bSetCamParams;
	cv::Mat m_K;
	cv::Mat m_distCoffs;
	//Camera extrinsic parameters
	cv::Mat m_R;
	cv::Mat m_t;

public:
	CReconstructor3D()
	{
		m_bSetCamParams = false;
	}

	virtual ~CReconstructor3D()
	{
	}

public:
	int setCamParameters(cv::Mat& K, cv::Mat& distCoffs)
	{
		if (K.empty() || distCoffs.empty())
			return -1;
		m_bSetCamParams = false;
		K.copyTo(m_K);
		distCoffs.copyTo(m_distCoffs);
		m_bSetCamParams = true;
		return 1;
	}
	
	virtual int removeDistortion(cv::Mat& imSrcDst)
	{
		if (imSrcDst.empty())
			return -1;
		if (!m_bSetCamParams)
			return -2;
		cv::initUndistortRectifyMap(
			m_K,
			m_distCoffs,
			cv::Mat(),
			cv::Mat(),
			imSrcDst.size(),
			CV_32FC1,
			m_map01,
			m_map02
		);
		cv::remap(imSrcDst, imSrcDst, m_map01, m_map02, cv::INTER_LINEAR);
		return 1;
	}
	
	virtual int computeEssentialMat(
		const std::vector<cv::KeyPoint>&vKps01,
		const std::vector<cv::KeyPoint>&vKps02,
		const std::vector<cv::DMatch>& vMatches)
	{
		m_vPointf01.clear();
		m_vPointf02.clear();
		for (const cv::DMatch& match : vMatches)
		{
			m_vPointf01.push_back(vKps01[match.queryIdx].pt);
			m_vPointf02.push_back(vKps02[match.trainIdx].pt);
		}
		m_vInlier.clear();
		m_Essential = cv::findEssentialMat(m_vPointf01,m_vPointf02, m_K, cv::FM_RANSAC, 
			0.95, 1.0, 1000, m_vInlier);
		cv::recoverPose(m_vPointf01, m_vPointf02, m_K, m_R, m_t);
		return 1;
	}

	virtual int triangulate()
	{
		//cv::triangulatePoints
		return 1;
	}

	virtual int tracking(std::vector<cv::DMatch>& vMatches)
	{
		if (vMatches.empty())
			return -1;
		if (!m_bSetCamParams)
			return -2;
		//Get the Essential matrix

	}

};

class CCamCalibrator : virtual public CReconstructor3D
{
private:
	std::string m_strChessRootPath;
	std::vector<std::string> m_vStrChessFilename;
	bool m_bFindChessboard2D;
	std::vector<cv::Point2f> m_vChessPnt2D;
	bool m_bFindChessboard3D;
	std::vector<cv::Point3f> m_vChessPnt3D;
	std::vector<std::vector<cv::Point2f>> m_vvChessPnt2D;
	std::vector<std::vector<cv::Point3f>> m_vvChessPnt3D;
	cv::Size m_szImage;
	cv::Mat m_K;
	cv::Mat m_distCoeff;
	cv::Mat m_rvec;
	cv::Mat m_tvec;
public:
	CCamCalibrator(const char*strChessRootPath=nullptr):
		m_bFindChessboard2D(false),
		m_bFindChessboard3D(false),
		m_szImage(cv::Size(0,0)),
		m_K(cv::Mat()),
		m_rvec(cv::Mat()),
		m_tvec(cv::Mat()),
		m_distCoeff(cv::Mat())
	{
		if (nullptr != strChessRootPath)
		{
			m_strChessRootPath.assign(strChessRootPath);
			lvcv::CTools::getFilenamesFromDir(m_strChessRootPath.c_str(), m_vStrChessFilename);
		}
	}
	virtual ~CCamCalibrator()
	{

	}
private:
	int findChessboardPoints2D(const cv::Mat&imSrc,const cv::Size&szChessboard)
	{
		if (imSrc.empty())
			return -1;
		m_vChessPnt2D.clear();
		m_bFindChessboard2D = 
			cv::findChessboardCorners(imSrc, szChessboard, m_vChessPnt2D);
		return m_bFindChessboard2D ? 1 : -1;
	}
	int findChessboardPoints3D(const cv::Size& szChessboard)
	{
		cv::Point3f pnt3D;
		m_vChessPnt3D.clear();
		for(int i=0; i<szChessboard.height;++i)
			for (int j = 0; j < szChessboard.width; ++j)
			{
				pnt3D.x = j; pnt3D.y = i; pnt3D.z = 0;
				m_vChessPnt3D.push_back(pnt3D);
			}
		return m_bFindChessboard3D ? 1 : -1;
	}

public:
	int getChessboardPoints2D(const cv::Size& szChessboard)
	{
		cv::Mat imRead;
		for (std::string& strTmp: m_vStrChessFilename)
		{
			imRead = cv::imread(strTmp);
			findChessboardPoints2D(imRead, szChessboard);
			lvcv::CTools::drawKps(imRead, m_vChessPnt2D);
			if (m_bFindChessboard2D)
				m_vvChessPnt2D.push_back(m_vChessPnt2D);

		}
		m_szImage = imRead.size();
		return 1;
	}
	int getChessboardPoints3D(const cv::Size& szChessboard)
	{
		if (m_vvChessPnt2D.empty())
			return -1;
		for (int i = 0; i < m_vvChessPnt2D.size(); ++i)
		{
			if (m_bFindChessboard3D)
				m_vvChessPnt3D.push_back(m_vChessPnt3D);
			else
			{
				findChessboardPoints3D(szChessboard);
				m_vvChessPnt3D.push_back(m_vChessPnt3D);
			}
		}
		return 1;
	}
	int calibrateCamera(double &dError){
		int flag = cv::CALIB_ZERO_TANGENT_DIST;
		dError = cv::calibrateCamera(m_vvChessPnt3D, m_vvChessPnt2D, m_szImage, 
			m_K, m_distCoeff, m_rvec, m_tvec,flag);
		return -1;
	}
};



}