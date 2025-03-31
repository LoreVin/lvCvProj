#pragma once
#include <opencv2/opencv.hpp>
namespace lvcv
{
class CMatcherFeature
{
protected:
	cv::Ptr<cv::DescriptorMatcher> m_objMatcher;
	bool m_bKnn{ false };
	int m_iK;
private:
	std::vector<std::vector<cv::DMatch>> m_vvMatches;
	double m_dRatio;
public:
	CMatcherFeature()
	{
		m_bKnn = false;
		m_iK = 2;
		m_dRatio = 0.85;
	}
	virtual ~CMatcherFeature() {}

public:
	virtual int Match(
		cv::Mat& des01,
		cv::Mat& des02,
		std::vector<cv::DMatch>& vMatches)
	{
		//Judge the input parameters

		//Match the descriptors
		if (m_bKnn)
		{
			if (MatchKnn(des01, des02, vMatches) < 0)
				return -1;
		}
		else
		{
			if (MatchGeneral(des01, des02, vMatches) < 0)
				return -1;
		}
		return 1;
	}

private:
	virtual int MatchGeneral(
		cv::Mat& des01,
		cv::Mat& des02,
		std::vector<cv::DMatch>& vMatches)
	{
		if (m_objMatcher == nullptr)
			return -1;
		try
		{
			m_objMatcher->match(des01, des02, vMatches);
		}
		catch (...)
		{
			return -2;
		}
		return 1;
	}

	virtual int MatchKnn(
		cv::Mat& des01,
		cv::Mat& des02,
		std::vector<cv::DMatch>& vMatches)
	{
		if (nullptr == m_objMatcher)
			return -1;
		const char* strError;
		try 
		{

			m_objMatcher->knnMatch(des01, des02, m_vvMatches, m_iK);
			//Filter the m_vvMatches to assign values for vMatches
			for (int i = 0; i < m_vvMatches.size(); ++i)
				if (m_vvMatches[i][0].distance / m_vvMatches[i][1].distance < m_dRatio)
					vMatches.push_back(m_vvMatches[i][0]);
			if (vMatches.empty())
				return -2;
		}
		catch (std::exception&e)
		{
			strError = e.what();
		}
		catch (cv::Exception& e)
		{
			strError = e.what();
		}
		return 1;
	}
};
class CMatcher_BF_HAM :public CMatcherFeature
{
public:
	CMatcher_BF_HAM()
	{
		m_objMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
	}
};
class CMatcher_BF_L1 :public CMatcherFeature
{
public:
	CMatcher_BF_L1()
	{
		m_objMatcher = cv::BFMatcher::create(cv::NORM_L1, false);
	}
};
class CMatcher_BF_L2 :public CMatcherFeature
{
public:
	CMatcher_BF_L2()
	{
		m_objMatcher = cv::BFMatcher::create(cv::NORM_L2, false);
	}
};
class CMatcher_BF_KNN_HAM :public CMatcherFeature
{
public:
	CMatcher_BF_KNN_HAM()
	{
		m_bKnn = true;
		m_objMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
	}
};
class CMatcher_BF_KNN_L1 :public CMatcherFeature
{
public:
	CMatcher_BF_KNN_L1()
	{
		m_bKnn = true;
		m_objMatcher = cv::BFMatcher::create(cv::NORM_L1, false);
	}
};
class CMatcher_BF_KNN_L2 :public CMatcherFeature
{
public:
	CMatcher_BF_KNN_L2()
	{
		m_bKnn = true;
		m_objMatcher = cv::BFMatcher::create(cv::NORM_L2, false);
	}
};
class CMatcher_FLANN_HAM : public CMatcherFeature
{
public:
	CMatcher_FLANN_HAM()
	{ 
		m_objMatcher = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(6, 12, 1));
	}
};
class CMatcher_FLANN_NO_HAM : public CMatcherFeature
{
public: 
	CMatcher_FLANN_NO_HAM()
	{
		m_objMatcher = new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(5));
	}
};
class CMatcher_FLANN_KNN_HAM : public CMatcherFeature
{
public:
	CMatcher_FLANN_KNN_HAM()
	{
		m_bKnn = true;
		m_objMatcher = new cv::FlannBasedMatcher(new cv::flann::LshIndexParams(6, 12, 1));
	}
};
class CMatcher_FLANN_KNN_NO_HAM :public CMatcherFeature
{
public:
	CMatcher_FLANN_KNN_NO_HAM()
	{
		m_bKnn = true;
		m_objMatcher = new cv::FlannBasedMatcher(new cv::flann::KDTreeIndexParams(5));
	}
};
class CMatcher_Ransac_F : public CMatcherFeature
{
private:
	cv::Ptr<CMatcherFeature> m_obj;
	std::vector<cv::DMatch> m_vMatchesTmp;
	std::vector<cv::Point2f> m_vPointf01, m_vPointf02;
	std::vector<uchar> m_vInlier;
	double m_dDisToEpline;
	double m_dConf;
	cv::Mat m_F;
	bool m_bComputedF;
public:
	CMatcher_Ransac_F(cv::Ptr<CMatcherFeature> obj)
	{
		m_obj = obj;
		m_dDisToEpline = 2.0;
		m_dConf = 0.97;
		m_F = cv::Mat();
		m_bComputedF = false;
	}

public:
	virtual int Match(std::vector<cv::KeyPoint>& vKps01,
		std::vector<cv::KeyPoint>& vKps02,
		cv::Mat& des01,
		cv::Mat& des02,
		std::vector<cv::DMatch>& vMatches,
		const bool& bRefine = false)
	{
		//Matche the descriptors to get vMatchesTmp(roughly)
		if (!m_vMatchesTmp.empty())
			return -1;
		if (m_obj->Match(des01, des02, m_vMatchesTmp) < 0)
			return -2;
		//Compute the Fundamental Matrix by RANSAC algorithm
		m_vInlier.clear();
		m_vPointf01.clear();
		m_vPointf02.clear();
		for (cv::DMatch& match : m_vMatchesTmp)
		{
			m_vPointf01.push_back(vKps01[match.queryIdx].pt);
			m_vPointf02.push_back(vKps02[match.trainIdx].pt);
		}
		cv::findFundamentalMat(m_vPointf01, m_vPointf02, m_vInlier, 
			cv::FM_RANSAC, m_dDisToEpline, m_dConf);
		//Filter the vMatchesTmp to vMatches
		for (int i = 0; i < m_vInlier.size(); ++i)
			if (m_vInlier[i])
				vMatches.push_back(m_vMatchesTmp[i]);
		//Compute the Fundamental Matrix by POINT_8 algorithm 
		m_vPointf01.clear();
		m_vPointf02.clear();
		for (cv::DMatch& match : vMatches)
		{
			m_vPointf01.push_back(vKps01[match.queryIdx].pt);
			m_vPointf02.push_back(vKps02[match.trainIdx].pt);
		}
		m_F = cv::findFundamentalMat(m_vPointf01, m_vPointf02, cv::FM_8POINT);
		//Correct the keypoints
		if (bRefine)
		{
			std::vector<cv::Point2f> vNew01, vNew02;
			cv::correctMatches(m_F, m_vPointf01, m_vPointf02, vNew01, vNew02);
			for (int i = 0; i < vNew01.size(); ++i)
			{
				vKps01[vMatches[i].queryIdx].pt = vNew01[i];
				vKps02[vMatches[i].trainIdx].pt = vNew02[i];
			}
		}
		m_bComputedF = true;
		return 1;
	}

	int getFundamentalMat(cv::Mat& F)
	{
		if (!m_bComputedF)
			return -1;
		else
			m_F.copyTo(F);

		return 1;
	}
};
class CMatcher_Ransac_H : public CMatcherFeature
{
private:
	cv::Ptr<CMatcherFeature> m_obj;
	std::vector<cv::Point2f> m_vPointf01, m_vPointf02;
	std::vector<uchar> m_vInlier;
	double m_dDisToEpline;
	cv::Mat m_H;
	bool m_bComputedH;
public:
	CMatcher_Ransac_H(cv::Ptr<CMatcherFeature> obj)
	{
		m_obj = obj;
		m_dDisToEpline = 2.0;
		m_H = cv::Mat();
		m_bComputedH = false;

	}

	virtual int Match(std::vector<cv::KeyPoint>& vKps01, 
		std::vector<cv::KeyPoint>& vKps02,
		cv::Mat& des01,
		cv::Mat& des02,
		std::vector<cv::DMatch>&vMatches, 
		const bool& bRefine = false)
	{
		//Judge the input parameters

		//Match the descriptors(roughly)
		std::vector<cv::DMatch> vMatchesTmp;
		m_obj->Match(des01, des02, vMatchesTmp);
		//Compute the Homography Matrix by RANSAC algorithm
		m_vPointf01.clear();
		m_vPointf02.clear();
		m_vInlier.clear();
		for (cv::DMatch& match : vMatchesTmp)
		{
			m_vPointf01.push_back(vKps01[match.queryIdx].pt);
			m_vPointf02.push_back(vKps02[match.trainIdx].pt);
		}
		cv::findHomography(m_vPointf01, m_vPointf02, m_vInlier, cv::FM_RANSAC, m_dDisToEpline);
		//Filter the matching result
		for (int i = 0; i < m_vInlier.size(); ++i)
			if (m_vInlier[i])
				vMatches.push_back(vMatchesTmp[i]);
		if (vMatches.empty())
			return -2;
		//Compute the Homography Matrix by RHO algorithm
		m_vPointf01.clear();
		m_vPointf02.clear();
		for (cv::DMatch& match : vMatches)
		{
			m_vPointf01.push_back(vKps01[match.queryIdx].pt);
			m_vPointf02.push_back(vKps02[match.trainIdx].pt);
		}
		m_H = cv::findHomography(m_vPointf01, m_vPointf02, cv::RHO);
		m_bComputedH = true;
		return 1;
	}

	int getHomographyMat(cv::Mat& H)
	{
		if (!m_bComputedH)
			return -1;
		else
			m_H.copyTo(H);
		return 1;
	}
};
}
