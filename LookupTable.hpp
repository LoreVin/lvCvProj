#pragma once
#include <opencv2/opencv.hpp>
namespace lvcv
{
class CLookupTable
{

//Member variable
private:
	cv::Mat m_lookupTable;
	cv::Mat m_lookupTableInverse;
	uchar* m_ptrLookup{ nullptr };
	float* m_ptrHist{ nullptr };
	std::shared_ptr<CHistogram> m_objHist;

//Constructor and Deconstructor
public:
	CLookupTable() 
	{
		m_lookupTable.create(256, 1, CV_8UC1);
		m_lookupTable.setTo(0);
		m_lookupTableInverse.create(256, 1, CV_8UC1);
		m_lookupTableInverse.setTo(0);
		m_ptrLookup = nullptr;
		for (int i = 0; i < m_lookupTableInverse.rows; ++i)
		{
			m_ptrLookup = m_lookupTableInverse.ptr<uchar>(i);
			*m_ptrLookup = 255 - i;
		}
		m_ptrLookup = nullptr;
		m_objHist = std::make_shared<CHistogram>();
		m_ptrHist = nullptr;
	}
	virtual ~CLookupTable() {}

//Member function
protected:
	virtual int getContrastLookupTable(cv::Mat& imHistOne, int iMinVal = 30)
	{
		//Judge the input parameters
		if (imHistOne.empty() || imHistOne.cols > 1)
			return -1;
		//Recalculate the lookup table for contrasting the image
		int iMinIndex = 0;
		int iMaxIndex = 255;
		for (int i = 0; i < imHistOne.rows; ++i)
		{
			m_ptrHist = imHistOne.ptr<float>(i);
			if (*m_ptrHist < iMinVal)
				iMinIndex++;
		}
		for (int i = imHistOne.rows - 1; i > 0; --i)
		{
			m_ptrHist = imHistOne.ptr<float>(i);
			if (*m_ptrHist < iMinVal)
				iMaxIndex--;
		}
		int iDetaMaxMin = iMaxIndex - iMinIndex;
		for (int i = 0; i < m_lookupTable.rows; ++i)
		{
			m_ptrLookup = m_lookupTable.ptr<uchar>(i);
			if (i < iMinIndex)
				*m_ptrLookup = 0;
			else if (i > iMaxIndex)
				*m_ptrLookup = 255;
			else
				*m_ptrLookup = cv::saturate_cast<uchar>(cvRound(255.0 * (i - iMinIndex) / iDetaMaxMin));
		}
		return 1;
	}
public:
	virtual int inverseColor(const cv::Mat& imSrc, cv::Mat& imDst)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		//Call LUT function to inverse the image
		cv::LUT(imSrc, m_lookupTableInverse, imDst);
		if (imDst.empty())
			return -1;
		return 1;
	}

	virtual int increaseContrast(const cv::Mat& imSrc, cv::Mat& imDst, int iMinVal = 20)
	{
		//Judge whether the input 
		if (imSrc.empty())
			return -1;
		//Reset lookupTable for contrast current input image
		if (imSrc.channels() == 1)
		{
			double dMin = 0.0, dMax = 0.0;
			cv::minMaxLoc(imSrc, &dMin, &dMax, 0, 0);
			cv::Mat imHist;
			int iMinIndex = 0, iMaxIndex = 255;
			m_objHist->computeHistogram(imSrc, imHist);
			getContrastLookupTable(imHist, iMinVal);
			cv::LUT(imSrc, m_lookupTable, imDst);

		}
		else if (imSrc.channels() == 3)
		{
			//Convert color space from BGR to HSV
			cv::Mat imHsv;
			lvcv::CTools::convertBGR2HSV(imSrc, imHsv);
			//Split the HSV image to vIms
			std::vector<cv::Mat> vIms;
			cv::split(imHsv, vIms);			
			//Contrast the value image
			cv::Mat imHistValue;
			m_objHist->computeHistogram(vIms[2], imHistValue);
			getContrastLookupTable(imHistValue, 30);
			cv::LUT(vIms[2], m_lookupTable, vIms[2]);
			//Merge the vIms to hsv image
			cv::merge(vIms, imDst);
			//Convert color space from HSV to BGR
			lvcv::CTools::convertHSV2BGR(imDst, imDst);
		}
		return 1;
	}

	virtual int reduceColor(const cv::Mat& imSrc, cv::Mat& imDst, int iDiv = 32)
	{
		//Judge the input parameters
		if (imSrc.empty())
			return -1;
		if (iDiv < 4)
			iDiv = 4;
		else if (iDiv > 128)
			iDiv = 128;
		//Set lookup table for reducing the color of the image
		int iDivHalf = iDiv >> 1;
		for (int i = 0; i < m_lookupTable.rows; ++i)
		{
			m_ptrLookup = m_lookupTable.ptr<uchar>(i);
			*m_ptrLookup = i / iDiv * iDiv + iDivHalf;
		}
		//Call LUT function for the image 
		cv::LUT(imSrc, m_lookupTable, imDst);
		return 1;
	}

//Static function
public:

};//end of class

}//end of namespace lvcv
