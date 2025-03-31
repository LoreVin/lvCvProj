#pragma once
#include <opencv2/opencv.hpp>
#include "ImageProcessor.hpp"
#include "Tools.hpp"
namespace lvcv
{

class CVideoProcessor
{
private:
	cv::Ptr<cv::VideoCapture> m_objCap;
	std::string m_strVideoPath;
	std::string m_strVideoSavePath;
	int m_iFourccR;
	double m_dFpsR;
	cv::Size m_szFrameR;
	
protected:
	bool m_bProcessImage;
	cv::Ptr<lvcv::CImageProcessor> m_objImgProcessor;
	bool m_bSaveVideo;
	cv::Ptr<cv::VideoWriter> m_objWriter;
	std::string m_strWinNameOrg;
	std::string m_strWinNameDst;
	cv::Mat m_imRead;
	cv::Mat m_imProcessed;
	double m_dDelay;
	int m_iFourccW;
	double m_dFpsW;
	cv::Size m_szFrameW;

//Constructor and destructor
public:
	CVideoProcessor(const bool& bSaveVideo =false)
	{
		m_bSaveVideo = bSaveVideo;
		m_bProcessImage = false;
		m_objCap = new cv::VideoCapture;
		if (m_bSaveVideo)
			m_objWriter = new cv::VideoWriter;
		m_strWinNameOrg = "original image";
		m_strWinNameDst = "destinational image";
		cv::namedWindow(m_strWinNameOrg, cv::WINDOW_FREERATIO);
		cv::namedWindow(m_strWinNameDst, cv::WINDOW_FREERATIO);
		m_dDelay = 100.0;
		m_imRead = cv::Mat();
		m_imProcessed = cv::Mat();
	}

	virtual ~CVideoProcessor()
	{

	}

private:
	virtual int readImage()
	{
		if (!m_objCap->isOpened())
			return -1;
		if (!m_objCap->read(m_imRead))
			return -2;
		return 1;
	}
	
	virtual int getVideoProperty()
	{
		m_iFourccR = (int)m_objCap->get(cv::CAP_PROP_FOURCC);
		m_dFpsR = m_objCap->get(cv::CAP_PROP_FPS);
		m_szFrameR.width = (int)m_objCap->get(cv::CAP_PROP_FRAME_WIDTH);
		m_szFrameR.height = (int)m_objCap->get(cv::CAP_PROP_FRAME_HEIGHT);

		return 1;
	}

	virtual int setVideoWriterProperty(int iFourcc, double dFps, cv::Size szFrame)
	{
		if (iFourcc < 0 || dFps <= 0 || szFrame.width <= 0 || szFrame.height <= 0)
			return -1;
		m_iFourccW = iFourcc;
		m_dFpsW = dFps;
		m_szFrameW = szFrame;
		return 1;
	}


public:
	virtual int setImageProcessor(cv::Ptr<lvcv::CImageProcessor>obj)
	{
		if (nullptr == obj)
		{
			m_bProcessImage = false;
			return -1;
		}
		m_objImgProcessor = obj;
		m_bProcessImage = true;
		return 1;
	}

	virtual int openReader(const std::string &strVideoPath)
	{
		if (strVideoPath.empty() || strVideoPath == "")
			return -1;
		if (!CTools::existedFileOrDirectory(strVideoPath))
			return -2;
		m_strVideoPath.assign(strVideoPath);
		if (!m_objCap->open(m_strVideoPath))
			return -3;
		getVideoProperty();
		return 1;
	}

	virtual int openWriter(const std::string& strVideoSavePath)
	{
		if (strVideoSavePath.empty() || strVideoSavePath == "")
			return -1;
		if (CTools::existedFileOrDirectory(strVideoSavePath))
			return -2;
		if (m_objWriter.empty())
			m_objWriter = new cv::VideoWriter;
		int iFourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		setVideoWriterProperty(iFourcc, m_dFpsR, m_szFrameR);
		m_strVideoSavePath.assign(strVideoSavePath);
		if(!m_objWriter->open(m_strVideoSavePath,m_iFourccW,m_dFpsW,m_szFrameW))
			return -2;
		m_bSaveVideo = true;
		return 1;
	}
	
	virtual int showVideo(const bool& bShowOrgVideo = true, const bool& bShowDstVideo = false)
	{
		while (readImage() > 0)
		{
			cv::imshow(m_strWinNameOrg, m_imRead);
			if (bShowDstVideo && m_bProcessImage)
			{
				m_objImgProcessor->processImage(m_imRead, m_imProcessed);
				cv::imshow(m_strWinNameDst, m_imProcessed);
			}
			if (m_bSaveVideo)
			{
				if (m_imProcessed.channels() == 1)
					lvcv::CTools::convertGray2BGR(m_imProcessed, m_imProcessed);
				m_objWriter->write(m_imProcessed);
			}
			if (cv::waitKey(m_dDelay) >= 0)
				break;
		}
		return 1;
	}

};

class CImgSeqProcessor : virtual public CVideoProcessor
{
private:
	std::string m_strDirPath;
	std::string m_strDirSavePath;
	std::vector<std::string> m_vStrImgSeq;
	bool m_bReadDir;
	bool m_bWriteDir;
	int m_iIndexImg;

public:
	CImgSeqProcessor()
	{
		m_bReadDir = false;
		m_bWriteDir = false;
		m_iIndexImg = 0;
		m_dDelay = 100;
		//cv::namedWindow(m_strWinNameOrg, cv::WINDOW_FREERATIO);
		//cv::namedWindow(m_strWinNameDst, cv::WINDOW_FREERATIO);
	}
	virtual ~CImgSeqProcessor()
	{ }
public:
	virtual int setDelay(double dDelay=100)
	{
		if (dDelay <= 0)
			return -1;
		m_dDelay = dDelay;
		return 1;
	}

	virtual int openReader(const std::string &strDirPath)
	{
		if (m_bReadDir)
			return 1;
		m_strDirPath.assign(strDirPath);
		if (!lvcv::CTools::existedFileOrDirectory(m_strDirPath))
			return -1;
		if (lvcv::CTools::getFilenamesFromDir(m_strDirPath.c_str(), m_vStrImgSeq) < 0)
			return -2;
		if (m_vStrImgSeq.empty())
			return -3;
		m_bReadDir = true;
		return 1;
	}

	virtual int openWriter(const std::string& strDirSavePath)
	{
		if (m_bWriteDir)
			return 1;
		m_strDirSavePath.assign(strDirSavePath);
		if (m_strDirSavePath.empty())
			return -1;
		m_bWriteDir = true;
		return 1;
	}

	virtual int readImg()
	{
		if (!m_imRead.empty())
			m_imRead.release();
		m_imRead = cv::imread(m_vStrImgSeq[m_iIndexImg++]);
		if (m_imRead.empty())
			return -1;
		return 1;
	}

	virtual int showVideo(const bool&bShowOrg=true,const bool&bShowDst=false)
	{

		char strSavePathTmp[128];
		while (readImg() > 0)
		{
			if (bShowOrg)
				cv::imshow(m_strWinNameOrg, m_imRead);
			if (bShowDst)
			{
				if (m_bProcessImage)
					m_objImgProcessor->processImage(m_imRead, m_imProcessed);
				cv::imshow(m_strWinNameDst, m_imProcessed);
			}
			if (m_bWriteDir)
			{
				sprintf_s(strSavePathTmp, "%s/%d.jpg", m_strDirSavePath.c_str(), m_iIndexImg-1);
				cv::imwrite(strSavePathTmp, m_imProcessed);
			}
			if (cv::waitKey(m_dDelay) >= 0)
				break;
		}
		return 1;
	}
};

}