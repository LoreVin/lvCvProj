#include "Histogram.hpp"
#include "LookupTable.hpp"
#include "Morph.hpp"
#include "Hough.hpp"
#include "DetectorFeature.hpp"
#include "MatcherFeature.hpp"
#include "Reconstructor3D.hpp"
#include "VideoProcessor.hpp"
void Test()
{
	std::string g_strImgsRtPath = "D:/MjData/images";
	
	bool bTestHistogram = false;
	if (bTestHistogram)
	{
		char strImgRead[128];
		sprintf_s(strImgRead, "%s/group.jpg", g_strImgsRtPath.c_str());
		cv::Mat imRead = cv::imread(strImgRead);
		cv::Mat imHist;
		std::shared_ptr<lvcv::CHistogram> objHist = std::make_shared<lvcv::CHistogram>();
		//Test function ComputeHistogram 
		if (0)
		{
			objHist->computeHistogram(imRead, imHist);
		}
		//Test function ComputeHueHistogram
		if (0)
		{
			objHist->computeHueHistogram(imRead, imHist);
		}
		//Test function ComputeHueBackproject and ComputeBackproject
		if (0)
		{
			cv::Mat imBackproject;
			cv::Mat imHistTarget;
			cv::Point ptLftTop(70, 63);
			cv::Point ptRgtBtm(140, 112);
			cv::Rect rect(ptLftTop, ptRgtBtm);
			cv::Mat imGray;
			cv::cvtColor(imRead, imGray, cv::COLOR_BGR2GRAY);
			objHist->computeHueHistogram(imRead, imHistTarget);
			objHist->computeHueBackproject(imRead, imHistTarget, imBackproject);
		}
		//Test function EqualizeImage
		if (0)
		{
			cv::Mat imEqualized;
			objHist->equalizeImage(imRead, imEqualized);
		}
		//Test function CompareHist
		if (0)
		{
			std::string strIm01=g_strImgsRtPath+"/parliament2.jpg";
			std::string strIm02=g_strImgsRtPath+"/parliament3.jpg";
			cv::Mat imRead01 = cv::imread(strIm01);
			cv::Mat imRead02 = cv::imread(strIm02);
			double dRes = 0.0;
			objHist->compareHist(imRead01, imRead02, dRes);
		}
		//Test function ConvertHist2Image
		if (0)
		{
			cv::Mat imGray;
			lvcv::CTools::convertToGray(imRead, imGray);
			objHist->computeHistogram(imGray, imHist);
			cv::Mat imDst;
			objHist->convertHist2Image(imHist, imDst);
		}

	}
	
	bool bTestLookupTable = false;
	if (bTestLookupTable)
	{
		char strImgRead[128];
		sprintf_s(strImgRead, "%s/group.jpg", g_strImgsRtPath.c_str());
		cv::Mat imRead = cv::imread(strImgRead);
		std::shared_ptr<lvcv::CLookupTable> objLookup =
			std::make_shared<lvcv::CLookupTable>();
		//Test function InverseColor
		if (0)
		{
			cv::Mat imInverse;
			objLookup->inverseColor(imRead, imInverse);
		}
		//Test function IncreaseContrast
		if (0)
		{
			cv::Mat imGray;
			lvcv::CTools::convertToGray(imRead, imGray);
			cv::Mat imContrast;
			objLookup->increaseContrast(imRead, imContrast, 30);
		}
		//Test function ReduceColor
		if (0)
		{
			int iDiv = 64;
			cv::Mat imReduced;
			objLookup->reduceColor(imRead, imReduced, iDiv);
		}
	}
	
	bool bTestMorph = false;
	if (bTestMorph)
	{
		char strImgRead[128];
		sprintf_s(strImgRead, "%s/group.jpg", g_strImgsRtPath.c_str());
		cv::Mat imRead = cv::imread(strImgRead);
		std::shared_ptr<lvcv::CMorph> objMorph =
			std::make_shared<lvcv::CMorph>();
		//Test function erodeImage
		if (0)
		{
			cv::Mat imGray;
			lvcv::CTools::convertToGray(imRead, imGray);
			cv::Mat imSrc = imRead.clone();
			cv::Mat imDst;
			objMorph->erodeImage(imSrc, imSrc,2);
		}
		//Test function dilateImage
		if (0)
		{
			cv::Mat imSrc = imRead.clone();
			cv::Mat imDst;
			objMorph->dilateImage(imSrc,imSrc, 1);
		}
		//Test function openImage
		if (0)
		{
			cv::Mat imSrc = imRead.clone();
			cv::Mat imDst;
			objMorph->openImage(imSrc,imDst,1);
		}
		//Test function closeImage
		if (0)
		{
			cv::Mat imSrc = imRead.clone();
			cv::Mat imDst;
			objMorph->closeImage(imSrc, imDst, 1);
		}
		//Test function detectEdge
		if (0)
		{
			cv::Mat imSrc = imRead.clone();
			cv::Mat imEdge;
			objMorph->detectEdge(imSrc, imEdge, 1);
		}
		//Test function detectEdgeMorphEx
		if (1)
		{
			cv::Mat imSrc = imRead.clone();
			cv::Mat imEdge;
			objMorph->detectEdgeMorphEx(imSrc, imEdge, 1);
		}
	}

	bool bTestHough = false;
	if (bTestHough)
	{
		char strImgRead[128];
		sprintf_s(strImgRead, "%s/road.jpg", g_strImgsRtPath.c_str());
		cv::Mat imRead = cv::imread(strImgRead);
		std::shared_ptr<lvcv::CHough> objHough =
			std::make_shared<lvcv::CHough>();

		//Test function detectLines
		if (0)
		{
			cv::Mat imDraw;
			cv::Mat imSrc = imRead.clone();
			std::vector<cv::Vec4i> vLines;
			objHough->detectLines(imSrc, vLines, &imDraw);
		}
		//Test function detectCircles
		if (0)
		{
			sprintf_s(strImgRead, "%s/chariot.jpg", g_strImgsRtPath.c_str());
			cv::Mat imRead = cv::imread(strImgRead);
			cv::Mat imDraw;
			cv::Mat imSrc = imRead.clone();
			std::vector<cv::Vec3f> vCircles;
			objHough->detectCircles(imSrc, vCircles, &imDraw);
		}
	}

	bool bTestFeatureDetector = true;
	if (bTestFeatureDetector)
	{
		char strImg[128];
		sprintf_s(strImg, "%s/boldt.jpg", g_strImgsRtPath.c_str());
		cv::Mat imRead = cv::imread(strImg);
		cv::Mat imGray;
		lvcv::CTools::convertToGray(imRead, imGray);
		//Test function detectKpsByFAST
		if (0)
		{
			std::vector<cv::KeyPoint> vKps;
			cv::Mat imSrc = imRead.clone();
			lvcv::CDetectorFeature::detectKpsByFAST(imGray, vKps, 40);
			lvcv::CTools::drawKps(imSrc, vKps);
		}
		//Test function detectKpsByGoodHarris
		if (0)
		{
			std::vector<cv::Point2f> vPnts;
			cv::Mat imSrc = imRead.clone();
			lvcv::CDetectorFeature::detectKpsByGoodharris(imGray, vPnts, 300);
			lvcv::CTools::drawKps(imSrc, vPnts);
		}
		//Test ORB feature detector
		if (0)
		{
			std::vector<cv::KeyPoint> vKps;
			cv::Mat des;
			cv::Mat imSrc = imRead.clone();
			std::unique_ptr<lvcv::CDetectorFeature> objDetector =
				std::make_unique<lvcv::CDetector_ORB>();
			objDetector->detectAndCompute(imGray, vKps, des);
			lvcv::CTools::drawKps(imSrc, vKps);
		}
		//Test SIFT feature detector
		if (0)
		{
			std::vector<cv::KeyPoint> vKps;
			cv::Mat des;
			std::unique_ptr<lvcv::CDetectorFeature> objDetector =
				std::make_unique<lvcv::CDetector_SIFT>();
			objDetector->detectAndCompute(imGray, vKps, des);
			cv::Mat imSrc = imRead.clone();
			lvcv::CTools::drawKps(imSrc, vKps);
		}
	}

	bool bTestMatcherFeature = false;
	if (bTestFeatureDetector)
	{
		char strImg01[128];
		char strImg02[128];
		sprintf_s(strImg01, "%s/church01.jpg", g_strImgsRtPath.c_str());
		sprintf_s(strImg02, "%s/church02.jpg", g_strImgsRtPath.c_str());
		cv::Mat imRead01 = cv::imread(strImg01);
		cv::Mat imRead02 = cv::imread(strImg02);
		std::vector<cv::KeyPoint> vKps01, vKps02;
		cv::Mat des01, des02;
		cv::Ptr<lvcv::CDetectorFeature> objDetector;
		cv::Mat imGray01, imGray02;
		lvcv::CTools::convertToGray(imRead01, imGray01);
		lvcv::CTools::convertToGray(imRead02, imGray02);
		cv::Ptr<lvcv::CMatcherFeature> objMatcher;
		std::vector<cv::DMatch> vMatches;
		//Test CMatcher_BF_HAM
		if (0)
		{
			objDetector = new lvcv::CDetector_ORB();
			objDetector->detectAndCompute(imGray01, vKps01, des01);
			objDetector->detectAndCompute(imGray02, vKps02, des02);
			objMatcher = new lvcv::CMatcher_BF_L2();
			int iRes = objMatcher->Match(des01, des02, vMatches);
			cv::Mat imMatchRes;
			lvcv::CTools::drawMatchingRes(imRead01, imRead02, vKps01, vKps02, vMatches, imMatchRes);
		}
		//Test CMatcher_BF_KNN_L2
		if (0)
		{
			objDetector = new lvcv::CDetector_BRISK();
			objDetector->detectAndCompute(imGray01, vKps01, des01);
			objDetector->detectAndCompute(imGray02, vKps02, des02);
			objMatcher = new lvcv::CMatcher_BF_KNN_HAM();
			int iRes = objMatcher->Match(des01, des02, vMatches);
			cv::Mat imMatchRes;
			lvcv::CTools::drawMatchingRes(imRead01, imRead02, vKps01, vKps02, vMatches, imMatchRes);
		}
		//Test CMatcher_Ransac_F
		if (0)
		{
			objDetector = new lvcv::CDetector_BRISK();
			objDetector->detectAndCompute(imGray01, vKps01, des01);
			objDetector->detectAndCompute(imGray02, vKps02, des02);
			objMatcher = new lvcv::CMatcher_Ransac_F(new lvcv::CMatcher_BF_HAM);
			cv::Ptr<lvcv::CMatcher_Ransac_F> objRansacF =
				objMatcher.dynamicCast<lvcv::CMatcher_Ransac_F>();
			objRansacF->Match(vKps01, vKps02, des01, des02, vMatches, true);
			cv::Mat imDraw;
			lvcv::CTools::drawMatchingRes(imRead01, imRead02, vKps01, vKps02, vMatches, imDraw);
		}
		//Test CMatcher_Ransac_H
		if (0)
		{
			objDetector = new lvcv::CDetector_BRISK();
			objDetector->detectAndCompute(imGray01, vKps01, des01);
			objDetector->detectAndCompute(imGray02, vKps02, des02);
			objMatcher = new lvcv::CMatcher_Ransac_H(new lvcv::CMatcher_BF_HAM);
			cv::Ptr<lvcv::CMatcher_Ransac_H> objRansacH =
				objMatcher.dynamicCast<lvcv::CMatcher_Ransac_H>();
			objRansacH->Match(vKps01, vKps02, des01, des02, vMatches, false);
			cv::Mat imDraw;
			lvcv::CTools::drawMatchingRes(imRead01, imRead02, vKps01,vKps02, vMatches, imDraw);
		}
	}

	bool bTestReconstruct3D = false;
	if (bTestReconstruct3D)
	{
		std::string strChessRootPath = "D:\\MjData\\images\\chessboards";
		//Test camera calibrator
		if (1)
		{
			cv::Ptr<lvcv::CCamCalibrator> objCalibrator =
				new lvcv::CCamCalibrator(strChessRootPath.c_str());
			cv::Size szChessboard(7, 5);
			objCalibrator->getChessboardPoints2D(szChessboard);
			objCalibrator->getChessboardPoints3D(szChessboard);
			double dError = 0.0;
			objCalibrator->calibrateCamera(dError);

		}
	}

	bool bTestVideoProcessor = true;
	if (bTestVideoProcessor)
	{
		std::string strVideoPath = g_strImgsRtPath;
		strVideoPath.append("/bike.avi");
		if (0)
		{
			std::string strVideoSavePath = g_strImgsRtPath;
			strVideoSavePath.append("/bike_out01.avi");
			lvcv::CVideoProcessor objVideo(true);
			objVideo.openReader(strVideoPath);
			objVideo.openWriter(strVideoSavePath);
			objVideo.setImageProcessor(new lvcv::CImageCanny(100, 300));
			objVideo.showVideo(true, true);
		}
		if (1)
		{
			//The source path of image sequence
			std::string strSourcePath;
			strSourcePath.assign(g_strImgsRtPath).append("\\goose");
			//The destination path of image sequence
			std::string strDestinationPath;
			strDestinationPath.assign(g_strImgsRtPath).append("\\goose_out01");
			//open the reader
			lvcv::CImgSeqProcessor objSeq;
			bool bRes = objSeq.openReader(strSourcePath);
			//open the writer
			bRes = objSeq.openWriter(strDestinationPath);
			//set image-processor
			objSeq.setImageProcessor(new lvcv::CImageCanny(80,240));
			//show result
			objSeq.showVideo(true, true);
		}

	}
}