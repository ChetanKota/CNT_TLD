

#include "color_tracker.hpp"

using namespace cv::colortracker;

void test_video()
{
	ColorTrackerParameters params;

	params.visualization = 1;

	params.video_path = "Pigeon.avi";
	
	cv::Point pos = cv::Point(500, 365);
	cv::Size target_sz = cv::Size(60, 60);
	params.init_pos.x = (int)(floor(pos.x) + floor(target_sz.width / 2));
	params.init_pos.y = (int)(floor(pos.y) + floor(target_sz.height / 2));
	params.wsize = cv::Size((int)floor(target_sz.width), (int)floor(target_sz.height));
	ColorTracker tracker(params);
	tracker.track_video(1, 3.5);
}

void test_deer()
{
	ColorTrackerParameters params;

	params.visualization = 1;
	cv::Point pos = cv::Point(306,5);
	cv::Size target_sz = cv::Size(95,65);
	params.init_pos.x = (int)(floor(pos.x) + floor(target_sz.width / 2));
	params.init_pos.y = (int)(floor(pos.y) + floor(target_sz.height / 2));
	params.wsize = cv::Size((int)floor(target_sz.width), (int)floor(target_sz.height));
	
	ColorTracker tracker(params);
	tracker.init_tracking();
	for(int frame_index = 1;frame_index <= 71;frame_index++)
	{
		ostringstream ostr;
		ostr << "sequences/deer/imgs/img";
		ostr << setfill('0') << setw(5) << frame_index << ".jpg";
		cv::Mat current_frame = cv::imread(ostr.str());
		tracker.track_frame(current_frame);
	}
	cv::waitKey(0);
}

void test_soccer()
{
	ColorTrackerParameters params;

	params.visualization = 1;
	cv::Point pos = cv::Point(302,135);
	cv::Size target_sz = cv::Size(67,81);
	params.init_pos.x = (int)(floor(pos.x) + floor(target_sz.width / 2));
	params.init_pos.y = (int)(floor(pos.y) + floor(target_sz.height / 2));
	params.wsize = cv::Size((int)floor(target_sz.width), (int)floor(target_sz.height));
	
	ColorTracker tracker(params);
	tracker.init_tracking();
	for(int frame_index = 1;frame_index <= 392;frame_index++)
	{
		ostringstream ostr;
		ostr << "sequences/Soccer/img/";
		ostr << setfill('0') << setw(4) << frame_index << ".jpg";
		cv::Mat current_frame = cv::imread(ostr.str());
		tracker.track_frame(current_frame);
	}
	cv::waitKey(0);
}

void test_Scate()
{
	ColorTrackerParameters params;

	params.visualization = 1;
	cv::Point pos = cv::Point(162,188);
	cv::Size target_sz = cv::Size(34,84);
	params.init_pos.x = (int)(floor(pos.x) + floor(target_sz.width / 2));
	params.init_pos.y = (int)(floor(pos.y) + floor(target_sz.height / 2));
	params.wsize = cv::Size((int)floor(target_sz.width), (int)floor(target_sz.height));
	
	ColorTracker tracker(params);
	tracker.init_tracking();
	for(int frame_index = 1;frame_index <= 400;frame_index++)
	{
		ostringstream ostr;
		ostr << "sequences/Skating1/img/";
		ostr << setfill('0') << setw(4) << frame_index << ".jpg";
		cv::Mat current_frame = cv::imread(ostr.str());
		tracker.track_frame(current_frame);
	}
	cv::waitKey(0);
}

int main(int argc, char** argv)
{
	test_Scate();
	test_soccer();
	test_video();
	test_deer();
	return 0;
}
