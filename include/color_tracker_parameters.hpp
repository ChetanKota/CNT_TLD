

#ifndef _OPENCV_color_tracker_parameters_HPP_
#define _OPENCV_color_tracker_parameters_HPP_ 

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <iomanip>
#include <fstream>

#ifdef __cplusplus
namespace cv { namespace colortracker {


		//using namespace cv;
		using namespace std;

		class ColorTrackerParameters
		{
		public:
			//parameters according to the paper
			double padding;//params.padding = 1.0;         			   % extra area surrounding the target
			double output_sigma_factor;//params.output_sigma_factor = 1 / 16;		   % spatial bandwidth(proportional to target)
			double sigma;//params.sigma = 0.2;         			   % gaussian kernel bandwidth
			double lambda;//params.lambda = 1e-2;					   % regularization(denoted "lambda" in the paper)
			double learning_rate;//params.learning_rate = 0.075;			   % learning rate for appearance model update scheme(denoted "gamma" in the paper)
			double compression_learning_rate;//params.compression_learning_rate = 0.15;   % learning rate for the adaptive dimensionality reduction(denoted "mu" in the paper)
			vector<string> non_compressed_features;//params.non_compressed_features = { 'gray' }; % features that are not compressed, a cell with strings(possible choices : 'gray', 'cn')
			vector<string> compressed_features;//params.compressed_features = { 'cn' };       % features that are compressed, a cell with strings(possible choices : 'gray', 'cn')
			int num_compressed_dim;//params.num_compressed_dim = 2;             % the dimensionality of the compressed features
			//
			//params.visualization = 1;
			int visualization;
			string video_path;
			cv::Point init_pos;
			cv::Size wsize;

			ColorTrackerParameters();
		};

	}
}
#endif
#endif
