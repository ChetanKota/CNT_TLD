//Aum Sri Sai Ram
// Tracker using Latch descriptor written by Chetan Kota
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include<bitset>
#include<iomanip>
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include<opencv2/imgproc/imgproc.hpp>
//#include "opencv2/nonfree/nonfree.hpp"
#define BITS 64
#define IN_HOMO 0.8
using namespace cv;
using namespace std;


constexpr float points[3073]={-5,-16,-9,1,16,-21,-7,-10,-3,16,-14,9,11,7,3,6,15,-11,-22,-12,-22,2,12,21,17,2,14,-10,10,-18,22,-2,23,-14,-20,5,4,24,4,16,-11,-21,13,19,23,-6,19,-4,22,1,-10,7,-8,6,-5,19,-6,18,11,-16,12,24,20,-9,20,-20,-3,5,10,-14,-21,19,-3,-17,-5,9,-1,-19,7,22,13,-2,-23,20,19,-1,15,12,-19,-9,-19,-2,-22,2,-9,24,-2,-11,-3,-21,-18,7,23,4,17,8,7,-19,16,10,20,18,-23,4,-6,-1,6,22,-11,-14,-24,15,-23,-15,-14,-11,24,18,14,-13,-14,-19,-17,-11,-21,10,-12,-17,19,3,-2,-4,8,19,-18,-8,-18,-3,-15,5,-6,1,-6,-13,16,23,-22,-4,-15,-8,-5,20,9,-6,10,13,-23,15,20,11,9,12,-24,22,-19,-1,-20,-4,5,2,-24,15,-19,4,-16,-5,-23,-11,-16,10,5,19,-3,-22,-7,5,23,-22,-15,-9,-21,2,21,20,-18,16,-12,11,-10,4,2,14,6,-7,-13,20,3,23,5,15,-23,-9,1,-1,-3,8,-4,7,-22,-1,16,16,19,-12,14,9,19,-12,13,12,21,-22,15,2,-17,3,0,11,8,5,-22,18,-15,13,-18,9,3,-17,-12,8,-21,10,5,-18,19,0,-12,17,18,-7,-14,1,-12,-20,-7,-2,24,3,-16,15,-22,7,-17,24,-24,13,-14,-24,-7,-4,-1,-23,10,20,-8,-1,-5,-13,20,-22,13,16,0,-22,16,-14,-10,0,-9,12,-9,-16,20,-2,19,15,-19,-15,-19,22,-14,19,23,-2,-24,10,-8,16,-3,-1,-22,18,-23,-15,23,21,9,9,21,10,-23,-24,-17,12,-12,-19,-23,-8,19,0,-18,17,-16,-19,-19,-7,-14,21,20,3,20,8,17,-2,0,-10,-15,6,-21,24,2,-22,-18,8,-24,12,19,-14,16,-13,24,-1,-16,-18,23,4,20,-1,-10,22,22,2,16,-14,-22,11,-14,24,-9,-19,-16,-6,11,0,-20,-5,24,-12,17,-9,18,16,1,-15,15,-24,24,-10,16,11,5,-3,4,3,-16,-2,9,-15,7,19,-18,-19,11,-5,14,6,22,-10,-8,17,-19,-3,24,17,13,24,9,16,-20,-4,7,16,8,-6,11,1,3,18,20,23,15,-24,22,2,-12,15,18,-10,-15,11,-13,13,9,9,-10,17,-13,-9,13,-13,22,-11,20,15,7,-9,18,-11,17,0,16,11,22,-19,24,-18,-3,-15,22,-5,19,20,16,-23,21,0,-17,14,-8,2,9,2,7,0,3,12,7,-12,13,18,8,-24,-5,-15,-22,-11,-10,9,19,20,24,24,6,-11,-10,2,-17,6,-2,-21,19,21,21,-17,21,-13,-11,-20,-9,3,-5,-8,-3,-7,-1,-20,23,-9,-3,22,-1,21,-12,3,-19,2,-24,-4,-3,17,-16,2,-19,10,1,20,-9,-20,-17,21,-17,22,-18,-9,-12,7,19,-5,22,8,11,6,-7,22,-24,5,-24,10,-15,-20,10,-22,9,20,-4,-20,-24,0,-23,6,-13,-15,22,-3,19,-20,24,5,10,-1,8,-23,-9,-5,-9,23,-22,-6,0,13,-9,23,-17,-7,-13,-20,20,-21,22,7,15,-5,24,-24,12,24,22,-12,16,-18,9,-13,11,-6,-10,2,18,-9,15,18,-18,1,-24,3,24,20,-23,-22,-17,6,-12,7,-19,-18,-2,-18,22,13,-22,4,-22,24,8,2,7,10,20,-12,19,21,-18,3,-16,-7,8,5,-20,4,-17,-14,-9,-8,-7,-20,-20,-6,18,-3,0,4,18,4,-8,-8,-15,1,-17,-18,-22,10,8,21,24,13,17,-3,14,6,17,10,-24,-10,-22,12,-18,8,-8,23,-13,21,-15,-8,18,11,20,7,5,17,7,-24,-7,1,9,-12,14,17,16,-24,24,11,-2,21,10,-15,21,9,15,16,-10,14,-22,3,-11,8,22,-22,14,-13,4,3,-1,23,-2,21,-14,-17,16,-7,19,-7,-23,6,21,-15,6,-22,6,-4,11,-1,-10,7,-21,-4,-10,-2,-13,-23,23,5,-21,18,-8,-24,11,5,-18,23,-21,-24,-16,16,-8,19,-23,6,-13,23,-24,-16,-1,-23,13,-11,-23,12,-24,-10,1,2,-14,2,-20,-20,-11,24,-1,20,-6,-4,-22,-19,12,13,15,2,-16,1,11,-24,24,13,10,21,-10,8,-13,-12,10,24,23,-13,16,10,-14,10,0,-21,0,22,20,18,24,-13,18,7,7,19,-10,-24,-13,-24,-20,-5,19,0,21,-4,21,16,0,-13,-2,-15,24,9,-12,-10,-6,19,6,-16,-10,-4,-14,4,-22,-8,24,-18,20,-7,24,-1,18,-8,23,7,3,-11,-20,-11,-5,-6,17,-19,12,1,12,0,-11,-16,22,-2,-24,-1,-3,22,17,-7,17,17,17,-12,-22,-17,-6,-6,14,11,-11,-4,-2,-13,-9,-18,-24,12,-16,16,-5,13,-20,15,-14,23,-10,-16,21,-18,12,13,0,-12,-18,-20,-18,14,-22,12,-16,3,-17,21,-3,-15,12,15,5,15,-19,4,15,-16,17,-8,20,23,-21,7,-7,19,-2,1,-22,21,-21,13,19,-10,9,21,6,0,-23,-24,-24,4,-24,13,16,-13,-8,15,-11,-18,23,23,18,2,15,10,23,6,-14,22,-21,-14,9,17,3,-24,-2,-12,0,-3,18,-9,19,18,7,17,20,-20,11,-11,0,4,5,11,18,-7,-22,-5,-15,24,-17,-23,22,18,-23,-15,-7,-5,-14,11,-3,3,-4,-7,17,-13,19,-2,17,-19,24,1,2,8,11,-21,20,-12,22,-9,8,11,-14,-4,-19,20,-22,2,-14,23,9,-24,9,-17,-22,17,6,-1,-4,-21,0,18,8,-9,12,-13,-23,13,7,-13,16,-2,5,-5,-19,9,-10,-23,23,9,1,24,4,-17,-24,3,10,-2,12,-5,-1,15,22,6,-6,11,20,-9,24,8,-7,-22,23,-10,3,5,2,8,-6,14,13,-10,7,-12,11,11,16,-7,21,-13,-7,-12,9,14,5,-8,18,-9,12,-2,8,-22,13,21,3,-3,-10,-1,-2,19,-13,7,-21,-8,-5,-17,16,-4,18,-14,-22,17,13,23,12,2,-2,5,-12,-4,-14,1,9,-2,-17,6,-12,22,-9,-13,-9,22,-24,2,-1,-10,-14,-23,-13,14,6,-3,-23,7,-23,-12,12,16,-6,4,3,20,22,-20,5,14,3,-15,12,-13,19,-11,24,-22,-12,-20,9,-12,16,17,21,24,-20,-10,-14,-9,16,-21,-7,-23,-4,-20,10,-8,18,-14,20,6,24,10,-2,4,10,-21,-16,-2,-18,-11,-15,-12,14,-21,14,-15,18,-7,-17,-2,12,-1,20,18,0,7,-13,-12,-17,14,23,18,21,-5,24,-7,-15,-6,1,16,-5,1,9,18,22,-4,22,3,18,8,-10,6,-16,9,4,-24,5,-4,-18,-18,-12,-24,-16,-6,-19,-19,18,23,19,-23,23,-4,-10,-13,13,-6,19,8,19,-6,8,-17,12,-17,0,22,8,7,22,22,-22,-22,22,-19,19,-23,-17,16,0,21,13,12,21,-22,-19,-15,-12,3,-16,1,-1,22,-1,-24,19,13,-9,14,-6,7,24,-1,-18,-11,-18,-3,-22,-5,-13,1,15,9,18,7,-5,10,24,12,-24,22,-7,0,-1,-15,3,24,21,-23,19,-16,-5,8,18,-12,14,18,-16,-5,-20,16,-20,19,-2,-19,-20,22,-23,-13,21,-3,19,-16,23,-19,24,7,-19,-1,18,5,20,15,13,22,24,-11,3,-14,-19,14,-23,-9,12,17,-6,14,-9,-1,-22,-9,19,-1,-21,-18,23,-10,22,17,14,1,-12,-23,7,0,-3,-8,8,21,24,-2,24,2,-6,-12,8,2,9,14,-3,6,-10,-3,-11,-24,18,-5,-12,-20,-7,12,-5,-6,-17,-2,-17,-21,-15,-15,5,21,9,-15,-23,-12,-10,10,-8,10,4,7,23,-3,9,4,11,4,-1,-9,-21,-21,22,-4,-12,19,-15,-3,-24,0,-24,-8,-19,9,17,-7,20,-16,21,-24,-11,19,11,-21,19,-14,23,-21,23,-9,20,7,17,-6,-18,-5,5,21,-17,11,-3,-15,-21,-16,-13,11,14,-2,-20,3,2,-17,-16,19,-12,-21,-3,15,21,19,5,20,-6,-1,21,9,1,12,23,-9,9,-6,22,7,6,0,4,14,23,-21,-4,18,-6,-12,-23,-24,-13,9,14,22,6,21,-19,-9,-2,-8,-5,22,-16,23,-3,23,-3,19,10,-18,-22,7,-19,21,12,-12,-9,23,-16,-6,-24,7,1,6,14,21,15,-3,4,-2,-11,18,10,20,3,23,23,-20,-13,23,11,21,20,-12,19,-8,14,14,13,-4,22,-14,14,23,10,-12,7,21,2,4,11,-20,-13,-11,14,8,16,-13,24,-14,-9,17,-13,-8,-3,21,-17,4,-17,16,0,24,-22,-15,-1,18,-24,8,-16,-7,-14,-10,22,5,1,11,-1,-24,-9,10,7,-12,14,9,-4,18,21,15,5,-3,-23,-5,16,8,18,24,-11,23,17,-1,18,-12,13,19,-18,10,-14,12,-14,24,-4,-5,-9,20,12,20,16,-5,15,-8,-20,-7,-24,5,-16,10,-11,-6,-21,12,-24,-15,14,-15,21,-1,19,6,-5,19,24,12,3,10,12,-15,-22,1,-17,-13,-4,4,19,-8,22,-12,14,-23,11,-9,16,-4,-10,2,-24,-8,5,21,-21,14,14,-21,22,-11,-8,-22,-23,-1,19,-5,-11,13,13,5,4,-12,18,-24,-13,2,-14,22,9,19,13,-15,9,22,19,-17,-3,4,-3,5,-2,-14,11,2,4,-21,-15,2,-17,-5,-16,-1,15,-20,15,0,7,-7,-20,0,-2,23,-1,12,-19,-10,-3,13,19,-18,15,-23,-21,5,-19,10,3,18,-17,9,-16,-12,19,24,-20,-1,-24,-24,-22,-9,12,-16,18,-5,-19,-6,-12,10,-8,-23,8,-11,17,-8,15,21,-13,11,11,24,11,14,-23,6,15,7,19,22,-23,14,-4,21,-2,18,7,5,23,24,10,10,-24,10,-14,-4,-21,24,3,2,2,24,6,-6,7,24,14,-16,18,-8,-15,11,-21,-8,-20,0,14,23,-23,11,-24,22,7,22,7,24,9,19,0,23,14,-6,20,-9,7,-23,20,9,13,1,9,21,-11,-1,9,-11,15,23,8,-9,23,2,10,21,-19,4,-19,-22,-2,20,14,20,19,-23,1,23,12,-1,-15,22,13,-18,24,-5,21,11,6,12,11,-2,12,5,15,7,-16,10,-15,-13,10,17,-18,-15,-19,-5,3,8,12,12,14,-9,20,-12,-13,-23,-22,-18,-8,13,15,-22,16,-18,18,14,7,23,20,-9,2,-16,-17,-7,-24,-23,9,23,9,-5,5,-22,-16,-11,-22,-11,-20,11,-2,14,11,8,16,-24,18,-14,17,1,-6,12,7,18,4,8,7,-19,4,18,-10,-1,-19,19,-23,-23,-1,23,1,5,13,-24,-16,14,-19,-19,21,-22,-9,-23,-9,5,16,3,12,21,13,11,15,23,-21,23,-21,-12,-24,-17,-14,-8,-5,-21,13,-1,17,24,10,19,-4,6,21,-24,13,-22,-2,-8,23,-16,21,-6,-5,-15,-12,14,-11,-8,-6,5,-5,9,16,9,-23,24,12,-24,22,8,19,2,-10,0,18,23,3,-22,-19,3,-12,-20,15,6,-23,19,-23,5,-3,23,24,8,4,10,24,-3,8,15,7,-23,-2,20,19,18,23,4,-24,6,-9,-23,-6,16,-2,2,10,7,12,-1,21,7,-6,2,20,16,5,6,11,3,-4,-14,9,-12,13,-17,11,-22,23,15,11,1,17,22,4,-3,18,-9,21,11,-14,-23,-9,14,-14,-3,14,4,12,-4,21,0,-24,16,-21,6,-16,-15,-23,-20,14,-22,0,-20,-13,1,7,-12,13,0,-15,-11,16,0,10,17,-1,21,14,2,-9,-18,24,-5,-12,23,-10,-6,8,9,16,0,-14,24,-10,-16,-21,-14,-23,12,-21,13,-21,21,-21,-24,2,-15,1,19,13,8,8,-23,-8,6,17,-19,-24,7,-24,18,-21,10,20,-24,3,11,-4,0,22,-12,8,5,18,-3,17,10,-21,18,-21,18,-22,-6,-9,2,-5,24,-17,4,2,6,5,5,-24,21,20,-24,5,-17,16,-3,-10,-2,5,5,-22,-9,-3,-13,-21,-12,4,23,6,8,3,24,13,-8,5,19,11,-6,10,22,-18,17,-18,-10,-18,10,20,21,21,17,-3,18,-23,15,-19,2,-21,-10,-16,22,-18,-18,2,22,2,0,5,19,-8,7,5,-21,4,4,-14,2,-16,1,18,19,15,4,-15,-2,-20,-6,-18,-8,-19,-7,0,-21,10,-23,-15,1,-16,22,7,0,20,12,22,-12,18,1,23,14,13,-2,23,-2,5,16,3,-1,-7,9,-11,11,-13,-11,-20,-23,-22,9,-20,2,-23,-3,2,-1,17,3,24,19,-11,1,-24,3,20,20,-15,-8,-20,12,-14,-11,20,-8,19,-14,-14,9,-24,15,-8,-22,9,-4,22,0,0,-7,24,-3,19,-17,23,-17,1,11,-24,3,-16,-18,15,-7,10,-10,9,0,23,4,9,2,15,9,-20,-23,16,11,7,-19,8,23,17,-14,21,-17,-18,-14,-10,11,-10,-24,2,13,24,7,-2,15,-3,11,-20,18,-9,21,4,-7,-5,-18,-23,10,-6,11,9,0,13,-1,-24,-12,17,10,1,-1,-23,15,24,-8,24,-2,-4,14,-18,0,18,6,-24,-23,16,-19,11,-9,-18,-1,-24,-5,-23,-3,22,15,-23,0,-19,-15,9,12,3,-11,3,-10,22,-21,-3,-21,-14,-15,-20,19,22,8,1,-21,-11,23,-9,-4,-10,-17,-9,-2,15,20,9,20,-24,3,-16,19,-19,-16,0,-21,17,12,8,9,-12,11,19,-23,12,12,2,0,24,-11,24,-16,-22,-19,18,1,-7,-2,5,-23,10,23,2,-21,1,-10,15,-24,16,-11,4,1,16,-20,14,-13,-6,13,12,15,10,6,-12,-20,-23,13,-18,-5,21,1,-19,-7,-23,-1,-21,-15,-21,-9,-18,-8,-7,-15,0,18,-2,-17,-12,10,-22,-2,19,21,-14,22,24,6,18,1,21,-19,14,-14,15,12,3,17,-14,-3,4,-13,5,12,12,22,24,-11,-9,-23,0,11,11,9,-20,1,12,12,18,16,-4,9,19,-2,-23,-17,11,22,15,13,22,-4,17,-3,-9,11,-13,-2,-14,-5,5,-12,14,-4,18,-12,21,2,-19,8,-23,8,-10,0,8,-17,12,19,18,2,10,-18,-21,13,13,-3,19,4,-19,1,-21,8,-24,-10,-1,-23,18,9,9,9,20,3,-11,18,-5,-24,-23,3,-15,17,-19,-13,12,19,13,3,13,-14,-15,-23,10,-23,16,-10,-2,16,-14,-23,-6,-21,19,-6,-5,-14,-2,21,-8,11,13,16,15,6,-24,14,-21,-8,-1,-21,1,14,19,14,9,-19,5,21,8,-18,17,-6,12,1,-19,14,-21,-11,-13,23,12,-3,9,-18,-17,-14,12,21,19,16,-11,19,-23,-3,14,22,-24,20,-7,13,-18,-5,-18,-23,-22,-9,-10,-15,-18,16,-5,-12,-4,18,10,-16,9,-15,17,-10,6,16,16,-1,-20,-6,-7,-20,-24,-18,-13,-1,8,-2,16,6,18,-22,11,-12,4,-15,1,16,-20,6,17,2,21,23,15,-4,21,-24,13,3,21,8,-4,4,-12,-1,19,-7,11,5,-2,-12,-21,-13,-19,-11,-20,20,-2,-24,11,-1,-3,20,-4,-23,11,-1,14,2,7,-2,-16,-1,-17,23,-8,-14,21,6,10,-15,19,-24,15,12,1,-23,-6,11,-19,19,-18,17,0,-8,0};

int border_clear = 25;
int next_frame = 10;

void CalcuateSums(int , const cv::Mat&, const cv::KeyPoint&, int&, int&, float , float);
static void pixelTests64(const cv::Mat& , const std::vector<cv::KeyPoint>& , cv::Mat&);
void LATCH(const cv::Mat , cv::Mat&, std::vector<cv::KeyPoint>&);
float median(std::vector<float> );
std::vector<float> normCrossCorrelation(const Mat& ,const Mat& , vector<Point2f>& , vector<Point2f>& );
bool Latch_track(const cv::Mat& ,const cv::Mat& , std::vector<cv::KeyPoint>& ,  std::vector<cv::Point2f>& ,const cv::Rect );
void box_set(const cv::Rect ,const cv::Rect , cv::Rect&);


/*int main()
{
	int init_x=84, //			84,53,62,70			184, 43, 67, 75 
		init_y=53,
		init_w=62,
		init_h=70;
	cv::Rect init_rect = cv::Rect(init_x,init_y,init_w,init_h);
	cv::Rect next_bb = cv::Rect(init_x,init_y,init_w,init_h);
	cv::Mat frame1,frame2,image1,image2;
	char name[] = "./Mhyang/img/";
	char path[70];
	int frame_no = 1;
while(frame_no<1490)
{
	sprintf(path,"%s%04d.jpg",name,frame_no);
	frame1 = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	sprintf(path,"%s%04d.jpg",name,frame_no+1);
	frame2 = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);

	std::vector<cv::Point2f> keypoints2;
	std::vector<cv::KeyPoint> keypoints1;
	
	cv::Mat temp1 = frame2.clone();
	bool tbb = Latch_track(frame1,frame2,keypoints1,keypoints2,next_bb);
	if(tbb == true)
	{	
		next_bb = cv::boundingRect(keypoints2);
		box_set(next_bb, init_rect, next_bb);
		cv::rectangle(temp1,next_bb,cv::Scalar(255),4,8,0);
	}
	cv::imshow("frame2:Output",temp1);
	cv::waitKey(30);
frame_no++;
}
}*/

void box_set(const cv::Rect box1,const cv::Rect initb, cv::Rect &box2)
{
	initb.width = 37; initb.height =114; //FIXED Bounding BOX width to be 50 x 50
 	int cur_cen_x = box1.x+box1.width/2;
	int cur_cen_y = box1.y+box1.height/2;
	box2.x = cur_cen_x - initb.width/2;
	box2.y = cur_cen_y - initb.height/2;
	box2.width = initb.width;
	box2.height = initb.height;
}


bool Latch_track(const cv::Mat& frame1,const cv::Mat& frame2, std::vector<cv::KeyPoint>& dummy_points,  std::vector<cv::Point2f>& keypoints_2,const cv::Rect next_bb)
{
cv::Mat image1,image2;
std::vector<cv::KeyPoint> keypoints1,keypoints2;
if(((next_bb.x-(border_clear/2)) < 0) || 
((next_bb.y-(border_clear/2)) < 0) || 
((next_bb.width+(border_clear)) > frame1.cols) || 
((next_bb.height+(border_clear)) > frame1.rows ))
{ cout<<"OUT:from1";return false;}

image1 = frame1(cv::Rect(next_bb.x-(border_clear/2), next_bb.y-(border_clear/2), next_bb.width+(border_clear), next_bb.height+(border_clear))).clone();


if( ((next_bb.x-(next_frame/2)) < 0) || 
((next_bb.y-(next_frame/2)) < 0 ) || 
(next_bb.width+next_frame > frame2.cols) || 
((next_bb.height+next_frame) > frame2.rows) )
{cout<<"OUT:from2";return false;}

image2 = frame2(cv::Rect(next_bb.x-(next_frame/2), next_bb.y-(next_frame/2), next_bb.width+next_frame, next_bb.height+next_frame)).clone();


cv::Mat desc1,desc2;
LATCH(image1,desc1,keypoints1);
LATCH(image2,desc2,keypoints2);

cv::BFMatcher matcher(cv::NORM_HAMMING);
std::vector<cv::DMatch> matches;
matcher.match( desc1, desc2, matches);

 double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < desc1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector<cv::DMatch> good_matches;

  for( int i = 0; i < desc1.rows; i++ )
  { 
	if( matches[i].distance < IN_HOMO*max_dist ){ 
		good_matches.push_back( matches[i]); 
	}
  }

std::vector<cv::Point2f> points1,points2;
std::vector<cv::Point2f>::iterator iz;
int z=0;
for (int i=0; i<good_matches.size(); i++)
{
	cv::Point2f dummy;
	cv::KeyPoint temp1 = keypoints1[good_matches[i].trainIdx];
	dummy.x = temp1.pt.x+(int)(next_bb.x-(border_clear/2));
	dummy.y = temp1.pt.y+(int)(next_bb.y-(border_clear/2));
	points1.push_back(dummy);

	cv::KeyPoint temp2 = keypoints2[good_matches[i].queryIdx];
	dummy.x = temp2.pt.x+(int)(next_bb.x-(border_clear/2));
	dummy.y = temp2.pt.y+(int)(next_bb.y-(border_clear/2));
	points2.push_back(dummy);
	z=z+1;
}
points1.resize(z);
points2.resize(z);

std::vector<float> similarity = normCrossCorrelation(frame1,frame2,points1,points2);

std::vector<cv::Point2f> Return_vec;

float simmed = median(similarity);
int k=0;
for(int i=0;i<points1.size();i++)
{
	if(similarity[k] >= simmed)
	{
		Return_vec.push_back(points1[i]);
	}
	k++;
}


cv::Mat result = frame2.clone();
	for(int i=0;i<points2.size();i++){
		cv::Point2f center(points2[i].x,points2[i].y);
		cv::circle(result, center ,3, (90), 3, 8, 0);}
	cv::imshow("Keypoints_got in new frame",result); 

 result = frame2.clone();
	for(int i=0;i<(int )Return_vec.size();i++){
		cv::Point2f center(Return_vec[i].x,Return_vec[i].y);
		cv::circle(result, center ,3, (90), 3, 8, 0);}
	cv::imshow("Keypoints_got in after SIM",result); 

keypoints_2 = Return_vec;
return true;

}

std::vector<float> normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
        Mat rec0(10,10,CV_8U);
        Mat rec1(10,10,CV_8U);
        Mat res(1,1,CV_32F);
	std::vector<float> similarity;
        for (int i = 0; i < points1.size(); i++) 
	{
		getRectSubPix( img1, Size(10,10), points1[i],rec0 );
                getRectSubPix( img2, Size(10,10), points2[i],rec1);
                matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);
                similarity.push_back(((float *)(res.data))[0]); 
        }
        rec0.release();
        rec1.release();
        res.release();
	return similarity;
}

float median(std::vector<float> scores)
{
	size_t size = scores.size();
	if (size == 0){
    		return 0;  // Undefined, really.
  	}
	else if (size == 1){
		return scores[0];
	}
	else{
		sort(scores.begin(), scores.end());
		if (size % 2 == 0){
			return (scores[size / 2 - 1] + scores[size / 2]) / 2;
		}
		else {
			return scores[size / 2];
		}
	}
}

void CalcuateSums(int count, const cv::Mat &grayImage, const cv::KeyPoint &pt, int &suma, int &sumc, float cos_theta, float sin_theta)
{	
	int patch_size = 3; 		
	int half_ssd_size = (patch_size - 1)/2;
	bool rotationInvariance = true;
	
            int ax = points[count];
            int ay = points[count + 1];

            int	bx = points[count + 2];
            int	by = points[count + 3];

            int cx = points[count + 4];
            int cy = points[count + 5];
		//std::cout<<"triplet "<<count<<": a:"<<ax<<" "<<ay<<"\t b:"<<bx<<" "<<by<<"\t c:"<<cx<<" "<<cy<<std::endl;
            int ax2 = ax;
            int ay2 = ay;
            int bx2 = bx;
            int by2 = by;
            int cx2 = cx;
            int cy2 = cy;
	
            if (rotationInvariance)
		{
		ax2 =(int)(((float)ax)*cos_theta - ((float)ay)*sin_theta);
                ay2 = (int)(((float)ax)*sin_theta + ((float)ay)*cos_theta);
                bx2 = (int)(((float)bx)*cos_theta - ((float)by)*sin_theta);
                by2 = (int)(((float)bx)*sin_theta + ((float)by)*cos_theta);
                cx2 = (int)(((float)cx)*cos_theta - ((float)cy)*sin_theta);
                cy2 = (int)(((float)cx)*sin_theta + ((float)cy)*cos_theta);


                if (ax2 > 24)
                    ax2 = 24;
                if (ax2<-24)
                    ax2 = -24;

                if (ay2>24)
                    ay2 = 24;
                if (ay2<-24)
                    ay2 = -24;

                if (bx2>24)
                    bx2 = 24;
                if (bx2<-24)
                    bx2 = -24;

                if (by2>24)
                    by2 = 24;
                if (by2<-24)
                    by2 = -24;

                if (cx2>24)
                    cx2 = 24;
                if (cx2<-24)
                    cx2 = -24;

                if (cy2>24)
                    cy2 = 24;
                if (cy2 < -24)
                    cy2 = -24;

            }


            ax2 += (int)(pt.pt.x + 0.5);
            ay2 += (int)(pt.pt.y + 0.5);

            bx2 += (int)(pt.pt.x + 0.5);
            by2 += (int)(pt.pt.y + 0.5);

            cx2 += (int)(pt.pt.x + 0.5);
            cy2 += (int)(pt.pt.y + 0.5);


            int K = half_ssd_size;
            for (int iy = -K; iy <= K; iy++)
            {
                const uchar*  Mi_a = grayImage.ptr<uchar>(ay2 + iy);
                const uchar*  Mi_b = grayImage.ptr<uchar>(by2 + iy);
                const uchar*  Mi_c = grayImage.ptr<uchar>(cy2 + iy);
		

                for (int ix = -K; ix <= K; ix++)
                {
                    double difa = Mi_a[ax2 + ix] - Mi_b[bx2 + ix];
                    suma += (int)((difa)*(difa));

                    double difc = Mi_c[cx2 + ix] - Mi_b[bx2 + ix];
                    sumc += (int)((difc)*(difc));
                }
            	}
}

static void pixelTests64(const cv::Mat& grayImage, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
            //Mat descriptors = _descriptors.getMat();
		static int temp=0;
		std::vector<uchar*> desc_vector= std::vector<uchar*>();
		std::vector<uchar> desc_row = std::vector<uchar>();
		std::vector<std::vector<uchar>> desc_row_col = std::vector<std::vector<uchar>>();
            for (int i = 0; i < (int)keypoints.size(); ++i)
            {
                //uchar* desc = descriptors.ptr(i);
		uchar* desc=(uchar*)malloc(sizeof(uchar)*BITS);
		
                const cv::KeyPoint& pt = keypoints[i];
                int count = 0;

                //handling keypoint orientation
                float angle = pt.angle;
                angle *= (float)(CV_PI / 180.f);
                float cos_theta = cos(angle);
                float sin_theta = sin(angle);
                for (int ix = 0; ix < BITS; ix++)
		{
			desc[ix] = 0;
			for (int j = 7; j >= 0; j--)
			{
				int suma = 0;
				int sumc = 0;
				
				CalcuateSums(count,  grayImage, pt, suma, sumc, cos_theta, sin_theta);
				desc[ix] += (uchar)((suma < sumc) << j);
				count += 6;
			} desc_row.push_back(desc[ix]);
		}
		desc_vector.push_back(desc); //making a vector of descriptors for all the keypoints.
		desc_row_col.push_back(desc_row);
		
		if(i==0){std::cout<<"desc_row_size: "<<desc_row.size()<<std::endl;}
		desc_row.clear();
            }//end of for loop over keypoints
	std::cout<<"desc vector_size: "<<desc_vector.size()<<std::endl;
	std::vector<std::vector<uchar>>::iterator ii;
	std::vector<uchar>::iterator ij;
	cv::Mat descriptors_return = cv::Mat(desc_row_col.size(),BITS, CV_8U,cvScalar(0));
	int i,j;
	for(ii = desc_row_col.begin(),i=0; ii < desc_row_col.end();ii++,i++)
	{
	for(ij = (*ii).begin(),j=0; ij < (*ii).end();ij++,j++)
	{
		descriptors_return.at<uchar>(i,j) = (*ij);
	}
	//std::cout<<std::endl;
}
	/*cv::Mat descriptors_return.create(desc_row_col.size(),32, CV_8U);
	std::cout<<std::setfill('-')<<std::setw(80)<<"-"<<std::endl;
	for(int i =0;i<desc_row_col.size();i++)
	{
		for(int j=0;j<32;j++)
		{
			std::cout<<std::setw(3)<<(int)descriptors_return.at<uchar>(i,j)<<" ";
		}
		std::cout<<std::endl;
	}*/
	descriptors = descriptors_return.clone();
	
	//cv::imshow("latch_desc",latch_descriptor);
	//cv::waitKey();
/*	uchar* desc=desc_vector.at(0);
	for (int ix = 0; ix < 64; ix++)
	{
		std::bitset<8>y(desc[ix]);
		std::cout<<y<<std::endl;	
	}*/
}

void LATCH(const cv::Mat image, cv::Mat &descriptors, std::vector<cv::KeyPoint> &keypoints)
{	
	/*cv::ORB orb;
	cv::OrbFeatureDetector detector(500, 1.2f, 4, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);
	//std::vector<cv::KeyPoint> keypoints;
	orb.detect(image, keypoints);*/
	cv::SiftFeatureDetector detector( 400);
	detector.detect( image, keypoints);

	//clear borders
	//std::cout<<"keypoints:"<<keypoints.size()<<"\t";
	int width = image.cols,height = image.rows;
	keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), [width, height](const cv::KeyPoint& kp) 
	{
		return kp.pt.x <=border_clear/2 || kp.pt.y<=border_clear/2 || kp.pt.x>=width - border_clear/2 || kp.pt.y>=height - border_clear/2; 
	}
	), keypoints.end());

	cv::Mat result = image.clone();

	for(int i=0;i<(int )keypoints.size();i++)
	{
		cv::Point2f center(keypoints[i].pt.x,keypoints[i].pt.y);
		cv::circle(result, center ,8, (90), 3, 8, 0);
	}
	//cv::imshow("Keypoints",result); 
	//cv::waitKey(0);	
	//std::cout<<keypoints.size();
	/*for(int i=0;i<10;i++)
	{
		std::cout<<keypoints[i].pt.x<<" "<<keypoints[i].pt.y<<std::endl;;
	}*/
	//cv::Mat descriptors;
	/*cv::KeyPoint dummy;
	dummy.pt.x = keypoints[0].pt.x; 
	dummy.pt.y = keypoints[0].pt.y;
	cv::Mat patch = image(cv::Rect(dummy.pt.x,dummy.pt.y,24,24)).clone();*/
	
	//calling the
	pixelTests64(image, keypoints, descriptors);
	
}
