#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
 
#define WIDTH   50
#define HEIGHT  25
 
class depth_estimater{
public:
    depth_estimater();
    ~depth_estimater();
    void rgbImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void depthImageCallback(const sensor_msgs::ImageConstPtr& msg);
 
private:
    ros::NodeHandle nh;
    ros::Subscriber sub_rgb, sub_depth;
};
 
depth_estimater::depth_estimater(){
    sub_rgb = nh.subscribe<sensor_msgs::Image>("/camera/rgb/image_color", 1, &depth_estimater::rgbImageCallback, this);
    sub_depth = nh.subscribe<sensor_msgs::Image>("/camera/depth/image", 1, &depth_estimater::depthImageCallback, this);
}
 
depth_estimater::~depth_estimater(){
}
 
void depth_estimater::rgbImageCallback(const sensor_msgs::ImageConstPtr& msg){
 
    int i, j;
    int x1, x2, y1, y2;
    int width = WIDTH;
    int height = HEIGHT;
    cv_bridge::CvImagePtr cv_ptr;
 
    try{
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }catch (cv_bridge::Exception& ex){
        ROS_ERROR("error");
        exit(-1);
    }
 
    cv::Mat &mat = cv_ptr->image;
 
    x1 = int(mat.cols / 2) - width;
    x2 = int(mat.cols / 2) + width;
    y1 = int(mat.rows / 2) - height;
    y2 = int(mat.rows / 2) + height;
 
    for(i = y1; i < y2; i++){
        for(j = x1; j < x2; j++){
            // 0 : blue, 1 : green, 2 : red.
            mat.data[i * mat.step + j * mat.elemSize() + 0] = 0;
            mat.data[i * mat.step + j * mat.elemSize() + 1] = 0;
            //mat.data[i * mat.step + j * mat.elemSize() + 2] = 0;
        }
    }
 
    cv::rectangle(cv_ptr->image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3, 4);
    cv::imshow("RGB image", cv_ptr->image);
    cv::waitKey(10);
}
 
void depth_estimater::depthImageCallback(const sensor_msgs::ImageConstPtr& msg){
 
    int x1, x2, y1, y2;
    int i, j, k;
    int width = WIDTH;
    int height = HEIGHT;
    double sum = 0.0;
    double ave;
    cv_bridge::CvImagePtr cv_ptr;
 
    try{
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }catch (cv_bridge::Exception& ex){
        ROS_ERROR("error");
        exit(-1);
    }
 
    cv::Mat depth(cv_ptr->image.rows, cv_ptr->image.cols, CV_32FC1);
    cv::Mat img(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC1);
 
    x1 = int(depth.cols / 2) - width;
    x2 = int(depth.cols / 2) + width;
    y1 = int(depth.rows / 2) - height;
    y2 = int(depth.rows / 2) + height;
 
    for(i = 0; i < cv_ptr->image.rows;i++){
        float* Dimage = cv_ptr->image.ptr<float>(i);
        float* Iimage = depth.ptr<float>(i);
        char* Ivimage = img.ptr<char>(i);
        for(j = 0 ; j < cv_ptr->image.cols; j++){
            if(Dimage[j] > 0.0){
                Iimage[j] = Dimage[j];
                Ivimage[j] = (char)(255*(Dimage[j]/5.5));
            }else{
            }
 
            if(i > y1 && i < y2){
                if(j > x1 && j < x2){
                    if(Dimage[j] > 0.0){
                        sum += Dimage[j];
                    }
                }
            }
        }
    }
 
    ave = sum / ((width * 2) * (height * 2));
    ROS_INFO("depth : %f [m]", ave);
 
    cv::imshow("DEPTH image", img);
    cv::waitKey(10);
}
 
int main(int argc, char **argv){
    ros::init(argc, argv, "depth_estimater");
 
    depth_estimater depth_estimater;
 
    ros::spin();
    return 0;
}