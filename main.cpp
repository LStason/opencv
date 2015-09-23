#include <QCoreApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

Mat rgb_to_grayscale(Mat image)
{
    Mat gray = Mat(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Vec3b intensity = image.at<Vec3b>(i,j);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            gray.at<uchar>(i,j) = uchar(int(blue)*0.114 + int(green)*0.587 + int(red)*0.299);
        }
    }
    return gray;
}

int max_element(Mat image)
{
    int max = int(image.at<uchar>(0,0));
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (int(image.at<uchar>(i,j)) > max)
                max = int(image.at<uchar>(i,j));
        }
    }
    return max;
}

int min_element(Mat image)
{
    int min = int(image.at<uchar>(0,0));
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            if (int(image.at<uchar>(i,j)) < min)
                min = int(image.at<uchar>(i,j));
        }
    }
    return min;
}

Mat linear_correction(Mat image)
{
    Mat result = Mat(image.rows, image.cols, CV_8UC1);
    int min = min_element(image);
    int max = max_element(image);    
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            result.at<uchar>(i,j) = uchar((int(image.at<uchar>(i,j)) - min)*255 / (max - min));
        }
    }
    return result;
}

Mat gray_world(Mat image)
{
    Mat result = Mat(image.rows, image.cols, CV_8UC3);
    long long size = image.rows * image.cols;
    long long red = 0, green = 0, blue = 0;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Vec3b intensity = image.at<Vec3b>(i,j);
            blue += int(intensity.val[0]);
            green += int(intensity.val[1]);
            red += int(intensity.val[2]);
        }
    }

    blue /= size;
    green /= size;
    red /= size;
    int avg = int((red + green + blue)/3);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Vec3b intensity = image.at<Vec3b>(i,j);
            int b = int(intensity.val[0]);
            int g = int(intensity.val[1]);
            int r = int(intensity.val[2]);
            Vec3b temp(b*avg/blue, g*avg/green, r*avg/red);
            result.at<Vec3b>(i,j) = temp;
        }
    }
    return result;
}

Mat larger_image(Mat image)
{
    Mat new_image = Mat(image.rows + 2, image.cols + 2, CV_8UC1);
    new_image.at<uchar>(0, 0) = image.at<uchar>(0, 0);
    new_image.at<uchar>(0, image.cols + 1) = image.at<uchar>(0, image.cols - 1);
    new_image.at<uchar>(image.rows + 1, 0) = image.at<uchar>(image.rows - 1, 0);
    new_image.at<uchar>(image.rows + 1, image.cols + 1) = image.at<uchar>(image.rows - 1, image.cols - 1);
    for (int i = 0; i < image.rows; i++)
    {
        new_image.at<uchar>(i + 1,0) = image.at<uchar>(i,0);
        new_image.at<uchar>(i + 1, image.cols + 1) = image.at<uchar>(i, image.cols - 1);
    }
    for (int i = 0; i < image.cols; i++)
    {
        new_image.at<uchar>(0, i + 1) = image.at<uchar>(0, i);
        new_image.at<uchar>(image.rows + 1, i + 1) = image.at<uchar>(image.rows - 1, i);
    }
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            new_image.at<uchar>(i + 1, j + 1) = image.at<uchar>(i, j);
    return new_image;
}

Mat larger_image(Mat image, int filter_size)
{
    int n = filter_size/2;
    for (int i = 0; i < n; i++)
        image = larger_image(image);
    return image;
}

Mat gradient(Mat image)
{
    Mat grad = Mat(image.rows, image.cols, CV_32SC2);
    Mat new_image = larger_image(image);
    int grad_x = 0, grad_y = 0;
    for (int i = 1; i <= image.rows; i++)
    {
        for (int j = 1; j <= image.cols; j++)
        {
            int x11 = int(new_image.at<uchar>(i - 1, j - 1));
            int x12 = int(new_image.at<uchar>(i - 1, j));
            int x13 = int(new_image.at<uchar>(i - 1, j + 1));
            int x21 = int(new_image.at<uchar>(i, j - 1));
            int x23 = int(new_image.at<uchar>(i, j + 1));
            int x31 = int(new_image.at<uchar>(i + 1, j - 1));
            int x32 = int(new_image.at<uchar>(i + 1, j));
            int x33 = int(new_image.at<uchar>(i + 1, j + 1));

            grad_x = x13 + 2*x23 + x33 - x11 - 2*x21 - x31;
            grad_y = x31 + 2*x32 + x33 - x11 - 2*x12 - x13;
            grad.at<Vec2i>(i - 1, j - 1) = Vec2i(grad_x, grad_y);
        }
    }
    return grad;
}

Mat gradient_magnitudes(Mat grad)
{
    Mat grad_value = Mat(grad.rows, grad.cols, CV_32SC1);
    for (int i = 0; i < grad.rows; i++)
    {
        for (int j = 0; j < grad.cols; j++)
        {
            Vec2i v = grad.at<Vec2i>(i, j);
            int x = v.val[0];
            int y = v.val[1];
            int value = int(sqrt(x*x + y*y));
            grad_value.at<int>(i,j) = value;
        }
    }
    return grad_value;
}

Mat gradient_angles(Mat grad)
{
    Mat grad_angle = Mat(grad.rows, grad.cols, CV_32SC1);
    double theta = 0;
    for (int i = 0; i < grad.rows; i++)
    {
        for (int j = 0; j < grad.cols; j++)
        {
            Vec2i v = grad.at<Vec2i>(i, j);
            double x = double(v.val[0]);
            double y = double(v.val[1]);
            if (x == 0.0)
            {
                if (y == 0.0)
                    theta = 0.0;
                else theta = 90.0;
            }
            else theta = abs(atan2(y, x) * 180 / M_PI);
            if ((theta >= 0 && theta < 22.5)||(theta >= 157.5 && theta <= 180))
                grad_angle.at<int>(i, j) = 0;
            else if (theta >= 22.5 && theta < 67.5)
                grad_angle.at<int>(i, j) = 45;
            else if (theta >= 67.5 && theta < 112.5)
                grad_angle.at<int>(i, j) = 90;
            else
                grad_angle.at<int>(i, j) = 135;
        }
    }
    return grad_angle;
}

Mat gauss_filter(Mat image)
{
    Mat new_image = larger_image(image);
    Mat gauss_image = Mat(image.rows, image.cols, CV_8UC1);
    for (int i = 1; i <= image.rows; i++)
    {
        for (int j = 1; j <= image.cols; j++)
        {
            int x11 = int(new_image.at<uchar>(i - 1, j - 1));
            int x12 = int(new_image.at<uchar>(i - 1, j));
            int x13 = int(new_image.at<uchar>(i - 1, j + 1));
            int x21 = int(new_image.at<uchar>(i, j - 1));
            int x22 = int(new_image.at<uchar>(i, j));
            int x23 = int(new_image.at<uchar>(i, j + 1));
            int x31 = int(new_image.at<uchar>(i + 1, j - 1));
            int x32 = int(new_image.at<uchar>(i + 1, j));
            int x33 = int(new_image.at<uchar>(i + 1, j + 1));

            int value = int((x11 + 2*x12 + x13 + 2*x21 + 4*x22 + 2*x23 + x31 + 2*x32 + x33)/16);
            gauss_image.at<uchar>(i - 1, j - 1) = uchar(value);
        }
    }
    return gauss_image;
}

Mat gauss()
{
    Mat m = Mat(5, 5, CV_32SC1);
    m.at<int>(0,0) = m.at<int>(0,4) = m.at<int>(4,0) = m.at<int>(4,4) = 2;
    m.at<int>(0,1) = m.at<int>(0,3) = m.at<int>(1,0) = m.at<int>(1,4) = m.at<int>(3,0) = m.at<int>(3,4) = m.at<int>(4,1) = m.at<int>(4,3) = 4;
    m.at<int>(0,2) = m.at<int>(2,0) = m.at<int>(2,4) = m.at<int>(4,2) = 5;
    m.at<int>(1,1) = m.at<int>(1,3) = m.at<int>(3,1) = m.at<int>(3,3) = 9;
    m.at<int>(1,2) = m.at<int>(2,1) = m.at<int>(2,3) = m.at<int>(3,2) = 12;
    m.at<int>(2,2) = 15;
    return m;
}

int kernel_sum(Mat kernel)
{
    int sum = 0;
    for (int i = 0; i < kernel.rows; i++)
        for (int j = 0; j < kernel.cols; j++)
            sum += kernel.at<int>(i,j);
    return sum;
}

Mat gauss_filter2(Mat image, Mat kernel)
{
    int sum = 0;
    int k_sum = kernel_sum(kernel);
    Mat new_image = larger_image(image, kernel.rows);
    Mat gauss_image = Mat(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            sum = 0;
            for (int _i = 0; _i < kernel.rows; _i++)
            {
                for (int _j = 0; _j < kernel.cols; _j++)
                {
                    sum += int(new_image.at<uchar>(i + _i, j + _j))*kernel.at<int>(_i,_j);
                }
            }
            sum /= k_sum;            
            gauss_image.at<uchar>(i,j) = uchar(sum);
        }
    }
    return gauss_image;
}

Mat median_filter(Mat image)
{
    Mat new_image = larger_image(image);
    Mat median_image = Mat(image.rows, image.cols, CV_8UC1);
    vector<int> vec;
    for (int i = 1; i <= image.rows; i++)
    {
        for (int j = 1; j <= image.cols; j++)
        {
            vec.clear();
            vec.push_back(int(new_image.at<uchar>(i - 1, j - 1)));
            vec.push_back(int(new_image.at<uchar>(i - 1, j)));
            vec.push_back(int(new_image.at<uchar>(i - 1, j + 1)));
            vec.push_back(int(new_image.at<uchar>(i, j - 1)));
            vec.push_back(int(new_image.at<uchar>(i, j)));
            vec.push_back(int(new_image.at<uchar>(i, j + 1)));
            vec.push_back(int(new_image.at<uchar>(i + 1, j - 1)));
            vec.push_back(int(new_image.at<uchar>(i + 1, j)));
            vec.push_back(int(new_image.at<uchar>(i + 1, j + 1)));
            sort(vec.begin(), vec.end());
            median_image.at<uchar>(i - 1, j - 1) = uchar(vec[4]);
        }
    }
    return median_image;
}

Mat canny(Mat image, int low, int high)
{
    Mat gray = rgb_to_grayscale(image);
    Mat gauss_image = gauss_filter2(gray, gauss());
    Mat grad = gradient(gauss_image);
    Mat magnitudes = gradient_magnitudes(grad);
    Mat angels = gradient_angles(grad);
    Mat edges = Mat(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < edges.rows; i++)
    {
        edges.at<uchar>(i, 0) = uchar(0);
        edges.at<uchar>(i, edges.cols - 1) = uchar(0);
    }
    for (int i = 1; i < edges.cols - 1; i++)
    {
        edges.at<uchar>(0, i) = uchar(0);
        edges.at<uchar>(edges.rows - 1, i) = uchar(0);
    }
    for (int i = 1; i < image.rows - 1; i++)
    {
        for (int j = 1; j < image.cols - 1; j++)
        {
            int angel = angels.at<int>(i, j);
            int magnitude = magnitudes.at<int>(i, j);
            switch (angel)
            {
            case 0:
            {
                if (magnitude > magnitudes.at<int>(i, j - 1) && magnitude > magnitudes.at<int>(i, j + 1))
                {
                    if (magnitude > high)
                        edges.at<uchar>(i, j) = uchar(255);
                    else if (magnitude <= low)
                        edges.at<uchar>(i, j) = uchar(0);
                    else
                        edges.at<uchar>(i, j) = uchar(127);
                }
                else
                    edges.at<uchar>(i, j) = uchar(0);
                break;
            }
            case 90:
            {
                if (magnitude > magnitudes.at<int>(i - 1, j) && magnitude > magnitudes.at<int>(i + 1, j))
                {
                    if (magnitude > high)
                        edges.at<uchar>(i, j) = uchar(255);
                    else if (magnitude <= low)
                        edges.at<uchar>(i, j) = uchar(0);
                    else
                        edges.at<uchar>(i, j) = uchar(127);
                }
                else
                    edges.at<uchar>(i, j) = uchar(0);
                break;
            }
            case 45:
            {
                if (magnitude > magnitudes.at<int>(i - 1, j + 1) && magnitude > magnitudes.at<int>(i + 1, j - 1))
                {
                    if (magnitude > high)
                        edges.at<uchar>(i, j) = uchar(255);
                    else if (magnitude <= low)
                        edges.at<uchar>(i, j) = uchar(0);
                    else
                        edges.at<uchar>(i, j) = uchar(127);
                }
                else
                    edges.at<uchar>(i, j) = uchar(0);
                break;
            }
            case 135:
            {
                if (magnitude > magnitudes.at<int>(i - 1, j - 1) && magnitude > magnitudes.at<int>(i + 1, j + 1))
                {
                    if (magnitude > high)
                        edges.at<uchar>(i, j) = uchar(255);
                    else if (magnitude <= low)
                        edges.at<uchar>(i, j) = uchar(0);
                    else
                        edges.at<uchar>(i, j) = uchar(127);
                }
                else
                    edges.at<uchar>(i, j) = uchar(0);
                break;
            }
            default:
                break;
            }
        }
    }
    for (int i = 1; i < edges.rows - 1; i++)
    {
        for (int j = 1; j < edges.cols - 1; j++)
        {
            if (int(edges.at<uchar>(i, j)) == 127)
            {

                if (int(edges.at<uchar>(i - 1, j - 1)) == 255 || int(edges.at<uchar>(i - 1, j)) == 255 || int(edges.at<uchar>(i - 1, j + 1)) == 255 ||
                        int(edges.at<uchar>(i, j - 1)) == 255 || int(edges.at<uchar>(i, j + 1)) == 255 || int(edges.at<uchar>(i + 1, j - 1)) == 255 ||
                        int(edges.at<uchar>(i + 1, j)) == 255 || int(edges.at<uchar>(i + 1, j + 1)) == 255)
                {
                    edges.at<uchar>(i, j) = uchar(255);

                }
                else
                {
                    edges.at<uchar>(i, j) = uchar(0);
                }
            }
        }
    }
    return edges;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    Mat image;
    string path = "/home/stas/stas.jpg";
    image = imread(path);

    if(! image.data )
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", image );

    //Mat gray_image;
    //cvtColor(image, gray_image, CV_BGR2GRAY);
    //namedWindow("Gray image1", CV_WINDOW_AUTOSIZE);
    //imshow("Gray image1", gray_image);
    Mat gray = rgb_to_grayscale(image);
    //namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
    //imshow("Gray image", gray);

    //Mat Gauss = gauss_filter(gray);
    //namedWindow("Gauss image", CV_WINDOW_AUTOSIZE);
    //mshow("Gauss image", Gauss);

    //Mat median = median_filter(gray);
    //namedWindow("Median image", CV_WINDOW_AUTOSIZE);
    //imshow("Median image", median);

    Mat kernel = gauss();
    Mat gauss_image = gauss_filter2(gray, kernel);
    namedWindow("Gauss image", CV_WINDOW_AUTOSIZE);
    imshow("Gauss image", gauss_image);

    //Mat correct_gray = linear_correction(bad);
    //namedWindow("Gray correction image", CV_WINDOW_AUTOSIZE);
    //imshow("Gray correction image", correct_gray);
    //Mat gray_world_image = gray_world(image);
    //namedWindow("Gray world image", CV_WINDOW_AUTOSIZE);
    //imshow("Gray world image", gray_world_image);
    //Mat grad = gradient(gray);
    //Mat grad_value = gradient_magnitudes(grad);
    //Mat angels = gradient_angles(grad);
    //namedWindow("Gradient image", CV_WINDOW_AUTOSIZE);
    //imshow("Gradient image", grad_value);

    //Mat edges = canny(image, 80, 100);
    Mat edges2 = canny(image, 40, 90);
    //namedWindow("Edges image", CV_WINDOW_AUTOSIZE);
    //imshow("Edges image", edges);
    namedWindow("Edges image2", CV_WINDOW_AUTOSIZE);
    imshow("Edges image2", edges2);

    Mat dst;
    Canny(image, dst, 50, 150, 3);
    namedWindow("Opencv Edges image", CV_WINDOW_AUTOSIZE);
    imshow("Opencv Edges image", dst);
    return a.exec();
}

