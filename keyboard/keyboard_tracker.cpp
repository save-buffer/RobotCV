#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include "neural_net.cpp"

using namespace cv;
//For compatibility with opencv2
namespace cv
{
    using std::vector;
}

//./keyboard_tracker /mnt/c/Users/Sasha/Downloads/keyboard.png
Mat fft(Mat gray)
{
    Mat fft;
    gray.convertTo(fft, CV_32F);
    dft(fft, fft, DFT_SCALE | DFT_COMPLEX_OUTPUT);
    return(fft);
    
    Mat padded;
    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    copyMakeBorder(gray, padded, m - gray.rows, 0, n - gray.cols, 0, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complex;
    merge(planes, 2, complex);
    dft(complex, complex);
    split(complex, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat mag = planes[0];
    mag += Scalar::all(1);
    log(mag, mag);
    mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
    normalize(mag, mag, 0, 1, CV_MINMAX);
    return(mag);
}

Mat ifft(Mat fft)
{
    Mat ifft;
    dft(fft, ifft, DFT_INVERSE | DFT_REAL_OUTPUT);
    ifft.convertTo(ifft, CV_8U);
    return(ifft);
}

Mat switch_quadrants(Mat src)
{
    Mat mag;
    src.copyTo(mag);
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    Mat q0(mag, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(mag, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(mag, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(mag, Rect(cx, cy, cx, cy)); // Bottom-Right
    
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
        
    return(mag);
}

Mat high_pass(Mat src)
{
#define HIGHPASS_THRESH 0.0f
    Mat fft_img = fft(src);
    threshold(fft_img, fft_img, HIGHPASS_THRESH, 1.0f, THRESH_TOZERO);
    Mat orig = ifft(fft_img);
    return(orig);
#undef HIGHPASS_THRESH
}

void laplacian_keyboard_identifier(Mat src)
{
    Mat gray;    
    cvtColor(src, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray, gray, Size(3, 3), 3);
    Mat orig;
    
    Laplacian(gray, orig, CV_8U, 13);

    Mat se1 = getStructuringElement(MORPH_RECT, Size(1, 1));
    Mat se3 = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat se5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat se7 = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat se9 = getStructuringElement(MORPH_RECT, Size(9, 9));
    Mat se11 = getStructuringElement(MORPH_RECT, Size(11, 11));
    Mat se13 = getStructuringElement(MORPH_RECT, Size(13, 13));

    morphologyEx(orig, orig, MORPH_ERODE, se5);
    morphologyEx(orig, orig, MORPH_DILATE, se5);

    Mat key_image;
    int components = connectedComponents(orig, key_image);
    printf("found %d components\n", components);
    for(int i = 1; i < components; i++)
    {
#define HW_THRESH 2.0f
	Mat component_i;
	inRange(key_image, Scalar(i), Scalar(i), component_i);
	Rect r = boundingRect(component_i);
	if((float)r.size().height / (float)r.size().width > HW_THRESH)
	{
	    component_i /= i;
	    component_i *= 255;
	    orig -= component_i;
	}
#undef HW_THRESH
    }
//    morphologyEx(orig, orig, MORPH_DILATE, se5);
    morphologyEx(orig, orig, MORPH_ERODE, se5);
    morphologyEx(orig, orig, MORPH_DILATE, se13);
    
    components = connectedComponents(orig, key_image);
    printf("found %d components\n", components);

    Mat color_orig;
    for(int i = 1; i < components; i++)
    {
	Mat component_i;
	inRange(key_image, Scalar(i), Scalar(i), component_i);
	Rect r = boundingRect(component_i);
	rectangle(src, r.tl(), r.br(), Scalar(0, 0, 255), 1);
    }
    namedWindow("Keyboard Identifier");
    namedWindow("Morphology");
    imshow("Keyboard Identifier", src);
    imshow("Morphology", orig);
}

bool filter_rectangles(Rect r, Size size)
{
    float width_ratio = (float)r.width / (float)(size.width);
    float height_ratio = (float)r.height / (float)(size.height);

    return(0.01f < width_ratio && width_ratio < 0.5f &&
	   0.05 < height_ratio && height_ratio < 0.2f);
}    

//TODO(sasha): optimize?
void join_overlapping_rectangles(vector<Rect> &rects)
{
    for(int i = 0; i < rects.size() - 1; i++)
    {
	for(int j = i + 1; j < rects.size(); j++)
	{
	    Rect intersection = rects[i] & rects[j];
	    Rect minimum_enclosing = rects[i] | rects[j];
	    if(intersection.area() > 0)
	    {
		rects[i] = minimum_enclosing;
		rects.erase(rects.begin() + j--);
	    }
	}
    }
}

Mat src;
Mat gray_orig;
Mat gray;
int thresh = 30;
int max_thresh = 255;

int blur_std = 0;
int max_blur_std = 20;

int blur_size = 1;
int max_blur_size = 10;

void contour_keyboard_tracker()
{
    Mat edges;
    Canny(gray, edges, thresh, thresh * 3, 3);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    
    findContours(edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

    Mat contour_img = Mat::zeros(edges.size(), CV_8UC3); 
    for(int i = 0; i < contours.size(); i++)
    {
	drawContours(contour_img, contours, i, Scalar(0, 0, 255), 1, 8, hierarchy, 0, Point());
    }

    Mat gray_contours;
    cvtColor(contour_img, gray_contours, COLOR_BGR2GRAY);

    //we map 800x300 to 1
    float se_proportion = src.cols / 800.0f;

    int _1 = 1 * se_proportion;
    int _3 = 3 * se_proportion;
    int _5 = 5 * se_proportion;
    int _7 = 7 * se_proportion;
    int _9 = 9 * se_proportion;
    int _11 = 11 * se_proportion;
    int _13 = 13 * se_proportion;
    
    Mat se1 = getStructuringElement(MORPH_RECT, Size(_1, _1));
    Mat se3 = getStructuringElement(MORPH_RECT, Size(_3, _3));
    Mat se5 = getStructuringElement(MORPH_RECT, Size(_5, _5));
    Mat se7 = getStructuringElement(MORPH_RECT, Size(_7, _7));
    Mat se9 = getStructuringElement(MORPH_RECT, Size(_9, _9));
    Mat se11 = getStructuringElement(MORPH_RECT, Size(_11, _11));
    Mat se13 = getStructuringElement(MORPH_RECT, Size(_13, _13));
    
    morphologyEx(gray_contours, gray_contours, MORPH_DILATE, se5);
//    morphologyEx(gray_contours, gray_contours, MORPH_ERODE, se3);

    gray_contours = Scalar::all(255) - gray_contours;
    inRange(gray_contours, Scalar(255), Scalar(255), gray_contours);
    morphologyEx(gray_contours, gray_contours, MORPH_DILATE, se5);
    morphologyEx(gray_contours, gray_contours, MORPH_ERODE, se3);


    Mat key_image;
    int components = connectedComponents(gray_contours, key_image);
    printf("found %d components\n", components);
    for(int i = 1; i < components; i++)
    {
#define HW_THRESH 2.0f
	Mat component_i;
	inRange(key_image, Scalar(i), Scalar(i), component_i);
	Rect r = boundingRect(component_i);
	if((float)r.size().height / (float)r.size().width > HW_THRESH)
	{
	    component_i /= i;
	    component_i *= 255;
	    gray_contours -= component_i;
	}
#undef HW_THRESH
    }

    morphologyEx(gray_contours, gray_contours, MORPH_ERODE, se3);
    morphologyEx(gray_contours, gray_contours, MORPH_DILATE, se5);
    morphologyEx(gray_contours, gray_contours, MORPH_ERODE, se5);
    morphologyEx(gray_contours, gray_contours, MORPH_DILATE, se3);
    
    components = connectedComponents(gray_contours, key_image);
    printf("found %d components\n", components);

    vector<Rect> keys;
    Mat color_orig(src);
    for(int i = 1; i < components; i++)
    {
	Mat component_i;
	inRange(key_image, Scalar(i), Scalar(i), component_i);
	Rect r = boundingRect(component_i);
	if(filter_rectangles(r, src.size()))
	{
	    keys.push_back(r);
	}
    }
    join_overlapping_rectangles(keys);
    for(Rect &r : keys)
    {
	rectangle(color_orig, r.tl(), r.br(), Scalar(0, 0, 255), 1);
    }
    
    namedWindow("Contours");
    namedWindow("Gray");
    namedWindow("Output");
    imshow("Contours", contour_img);
    imshow("Gray", gray_contours);
    imshow("Output", color_orig);
}

void keyboard_identifier(Mat src)
{
    contour_keyboard_tracker();
}

void thresh_callback(int, void *)
{
    contour_keyboard_tracker();
}

void blur_callback(int, void *)
{
    gray_orig.copyTo(gray);
    int real_blur_size = 2 * blur_size + 1;
    blur(gray, gray, Size(real_blur_size, real_blur_size));
//    GaussianBlur(gray, gray, Size(real_blur_size, real_blur_size), blur_std);
    contour_keyboard_tracker();
}

int main(int argc, char** argv)
{
#if 0
    /// Read the image
    src = imread(argv[1], 1);
/*    float aspect_ratio = (float)src.rows / (float)src.cols;
      int target_x = 800;
      int target_y = (int)(target_x * aspect_ratio);
      resize(src, src, Size(target_x, target_y), 0, 0, CV_INTER_AREA);*/
    if(!src.data)
	return(-1);

    cvtColor(src, gray_orig, COLOR_BGR2GRAY);

    namedWindow("Source");
    imshow("Source", src);
    createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
    createTrackbar(" Blur std:", "Source", &blur_std, max_blur_std, blur_callback);
    createTrackbar(" Blur size:", "Source", &blur_size, max_blur_size, blur_callback);
    blur_callback(0, 0);
		 
//    keyboard_identifier(src);
    
    waitKey(0);
#else
/*    auto data_pair = read_data("data.yml");
    auto data = data_pair.first;
    auto filenames = data_pair.second;
    for(const auto &pair : data)
	cout << pair.first << endl;
    for(int i = 0; i < 10; i++)
    cout << filenames[i] << endl;*/
    cnn_model();
#endif
    return(0);
}
