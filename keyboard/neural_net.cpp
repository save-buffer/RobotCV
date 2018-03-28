#include <iostream>
#include <fstream>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <math.h>
#include <string>
#include <utility>

using namespace std;
using namespace cv;

vector<float> X { 5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 6.2, 3.4, 5.4, 2.3, 5.9, 3.0, 5.1, 1.8 };

vector<float> y { 0, 0, 1, 1 };

vector<float> W { 0.5, 0.5, 0.5, 0.5 };

vector<float> sigmoid_d(const vector<float> &m1)
{
    /*
      Returns the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
      where f(x) is sigmoid function.
      Input: m1, a vector.
      Output: x(1 - x) for every element of the input matrix m1.
    */

    const uint64_t VECTOR_SIZE = m1.size();
    vector<float> output(VECTOR_SIZE);

    for(uint64_t i = 0; i < VECTOR_SIZE; i++)
	output[i] = m1[i] * (1 - m1[i]);

    return(output);
}

vector<float> sigmoid(const vector<float> &m1)
{
    /*
      Returns the value of the sigmoid function f(x) = 1/(1 + e^-x).
      Input: m1, a vector.
      Output: 1/(1 + e^-x) for every element of the input matrix m1.
    */
    const uint64_t VECTOR_SIZE = m1.size();
    vector<float> output (VECTOR_SIZE);

    for(uint64_t i = 0; i < VECTOR_SIZE; i++) 
	output[i] = 1 / (1 + exp(-m1[i]));
    
    return(output);
}

vector<float> operator+(const vector<float> &m1, const vector<float> &m2)
{
    /*
      Returns the elementwise sum of two vectors.
      Inputs:
      m1: a vector
      m2: a vector
      Output: a vector, sum of the vectors m1 and m2.
    */

    const uint64_t VECTOR_SIZE = m1.size();
    vector<float> sum(VECTOR_SIZE);

    for (uint64_t i = 0; i < VECTOR_SIZE; ++i)
	sum[i] = m1[i] + m2[i];

    return(sum);
}

vector<float> operator-(const vector<float> &m1, const vector<float> &m2)
{
    /*
      Returns the difference between two vectors.
      Inputs:
      m1: vector
      m2: vector
      Output: vector, m1 - m2, difference between two vectors m1 and m2.
    */

    const uint64_t VECTOR_SIZE = m1.size();
    vector<float> difference(VECTOR_SIZE);

    for (uint64_t i = 0; i < VECTOR_SIZE; i++)
	difference[i] = m1[i] - m2[i];

    return(difference);
}

vector<float> operator*(const vector<float> &m1, const vector<float> &m2){

    /*
      Returns the product of two vectors (elementwise multiplication).
      Inputs:
      m1: vector
      m2: vector
      Output: vector, m1 * m2, product of two vectors m1 and m2
    */
    const uint64_t VECTOR_SIZE = m1.size();
    vector <float> product(VECTOR_SIZE);

    for (uint64_t i = 0; i < VECTOR_SIZE; i++)
	product[i] = m1[i] * m2[i];
    
    return product;
}

vector<float> transpose(float *m, const int C, const int R)
{
    /*
      Returns a transpose matrix of input matrix.
      Inputs:
      m: vector, input matrix
      C: int, number of columns in the input matrix
      R: int, number of rows in the input matrix
      Output: vector, transpose matrix mT of input matrix m
    */

    vector<float> mT(C*R);

    for(uint64_t n = 0; n < C*R; n++)
    {
	uint64_t i = n/C;
	uint64_t j = n%C;
	mT[n] = m[R*j + i];
    }
    
    return mT;
}

vector<float> dot(const vector<float> &m1, const vector<float> &m2, const int m1_rows, const int m1_columns, const int m2_columns)
{
    /*
      Returns the product of two matrices: m1 x m2.
      Inputs:
      m1: vector, left matrix of size m1_rows x m1_columns
      m2: vector, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
      must be equal to the number of the columns in the left one)
      m1_rows: int, number of rows in the left matrix m1
      m1_columns: int, number of columns in the left matrix m1
      m2_columns: int, number of columns in the right matrix m2
      Output: vector, m1 * m2, product of two vectors m1 and m2, a matrix of size m1_rows x m2_columns
    */

    vector <float> output (m1_rows*m2_columns);

    for(int row = 0; row < m1_rows; row++)
    {
	for(int col = 0; col < m2_columns; col++)
	{
	    output[row * m2_columns + col] = 0.0f;
	    for(int k = 0; k < m1_columns; k++)
		output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
	}
    }

    return output;
}

void print(const vector <float>& m, int n_rows, int n_columns)
{
    /*
      "Couts" the input vector as n_rows x n_columns matrix.
      Inputs:
      m: vector, matrix of size n_rows x n_columns
      n_rows: int, number of rows in the left matrix m1
      n_columns: int, number of columns in the left matrix m1
    */

    for(int i = 0; i < n_rows; i++)
    {
	for(int j = 0; j < n_columns; j++)
	    cout << m[ i * n_columns + j ] << " ";
	
	cout << '\n';
    }
    cout << endl;
}

pair<unordered_map<string, Mat>, vector<string> > read_data(string filename)
{
    unordered_map<string, Mat> matrices;
    vector<string> filenames;

    FileStorage f(filename, FileStorage::READ);
    f["ALLlabels"] >> matrices["ALLlabels"];
    f["TRNind"] >> matrices["TRNind"];
    f["TSTind"] >> matrices["TSTind"];
    f["VALind"] >> matrices["VALind"];

    Mat files;
    f["ALLnames"] >> files;    
    for(int i = 0; i < files.rows; i++)
    {
	string name = "English/";
	Mat n = files.row(i);
	//name.cols had better be 34
	for(int j = 0; j < n.cols; j++)
	    name += (char)n.at<float>(j);

	filenames.push_back(name);	
    }
    return(make_pair(matrices, filenames));
}

enum LayerType
{
    LayerConv,
    LayerFullyConnected,
};

struct layer
{
    LayerType type;
    vector<vector<Mat> > filters; //If it's fully connected, there'll just be one entry
};

vector<Mat> convolve(const vector<Mat> &inputs, const vector<vector<Mat> > &filters)
{
    vector<Mat> result;
    for(int i = 0; i < inputs.size(); i++)
    {
	Mat out;
	for(const Mat &j : filters[i])
	{
	    Mat conv;
	    filter2D(inputs[i], conv, -1, j);
	    out += conv;
	}
	result.push_back(out);
    }
    return(result);
}

vector<vector<Mat> > conv_grad(const vector<Mat> &activation, const vector<vector<Mat> > &filters)
{
    vector<vector<Mat> > result;
    for(const auto &act : activation)
    {
	for(const auto &filter : filters)
	{
	    vector<Mat> grads;
	    for(const auto &filter2d : filter)
	    {
		Mat flipped_filter;
		rotate(filter2d, flipped_filter, ROTATE_180);
		Mat grad;
		filter2D(act, grad, -1, flipped_filter);
		grads.push_back(grad);
	    }
	    result.push_back(grads);
	}
    }
    return(result);
}

vector<Mat> fully_connected(const vector<Mat> &inputs, const Mat weights)
{
    Mat unrolled;
    vector<Mat> result;
    for(const Mat &i : inputs)
	unrolled.push_back(i.reshape(1));

    unrolled = unrolled * weights;
    unrolled.setTo(0, unrolled < 0); //relu
    result.push_back(unrolled);
    return(result);
}

Mat relu_grad(Mat activation)
{
    return(activation > 0);
}

vector<vector<Mat> > for_prop(const vector<Mat> &input, const vector<layer> &layers)
{
    vector<vector<Mat> > activations;
    activations.push_back(input);
    for(int i = 0; i < layers.size(); i++)
    {
	vector<Mat> act;
	switch(layers[i].type)
	{
	case LayerConv:
	    act = convolve(activations[i], layers[i].filters); //activations[i] is the input into layers[i]
	    break;
	case LayerFullyConnected:
	    act = fully_connected(activations[i], layers[i].filters[0][0]);
	    break;
	}
	activations.push_back(act);
    }
    return(activations);
}

void back_prop(const vector<Mat> &input, vector<layer> &layers, int actual_class, int num_classes)
{
    vector<vector<Mat> > activations = for_prop(input, layers);
    Mat y_actual(num_classes, 1, CV_32FC1, Scalar(0));
    y_actual.at<float>(actual_class) = 1;

    vector<vector<vector<Mat> > > grads(layers.size());
    Mat error = y_actual - activations.back()[0];
    
    {
	switch(layers.back().type)
	{
	case LayerConv:
	{
	    vector<vector<Mat> > grad = conv_grad(activations.back(), layers.back().filters);
	    grads.back() = grad;
	} break;
	case LayerFullyConnected:
	{
	    vector<Mat> grad;
	    grad.push_back(error * relu_grad(activations.back()[0]));
	    vector<vector<Mat> > a;
	    a.push_back(grad);
	    grads.back() = a;
	} break;
	}

    }
    
    for(int i = layers.size() - 2; i >= 0; i--)
    {
	const layer &l = layers[i];
	switch(l.type)
	{
	case LayerConv:
	{
	    vector<vector<Mat> > grad = conv_grad(activations[i], layers[i].filters);
	    grads[i] = 
	} break;
	case LayerFullyConnected:	    
	    break;
	}
    }
}

vector<vector<Mat> > random_conv_layer_initialization(int filter_size, int num_inputs, int num_filters)
{
    vector<vector<Mat> > result;
    for(int i = 0; i < num_filters; i++)
    {
	vector<Mat> filter3d;
	for(int j = 0; j < num_inputs; j++)
	{
	    Mat filter(filter_size, filter_size, CV_32FC1);
	    randu(filter, -0.001, 0.001);
	    filter3d.push_back(filter);
	}
	result.push_back(filter3d);
    }
    return(result);
}

void train_nn()
{
    auto data_pair = read_data("data.yml");
    auto data = data_pair.first;
    auto filenames = data_pair.second;

    for(int i = 0; i < filenames.size(); i++)
    {
	Mat _img = imread(filenames[i]);
	Mat img;
	_img.convertTo(img, CV_32FC3);
	Mat bgr[3];
	split(img, bgr);
    }
}
