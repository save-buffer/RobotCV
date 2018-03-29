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

enum ActivationType
{
    ActivationRelu,
};

struct layer
{
    LayerType type;
    ActivationType activation;
    vector<vector<Mat> > filters; //If it's fully connected, there'll just be one entry
    vector<Mat> bias;
};

void relu(Mat &m)
{
    m.setTo(0, m < 0);
}

Mat relu_grad(Mat activation)
{
    return(activation > 0);
}

void activation_function(Mat &m, ActivationType activation)
{
    switch(activation)
    {
    case ActivationRelu:
	relu(m);
    }
}

Mat grad_activation_function(Mat &m, ActivationType activation)
{
    switch(activation)
    {
    case ActivationRelu:
	return(relu_grad(m));
    }
}

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

pair<vector<Mat>, vector<Mat> > conv_layer(const vector<Mat> &inputs, const layer &l)
{
    auto zs = convolve(inputs, l.filters);
    vector<Mat> as;
    //TODO(sasha): assert that result.size() == l.bias.size()
    for(int i = 0; i < zs.size(); i++)
    {
	Mat a = zs[i].clone();
	activation_function(a, l.activation);
	a += l.bias[i];
	as.push_back(a);
    }
    return(make_pair(as, zs));
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

vector<Mat> fully_connected(const vector<Mat> &inputs, const Mat &weights)
{
    Mat unrolled;
    vector<Mat> result;
    for(const Mat &i : inputs)
	unrolled.push_back(i.reshape(1));

    unrolled = unrolled * weights;
    result.push_back(unrolled);
    return(result);
}

pair<vector<Mat>, vector<Mat> > fully_connected_layer(const vector<Mat> &inputs, const layer &l)
{
    auto zs = fully_connected(inputs, l.filters[0][0]);
    vector<Mat> as;
    Mat a = zs[0].clone();
    activation_function(a, l.activation);
    a += l.bias[0];
    as.push_back(a);  

    return(make_pair(as, zs));
}

vector<Mat> fully_connected_delta(const vector<Mat> &prev_error, const layer &prev_layer, const vector<Mat> &z, const layer &l)
{
    vector<Mat> result;
    Mat delta = prev_layer.weights[0].t() * prev_err[0];
    result.push_back(delta);
    return(result);
}

pair<vector<Mat>, vector<Mat> > execute_layer(const vector<Mat> &inputs, const layer &l)
{
    switch(l.type)
    {
    case LayerFullyConnected:
	return(fully_connected_layer(inputs, l));
    case LayerConv:
	return(conv_layer(inputs, l));
    }
    return(make_pair(vector<Mat>(), vector<Mat>()));
}

vector<Mat> layer_grad(const vector<Mat> &prev_error, const layer &prev_layer, const vector<Mat> &z, const layer &l)
{
    switch(prev_layer.type)
    {
    case LayerFullyConnected:
	return(fully_connected_grad(prev_error, prev_layer, z, l));
	
    }
}

vector<Mat> layer_delta(const vector<Mat> &prev_error, const layer &l)
{
    switch(l.type)
    {
    case LayerFullyConnected:
	return(fully_connected_grad(prev_error, l));
	
    }

}

pair<vector<vector<Mat> >, vector<vector<Mat> > > for_prop(const vector<Mat> &input, const vector<layer> &layers)
{
    vector<vector<Mat> > activations;
    vector<vector<Mat> > zs;
    activations.push_back(input);
    for(int i = 0; i < layers.size(); i++)
    {
	auto act = execute_layer(activations[i], layers[i]);
	activations.push_back(act.first);
	zs.push_back(act.second);
    }
    return(make_pair(activations, zs));
}

void back_prop(const vector<Mat> &input, vector<layer> &layers, vector<Mat> expected_output)
{
    auto prop = for_prop(input, layers);
    auto activations = prop.first;
    auto zs = prop.second;

    vector<vector<Mat> > errors(layers.size());

    auto output = activations.back();
    for(int i = 0; i < expected_output.size(); i++)
    {
	Mat delta = expected_output[i] - output[i];
	Mat final_act_grad = grad_activation_function(zs.back(), layers.back().activation);
	errors.back().push_back(delta.mul(final_act_grad));
    }

    for(int i = errors.size() - 2; i >= 0; i--)
    {
	
    }

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
//	    grads[i] = 
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
