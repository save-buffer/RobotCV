#include <tensorflow/c/c_api.h>
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

TF_Output placeholder(TF_Graph *graph, TF_Status *s, const char *name,
		      TF_DataType dtype)
{
    TF_Output output;
    TF_OperationDescription *desc = TF_NewOperation(graph, "Placeholder", name);
    TF_SetAttrType(desc, "dtype", dtype);
    output.oper = TF_FinishOperation(desc, s);
    output.index = 0;
    return(output);
}

TF_Output constant(TF_Graph *graph, TF_Status *s, const char *name,
		   TF_DataType dtype, const void *values, size_t value_size)
{
    TF_OperationDescription *desc = TF_NewOperation(graph, "Const", name);
    TF_SetAttrType(desc, "dtype", dtype);
    TF_SetAttrType(	
}

TF_Output reshape(TF_Graph *graph, TF_Status *s, const char *name,
		  TF_Output input, const int64_t *dims, int num_dims)
{
    TF_OperationDescription *desc = TF_NewOperation(graph, "Reshape", name);
    TF_AddInput(desc, input);
    
//    TF_SetAttrShape(desc, "shape", dims, num_dims);
    TF_Output output;
    output.oper = TF_FinishOperation(desc, s);
    output.index = 0;
    return(output);
}

TF_Output conv2d(TF_Graph *graph, TF_Status *s, const char *name,
		     TF_Output input, int64_t filters, const int64_t *kernel_size,
		     const int64_t *strides = nullptr, const char *padding = "valid")
{
    TF_OperationDescription *desc = TF_NewOperation(graph, "Conv2D", name);
    TF_AddInput(desc, input);
    TF_SetAttrInt(desc, "filters", filters);
    TF_SetAttrString(desc, "kernel_size", kernel_size, sizeof(int64_t) * 2);
    int64_t default_strides[] = { 1, 1 };
    if(strides == nullptr)
	strides = default_strides;
    TF_SetAttrString(desc, "strides", strides, sizeof(int64_t) * 2);
    TF_SetAttrString(desc, "padding", padding, strlen(padding));
    TF_Output output;
    output.oper = TF_FinishOperation(desc, s);
    output.index = 0;
    return(output);
}

TF_Output relu(TF_Graph *graph, TF_Status *s, const char *name,
		   TF_Output input, const char *function_name)
{
    TF_OperationDescription *desc = TF_NewOperation(graph, function_name, name);
    TF_AddInput(desc, input);
    TF_Output output;
    output.oper = TF_FinishOperation(desc, s);
    output.index = 0;
    return(output);
}

TF_Tensor *uint8_tensor(const int64_t *dims, int num_dims, uint64_t len)
{
    int64_t num_bytes = 1;
    for(int i = 0; i < num_dims; i++)
	num_bytes *= dims[i];
    
    return(TF_AllocateTensor(TF_UINT8, dims, num_dims, sizeof(uint8_t) * len));
}

TF_Graph *cnn_model()
{
    TF_Graph *graph = TF_NewGraph();
    TF_Status *s = TF_NewStatus();
    cout << "Status: " << TF_GetCode(s) << endl;
    TF_Output input = placeholder(graph, s, "input", TF_UINT8);
    cout << "Status: " << TF_GetCode(s) << endl;
    int64_t resize_dim[] = { -1, 28, 28, 3 };
    TF_Output resized_input = reshape(graph, s, "resize", input, resize_dim, 4);
    cout << "Status: " << TF_GetCode(s) << endl;
    int64_t kernel_size[] = { 5, 5 };
    TF_Output layer1 = conv2d(graph, s, "layer1", resized_input, 32, kernel_size, nullptr, "same");
    cout << "Status: " << TF_GetCode(s) << endl;
    return(graph);
}

TF_Tensor *fill_input_tensor(const vector<Mat> &training_set)
{
    int64_t input_dims[3];
    input_dims[0] = training_set[0].rows, input_dims[1] = training_set[0].cols, input_dims[2] = 3;
    TF_Tensor *input = uint8_tensor(input_dims, 3, training_set.size());

    int64_t bytes_per_channel = training_set[0].rows * training_set[0].cols;
    uint8_t *data = (uint8_t *)TF_TensorData(input);
    for(int i = 0; i < training_set.size(); i++)
    {
	Mat bgr[3];
	split(training_set[i], bgr);
	for(int j = 0; j < 3; j++)
	{
	    if(bgr[j].isContinuous())
	    {
		memcpy(data, bgr[j].ptr(0), bgr[j].rows * bgr[j].cols);
		data += bytes_per_channel;
	    }
	    else
	    {
		for(int k = 0; k < bgr[j].cols; k++)
		{
		    memcpy(data, bgr[j].ptr(k), bgr[j].rows);
		    data += bgr[j].rows;
		}
	    }
	}
    }
    return(input);
}
