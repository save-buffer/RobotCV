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

TF_Graph *cnn_model(const vector<Mat> &features, Mat labels)
{
    TF_Graph *graph = TF_NewGraph();
    TF_Status *s = TF_NewStatus();
    TF_Operation *feed = Placeholder(graph, s);
    
}
