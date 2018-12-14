//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>

#include <fstream>

#include<opencv2\opencv.hpp>


int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
using std::cout;
using std::endl;
void load_mnist_data(std::string filename, float*& data, unsigned& num, unsigned& dim) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		/////////////////////////////////////////
		//number_of_images = 1000;
		dim = n_rows*n_cols;
		num = number_of_images;
		data = new float[(size_t)num * (size_t)dim];

		for (int i = 0; i < number_of_images; i++)
		{
			//cv::Mat dst(28, 28, CV_8UC1);
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					//tp.push_back(image);
					data[dim * i + r*n_cols + c] = static_cast<float>(image);
					//dst.at<uchar>(r, c) = image;
				}
			}
			//cv::imshow("img", dst);
			//cv::waitKey(0);
		}
	}
}

void read_Mnist_Label(std::string filename, std::vector<double>&labels)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		for (int i = 0; i < number_of_images; i++)
		{
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back((double)label);
		}
	}
}

void load_data(char* filename, float*& data, unsigned& num,
	unsigned& dim) {  // load data with sift10K pattern
	std::ifstream in(filename, std::ios::binary);
	if (!in.is_open()) {
		std::cout << "open file error" << std::endl;
		exit(-1);
	}
	in.read((char*)&dim, 4);
	in.seekg(0, std::ios::end);
	std::ios::pos_type ss = in.tellg();
	size_t fsize = (size_t)ss;
	num = (unsigned)(fsize / (dim + 1) / 4);
	data = new float[(size_t)num * (size_t)dim];

	in.seekg(0, std::ios::beg);
	for (size_t i = 0; i < num; i++) {
		in.seekg(4, std::ios::cur);
		in.read((char*)(data + i * dim), dim * 4);
	}
	in.close();
}


int main(int argc, char** argv) {
	srand((unsigned)time(NULL));
	if (argc != 7) {
		std::cout << argv[0] << " data_file nn_graph_path L R C save_graph_file"
			<< std::endl;
		exit(-1);
	}
	float* data_load = NULL;
	unsigned points_num, dim;
	//load_data(argv[1], data_load, points_num, dim);
	std::string filenames = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-images.idx3-ubyte";
	load_mnist_data(filenames, data_load, points_num, dim);
	std::cout << "dim:" << dim << " num:" << points_num << std::endl;
	std::string nn_graph_path(argv[2]);
	unsigned L = (unsigned)atoi(argv[3]);
	unsigned R = (unsigned)atoi(argv[4]);
	unsigned C = (unsigned)atoi(argv[5]);
	std::cout << "L:" << L << " R:" << R << " C:" << C << std::endl;

	// data_load = efanna2e::data_align(data_load, points_num, dim);//one must
	// align the data before build
	efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);

	auto s = std::chrono::high_resolution_clock::now();
	efanna2e::Parameters paras;
	paras.Set<unsigned>("L", L);
	paras.Set<unsigned>("R", R);
	paras.Set<unsigned>("C", C);
	paras.Set<std::string>("nn_graph_path", nn_graph_path);

	index.Build(points_num, data_load, paras);
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = e - s;

	std::cout << "indexing time: " << diff.count() << "\n";

	index.Save(argv[6]);

	std::string label1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-labels.idx1-ubyte";
	std::vector<double>labels;
	read_Mnist_Label(label1, labels);
	index.Show(labels);


	system("pause");
	return 0;
}
