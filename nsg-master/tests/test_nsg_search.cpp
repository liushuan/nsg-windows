//
// Created by 付聪 on 2017/6/21.
//

#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
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
		dim = n_rows*n_cols;
		num = number_of_images;
		data = new float[(size_t)num * (size_t)dim];

		for (int i = 0; i < number_of_images; i++)
		{
			//cv::Mat dst(28, 28, cv::CV_8UC1);
			cv::Mat dst(28, 28, CV_8UC1);
			for (int r = 0; r < n_rows; r++)
			{
				for (int c = 0; c < n_cols; c++)
				{
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					//tp.push_back(image);
					data[dim * i + r*n_cols + c] = static_cast<float>(image);
					dst.at<uchar>(r, c) = image;
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

void Show(std::vector<std::vector<unsigned>> final_graph_, std::vector<double>labels) {

	unsigned nd_ = final_graph_.size();
	for (unsigned i = 0; i < nd_; i++) {
		if (i < 5)
			std::cout << "Kernel:" << labels[i] << std::endl;
		unsigned GK = (unsigned)final_graph_[i].size();

		if (i < 5)
			for (size_t j = 0; j < GK; j++)
			{
				std::cout << " " << labels[final_graph_[i][j]];
			}
		if (i < 5) {
			std::cout << std::endl;
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
	std::cout << "data dimension: " << dim << std::endl;
	in.seekg(0, std::ios::end);
	std::ios::pos_type ss = in.tellg();
	size_t fsize = (size_t)ss;
	num = (unsigned)(fsize / (dim + 1) / 4);
	data = new float[num * dim * sizeof(float)];

	in.seekg(0, std::ios::beg);
	for (size_t i = 0; i < num; i++) {
		in.seekg(4, std::ios::cur);
		in.read((char*)(data + i * dim), dim * 4);
	}
	in.close();
}

void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
	std::ofstream out(filename, std::ios::binary | std::ios::out);

	for (unsigned i = 0; i < results.size(); i++) {
		unsigned GK = (unsigned)results[i].size();
		out.write((char*)&GK, sizeof(unsigned));
		out.write((char*)results[i].data(), GK * sizeof(unsigned));
	}
	out.close();
}
void save_result_text(const char* filename,
	std::vector<std::vector<unsigned> >& results, std::vector<double>label_train, std::vector<double>label_test) {
	std::ofstream out;
	out.open(filename, std::ios::trunc);

	for (unsigned i = 0; i < results.size(); i++) {
		unsigned GK = (unsigned)results[i].size();

		out << "Kernel:" << label_test[i] << std::endl;

		if (i < 5) {
			std::cout << "test :" << label_test[i] << std::endl;
		}

		std::string line = "";
		int length = results[i].size();
		for (size_t j = 0; j < length; j++)
		{
			line.append(std::to_string(label_train[results[i][j]]) + " ");

			if (i < 5) {
				std::cout << " " << label_train[results[i][j]];
			}
		}
		out << " Nei:" << line << std::endl;
		if (i < 5) {
			std::cout << std::endl;
		}
	}
	out.close();
}
int main(int argc, char** argv) {
	if (argc != 7) {
		std::cout << argv[0]
			<< " data_file query_file nsg_path search_L search_K result_path"
			<< std::endl;
		exit(-1);
	}
	std::string filenames1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-images.idx3-ubyte";
	std::string label1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-labels.idx1-ubyte";
	std::string filenames2 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\t10k-images.idx3-ubyte";
	std::string label2 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\t10k-labels.idx1-ubyte";
	float* data_load = NULL;
	unsigned points_num, dim;
	//load_data(argv[1], data_load, points_num, dim);
	load_mnist_data(filenames1, data_load, points_num, dim);
	std::vector<double> labels_train;
	read_Mnist_Label(label1, labels_train);

	float* query_load = NULL;
	unsigned query_num, query_dim;
	//load_data(argv[2], query_load, query_num, query_dim);
	load_mnist_data(filenames2, query_load, query_num, query_dim);
	std::vector<double> labels_test;
	read_Mnist_Label(label2, labels_test);
	assert(dim == query_dim);

	unsigned L = (unsigned)atoi(argv[4]);
	unsigned K = (unsigned)atoi(argv[5]);

	if (L < K) {
		std::cout << "search_L cannot be smaller than search_K!" << std::endl;
		exit(-1);
	}

	// data_load = efanna2e::data_align(data_load, points_num, dim);//one must
	// align the data before build query_load = efanna2e::data_align(query_load,
	// query_num, query_dim);
	efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
	index.Load(argv[3]);



	efanna2e::Parameters paras;
	paras.Set<unsigned>("L_search", L);
	paras.Set<unsigned>("P_search", L);

	auto s = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<unsigned> > res;
	for (unsigned i = 0; i < query_num; i++) {
		std::vector<unsigned> tmp(K);
		index.Search(query_load + i * dim, data_load, K, paras, tmp.data());
		res.push_back(tmp);
	}
	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = e - s;
	std::cout << "search time: " << diff.count() << "\n";

	save_result(argv[6], res);

	save_result_text(argv[6], res, labels_train, labels_test);



	system("pause");
	return 0;
}
