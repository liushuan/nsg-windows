/* 
    Copyright (C) 2013,2014 Wei Dong <wdong@wdong.org>. All Rights Reserved.
*/

#ifndef KGRAPH_VALUE_TYPE
#define KGRAPH_VALUE_TYPE float
#endif

#include <time.h>
#include <cctype>
#include <random>
#include <iomanip>
#include <type_traits>
#include <boost/timer/timer.hpp>
#include <boost/tr1/random.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include "kgraph.h"
#include "kgraph-data.h"
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace boost;
using namespace boost::timer;
using namespace kgraph;

namespace po = boost::program_options; 

typedef KGRAPH_VALUE_TYPE value_type;

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
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

int main (int argc, char *argv[]) {
    string data_path;
    string output_path;
    KGraph::IndexParams params;
    unsigned D;
    unsigned skip;
    unsigned gap;
    unsigned synthetic;
    float noise;

    bool lshkit = true;

    po::options_description desc_visible("General options");
    desc_visible.add_options()
    ("help,h", "produce help message.")
    ("version,v", "print version information.")
    ("data", po::value(&data_path), "input path")
    ("output", po::value(&output_path), "output path")
    (",K", po::value(&params.K)->default_value(default_K), "number of nearest neighbor")
    ("controls,C", po::value(&params.controls)->default_value(default_controls), "number of control pounsigneds")
    ;

    po::options_description desc_hidden("Expert options");
    desc_hidden.add_options()
    ("iterations,I", po::value(&params.iterations)->default_value(default_iterations), "")
    (",S", po::value(&params.S)->default_value(default_S), "")
    (",R", po::value(&params.R)->default_value(default_R), "")
    (",L", po::value(&params.L)->default_value(default_L), "")
    ("delta", po::value(&params.delta)->default_value(default_delta), "")
    ("recall", po::value(&params.recall)->default_value(default_recall), "")
    ("prune", po::value(&params.prune)->default_value(default_prune), "")
    ("reverse", po::value(&params.reverse)->default_value(default_reverse), "")
    ("noise", po::value(&noise)->default_value(0), "noise")
    ("seed", po::value(&params.seed)->default_value(default_seed), "")
    ("dim,D", po::value(&D), "dimension, see format")
    ("skip", po::value(&skip)->default_value(0), "see format")
    ("gap", po::value(&gap)->default_value(0), "see format")
    ("raw", "read raw binary file, need to specify D.")
    ("synthetic", po::value(&synthetic)->default_value(0), "generate synthetic data, for performance evaluation only, specify number of points")
    ("l2norm", "l2-normalize data, so as to mimic cosine similarity")
    ;

    po::options_description desc("Allowed options");
    desc.add(desc_visible).add(desc_hidden);

    po::positional_options_description p;
    p.add("data", 1);
    p.add("output", 1);

    po::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("raw") == 1) {
        lshkit = false;
    }

    if (vm.count("version")) {
        std::cout << "KGraph version " << KGraph::version() << endl;
        return 0;
    }

    if (vm.count("help")
            || (synthetic && (vm.count("dim") == 0 || vm.count("data")))
            || (!synthetic && (vm.count("data") == 0 || (vm.count("dim") == 0 && !lshkit)))) {
        std::cout << "Usage: index [OTHER OPTIONS]... INPUT [OUTPUT]" << endl;
        std::cout << desc_visible << endl;
		std::cout << desc_hidden << endl;
        return 0;
    }

    if (params.S == 0) {
        params.S = params.K;
    }

    if (lshkit && (synthetic == 0)) {   // read dimension information from the data file
        static const unsigned LSHKIT_HEADER = 3;
        ifstream is(data_path.c_str(), ios::binary);
        unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
        is.read((char *)header, sizeof header);
        BOOST_VERIFY(is);
        BOOST_VERIFY(header[0] == sizeof(value_type));
        is.close();
        D = header[2];
        skip = LSHKIT_HEADER * sizeof(unsigned);
        gap = 0;
    }

    Matrix<value_type> data;
    if (synthetic) {
        if (!std::is_floating_point<value_type>::value) {
            throw std::runtime_error("synthetic data not implemented for non-floating-point values.");
        }
        data.resize(synthetic, D);
        std::cout << "Generating synthetic data..." << endl;
        default_random_engine rng(params.seed);
        uniform_real_distribution<double> distribution(-1.0, 1.0);
        data.zero(); // important to do that
        for (unsigned i = 0; i < synthetic; ++i) {
            value_type *row = data[i];
            for (unsigned j = 0; j < D; ++j) {
                row[j] = distribution(rng);
            }
        }
    }
    else {
		/*D = 128;
		gap = 4;
		skip = 0;
		std::cout << data_path <<" D:"<<D <<" skip:"<<skip<<" gap:"<<gap<< std::endl;
        data.load(data_path, D, skip, gap);*/

		std::string filenames = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-images.idx3-ubyte";
		data.load_mnist_data(filenames, 28*28);

		
	}
	std::string label1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-labels.idx1-ubyte";
	std::vector<double>labels;
	read_Mnist_Label(label1, labels);

    if (noise != 0) {
        if (!std::is_floating_point<value_type>::value) {
            throw std::runtime_error("noise injection not implemented for non-floating-point value.");
        }
        tr1::ranlux64_base_01 rng;
        double sum = 0, sum2 = 0;
        for (unsigned i = 0; i < data.size(); ++i) {
            for (unsigned j = 0; j < data.dim(); ++j) {
                value_type v = data[i][j];
                sum += v;
                sum2 += v * v;
            }
        }
        double total = double(data.size()) * data.dim();
        double avg2 = sum2 / total, avg = sum / total;
        double dev = sqrt(avg2 - avg * avg);
        std::cout << "Adding Gaussian noise w/ " << noise << "x sigma(" << dev << ")..." << endl;
        std::normal_distribution<double> gaussian(0, noise * dev);
        for (unsigned i = 0; i < data.size(); ++i) {
            for (unsigned j = 0; j < data.dim(); ++j) {
                data[i][j] += gaussian(rng);
            }
        }
    }
    if (vm.count("l2norm")) {
        std::cout << "L2-normalizing data..." << endl;
        data.normalize2();
    }

    MatrixOracle<value_type, metric::l2sqr> oracle(data);
    KGraph::IndexInfo info;
    KGraph *kgraph = KGraph::create(); //(oracle, params, &info);
    {
        auto_cpu_timer timer;
        kgraph->build(oracle, params, &info);
        std::cout << info.stop_condition << endl;
    }
    if (output_path.size()) {
        kgraph->save(output_path.c_str(), labels, KGraph::FORMAT_NO_DIST);
    }
    delete kgraph;


	system("pause");
    return 0;
}

