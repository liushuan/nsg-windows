#include <efanna2e/index_nsg.h>
#include <efanna2e/exceptions.h>
#include <efanna2e/parameters.h>
#include <omp.h>
#include <chrono>
#include <boost/dynamic_bitset.hpp>
#include <bitset>
#include <cmath>

#include <opencv2\opencv.hpp>

static int ReverseInt(int i)
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
static void read_Mnist_Label(std::string filename, std::vector<double>&labels)
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

void Show_img(const float * data) {
	int rows = 28;
	int cols = 28;
	cv::Mat img(28, 28, CV_8UC1);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			img.at<uchar>(i, j) = static_cast<uchar>(data[i*cols + j]);
		}
	}
	cv::imshow("img", img);
	cv::waitKey(0);
}

namespace efanna2e {
#define _CONTROL_NUM 100
	IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer) : Index(dimension, n, m),
		initializer_{ initializer } {
	}

	IndexNSG::~IndexNSG() {}

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

	void IndexNSG::Show(std::vector<double>labels) {

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

	void IndexNSG::Save(const char *filename) {
		std::ofstream out(filename, std::ios::binary | std::ios::out);
		assert(final_graph_.size() == nd_);

		out.write((char *)&width, sizeof(unsigned));
		out.write((char *)&ep_, sizeof(unsigned));
		for (unsigned i = 0; i < nd_; i++) {
			unsigned GK = (unsigned)final_graph_[i].size();
			out.write((char *)&GK, sizeof(unsigned));
			out.write((char *)final_graph_[i].data(), GK * sizeof(unsigned));
		}
		out.close();
	}

	void IndexNSG::Load(const char *filename) {
		std::ifstream in(filename, std::ios::binary);
		in.read((char *)&width, sizeof(unsigned));
		in.read((char *)&ep_, sizeof(unsigned));
		//width=100;
		unsigned cc = 0;
		while (!in.eof()) {
			unsigned k;
			in.read((char *)&k, sizeof(unsigned));
			if (in.eof())break;
			cc += k;
			std::vector<unsigned> tmp(k);
			in.read((char *)tmp.data(), k * sizeof(unsigned));
			final_graph_.push_back(tmp);
		}
		cc /= nd_;
		//std::cout<<cc<<std::endl;

		//write_final graph////////////////////
		std::string final_graph_files = "D:\\work_space\\work_code\\nsg\\source_code\\nsg-master\\siftsmall\\result_path\\final_graph.text";
		std::ofstream out;
		out.open(final_graph_files, std::ios::trunc);


		std::string label1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-labels.idx1-ubyte";
		std::vector<double>labels;
		read_Mnist_Label(label1, labels);
		for (unsigned i = 0; i < final_graph_.size(); i++) {
			unsigned GK = (unsigned)final_graph_[i].size();

			out << "KE:" << labels[i] << std::endl;

			std::string line = "";
			int length = final_graph_[i].size();
			for (size_t j = 0; j < length; j++)
			{
				line.append(std::to_string(labels[final_graph_[i][j]]) + " ");
			}
			out << " index:" << line << std::endl;
		}
		out.close();

		/////////////////////////////////////////

	}
	void IndexNSG::Load_nn_graph(const char *filename) {
		std::ifstream in(filename, std::ios::binary);

		in.seekg(0, std::ios::beg);

		unsigned num, k = 0;
		in.read((char *)&num, sizeof(unsigned));

		//in.seekg(0, std::ios::beg);
		
		final_graph_.resize(num);
		final_graph_.reserve(num);
		//unsigned kk = (k+3)/4*4;

		for (size_t i = 0; i < num; i++) {
			//in.seekg(4, std::ios::cur);

			in.read((char *)&k, sizeof(unsigned));
			unsigned kk = (k + 3) / 4 * 4;

			if (i == 0) {
				std::cout << "num is:" << num << " k:" << k << " KK:"<<kk<< std::endl;
			}

			final_graph_[i].resize(k);
			final_graph_[i].reserve(kk);
			in.read((char *)final_graph_[i].data(), k * sizeof(unsigned));
		}
		in.close();
	}

	void IndexNSG::get_neighbors(
		const float *query,
		const Parameters &parameter,
		std::vector <Neighbor> &retset, std::vector <Neighbor> &fullset) {
		unsigned L = parameter.Get<unsigned>("L");

		retset.resize(L + 1);
		std::vector<unsigned> init_ids(L);
		//initializer_->Search(query, nullptr, L, parameter, init_ids.data());

		boost::dynamic_bitset<> flags{ nd_, 0 };
		L = 0;
		std::cout << "final_graph_:" << final_graph_.size() << "  " << final_graph_[0].size() << std::endl;
		for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
			init_ids[i] = final_graph_[ep_][i];
			flags[init_ids[i]] = true;
			L++;
		}
		std::cout << " L is:" << L << " init_ids.size() :" << init_ids.size() << std::endl;
		while (L < init_ids.size()) {
			unsigned id = rand() % nd_;
			if (flags[id])continue;
			init_ids[L] = id;
			L++;
			flags[id] = true;
		}
		L = 0;
		for (unsigned i = 0; i < init_ids.size(); i++) {
			unsigned id = init_ids[i];
			if (id >= nd_)continue;
			//std::cout<<id<<std::endl;
			//Show_img(data_ + dimension_ * (size_t)id);
			//Show_img(query);
			float dist = distance_->compare(data_ + dimension_ * (size_t)id, query, (unsigned)dimension_);
			retset[i] = Neighbor(id, dist, true);
			//flags[id] = 1;
			L++;
		}
		std::sort(retset.begin(), retset.begin() + L);
		int k = 0;
		while (k < (int)L) {
			int nk = L;

			if (retset[k].flag) {
				retset[k].flag = false;
				unsigned n = retset[k].id;

				for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
					unsigned id = final_graph_[n][m];
					if (flags[id])continue;
					flags[id] = 1;

					float dist = distance_->compare(query, data_ + dimension_ * (size_t)id, (unsigned)dimension_);
					Neighbor nn(id, dist, true);
					fullset.push_back(nn);
					if (dist >= retset[L - 1].distance)continue;
					int r = InsertIntoPool(retset.data(), L, nn);

					if (L + 1 < retset.size()) ++L;
					if (r < nk)nk = r;
				}

			}
			if (nk <= k)k = nk;
			else ++k;
		}
	}

	void IndexNSG::get_neighbors(
		const float *query,
		const Parameters &parameter,
		boost::dynamic_bitset<>& flags,
		std::vector <Neighbor> &retset,
		std::vector <Neighbor> &fullset) {
		unsigned L = parameter.Get<unsigned>("L");

		retset.resize(L + 1);
		std::vector<unsigned> init_ids(L);
		//initializer_->Search(query, nullptr, L, parameter, init_ids.data());

		L = 0;
		for (unsigned i = 0; i < init_ids.size() && i < final_graph_[ep_].size(); i++) {
			init_ids[i] = final_graph_[ep_][i];
			flags[init_ids[i]] = true;
			L++;
		}
		while (L < init_ids.size()) {
			unsigned id = rand() % nd_;
			if (flags[id])continue;
			init_ids[L] = id;
			L++;
			flags[id] = true;
		}

		L = 0;
		for (unsigned i = 0; i < init_ids.size(); i++) {
			unsigned id = init_ids[i];
			if (id >= nd_)continue;
			//std::cout<<id<<std::endl;
			float dist = distance_->compare(data_ + dimension_ * (size_t)id, query, (unsigned)dimension_);
			retset[i] = Neighbor(id, dist, true);
			fullset.push_back(retset[i]);
			//flags[id] = 1;
			L++;
		}


		//std::cout << " Link :"<< " " << retset[0].id << " " << retset[1].id << " " << retset[2].id << " " << retset[3].id << " " << retset[4].id << std::endl;
		//system("pause");

		std::sort(retset.begin(), retset.begin() + L);
		int k = 0;
		while (k < (int)L) {
			int nk = L;

			if (retset[k].flag) {
				retset[k].flag = false;
				unsigned n = retset[k].id;

				for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
					unsigned id = final_graph_[n][m];
					if (flags[id])continue;
					flags[id] = 1;

					float dist = distance_->compare(query, data_ + dimension_ * (size_t)id, (unsigned)dimension_);
					Neighbor nn(id, dist, true);
					fullset.push_back(nn);
					if (dist >= retset[L - 1].distance)continue;
					int r = InsertIntoPool(retset.data(), L, nn);

					if (L + 1 < retset.size()) ++L;
					if (r < nk)nk = r;
				}

			}
			if (nk <= k)k = nk;
			else ++k;
		}
	}

	void IndexNSG::init_graph(const Parameters &parameters) {
		srand((unsigned)time(NULL));
		std::cout << "dimension_:" << dimension_ << " nd_:" << nd_ << std::endl;
		float *center = new float[dimension_];
		for (unsigned j = 0; j < dimension_; j++)center[j] = 0;
		for (unsigned i = 0; i < nd_; i++) {
			for (unsigned j = 0; j < dimension_; j++) {
				center[j] += data_[i * dimension_ + j];
			}
		}
		for (unsigned j = 0; j < dimension_; j++) {
			center[j] /= nd_;
		}
		std::vector <Neighbor> tmp, pool;
		ep_ = rand() % nd_;   // random initialize ep_
		std::cout << "----------------ep_ is:" << ep_ << std::endl;
		get_neighbors(center, parameters, tmp, pool);
		ep_ = tmp[0].id;
	}
	/*
	void IndexNSG::add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph &cut_graph_) {
	  LockGuard guard(cut_graph_[des].lock);
	  for (unsigned i = 0; i < cut_graph_[des].pool.size(); i++) {
		if (p.id == cut_graph_[des].pool[i].id)return;
	  }
	  cut_graph_[des].pool.push_back(p);
	  if (cut_graph_[des].pool.size() > range) {
		std::vector <Neighbor> result;
		std::vector <Neighbor> &pool = cut_graph_[des].pool;
		unsigned start = 0;
		std::sort(pool.begin(), pool.end());
		result.push_back(pool[start]);

		while (result.size() < range && (++start) < pool.size()) {
		  auto &p = pool[start];
		  bool occlude = false;
		  for (unsigned t = 0; t < result.size(); t++) {
			if (p.id == result[t].id) {
			  occlude = true;
			  break;
			}
			float djk = distance_->compare(data_ + dimension_ * result[t].id, data_ + dimension_ * p.id, dimension_);
			if (djk < p.distance dik ) {
					 occlude = true;
			  break;
			}

		  }
		  if (!occlude)result.push_back(p);
		}
		pool.swap(result);
	  }

	}
	*/
	void IndexNSG::sync_prune(unsigned q,
		std::vector <Neighbor> &pool,
		const Parameters &parameter,
		boost::dynamic_bitset<>& flags,
		SimpleNeighbor* cut_graph_) {
		unsigned range = parameter.Get<unsigned>("R");
		unsigned maxc = parameter.Get<unsigned>("C");
		width = range;
		unsigned start = 0;

		for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
			unsigned id = final_graph_[q][nn];
			if (flags[id])continue;
			float dist = distance_->compare(data_ + dimension_ * (size_t)q, data_ + dimension_ * (size_t)id, (unsigned)dimension_);
			pool.push_back(Neighbor(id, dist, true));
		}

		std::sort(pool.begin(), pool.end());
		std::vector <Neighbor> result;
		if (pool[start].id == q)start++;
		result.push_back(pool[start]);

		while (result.size() < range && (++start) < pool.size() && start < maxc) {
			auto &p = pool[start];
			bool occlude = false;
			for (unsigned t = 0; t < result.size(); t++) {
				if (p.id == result[t].id) {
					occlude = true;
					break;
				}
				float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id, data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);
				if (djk < p.distance/* dik */) {
					occlude = true;
					break;
				}

			}
			if (!occlude)result.push_back(p);
		}

		SimpleNeighbor* des_pool = cut_graph_ + (size_t)q * (size_t)range;
		for (size_t t = 0; t < result.size(); t++) {
			des_pool[t].id = result[t].id;
			des_pool[t].distance = result[t].distance;
		}
		if (result.size() < range) {
			des_pool[result.size()].distance = -1;
		}
		//for (unsigned t = 0; t < result.size(); t++) {
		  //add_cnn(q, result[t], range, cut_graph_);
		  //add_cnn(result[t].id, Neighbor(q, result[t].distance, true), range, cut_graph_);
		//}
	}

	void IndexNSG::InterInsert(unsigned n, unsigned range, std::vector<std::mutex>& locks,
		SimpleNeighbor* cut_graph_) {

		SimpleNeighbor* src_pool = cut_graph_ + (size_t)n * (size_t)range;
		for (size_t i = 0; i < range; i++) {
			if (src_pool[i].distance == -1)break;

			SimpleNeighbor sn(n, src_pool[i].distance);
			size_t des = src_pool[i].id;
			SimpleNeighbor* des_pool = cut_graph_ + des * (size_t)range;

			std::vector<SimpleNeighbor> temp_pool;
			int dup = 0;
			{
				LockGuard guard(locks[des]);
				for (size_t j = 0; j < range; j++) {
					if (des_pool[j].distance == -1)break;
					if (n == des_pool[j].id) { dup = 1; break; }
					temp_pool.push_back(des_pool[j]);
				}
			}
			if (dup)continue;

			temp_pool.push_back(sn);
			if (temp_pool.size() > range) {
				std::vector <SimpleNeighbor> result;
				unsigned start = 0;
				std::sort(temp_pool.begin(), temp_pool.end());
				result.push_back(temp_pool[start]);
				while (result.size() < range && (++start) < temp_pool.size()) {
					auto &p = temp_pool[start];
					bool occlude = false;
					for (unsigned t = 0; t < result.size(); t++) {
						if (p.id == result[t].id) {
							occlude = true;
							break;
						}
						float djk = distance_->compare(data_ + dimension_ * (size_t)result[t].id, data_ + dimension_ * (size_t)p.id, (unsigned)dimension_);
						if (djk < p.distance/* dik */) {
							occlude = true;
							break;
						}

					}
					if (!occlude)result.push_back(p);
				}
				{
					LockGuard guard(locks[des]);
					for (unsigned t = 0; t < result.size(); t++) {
						des_pool[t] = result[t];
					}
				}
			}
			else {
				LockGuard guard(locks[des]);
				for (unsigned t = 0; t < range; t++) {
					if (des_pool[t].distance == -1) {
						des_pool[t] = sn;
						if (t + 1 < range)des_pool[t + 1].distance = -1;
						break;
					}
				}
			}

		}
	}

	void IndexNSG::Link(const Parameters &parameters, SimpleNeighbor* cut_graph_) {
		std::cout << " graph link" << std::endl;
		unsigned progress = 0;
		unsigned percent = 100;
		unsigned step_size = nd_ / percent;
		std::mutex progress_lock;
		unsigned range = parameters.Get<unsigned>("R");
		std::vector<std::mutex> locks(nd_);


		std::string label1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-labels.idx1-ubyte";
		std::vector<double>labels;
		read_Mnist_Label(label1, labels);

#pragma omp parallel
		{
			unsigned cnt = 0;
			std::vector <Neighbor> pool, tmp;
			boost::dynamic_bitset<> flags{ nd_, 0 };
#pragma omp for schedule(dynamic, 100)
			for (int n = 0; n < nd_; ++n) {
				pool.clear();
				tmp.clear();
				flags.reset();
				//ep_ = n;
				get_neighbors(data_ + dimension_ * n, parameters, flags, tmp, pool);

				if (n < 5) {
					std::cout << " Link :" << labels[n] << " " << tmp[0].id << " " << tmp[1].id << " " << tmp[2].id << " " << tmp[3].id << " " << tmp[4].id << std::endl;
				}
				sync_prune(n, pool, parameters, flags, cut_graph_);
				cnt++;
				if (cnt % step_size == 0) {
					LockGuard g(progress_lock);
					std::cout << progress++ << "/" << percent << " completed" << std::endl;
				}
			}

#pragma omp for schedule(dynamic, 100)
			for (int n = 0; n < nd_; ++n) {
				InterInsert(n, range, locks, cut_graph_);

			}

		}

	}

	void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
		std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
		unsigned range = parameters.Get<unsigned>("R");
		std::cout << "file knn_graph:" << nn_graph_path << std::endl;
		Load_nn_graph(nn_graph_path.c_str());
		data_ = data;
		init_graph(parameters);
		SimpleNeighbor* cut_graph_ = new SimpleNeighbor[nd_*(size_t)range];
		std::cout << "memory allocated\n";
		Link(parameters, cut_graph_);

		std::string label1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-labels.idx1-ubyte";
		std::vector<double>labels;
		read_Mnist_Label(label1, labels);
		for (unsigned i = 0; i < nd_; i++) {
			if (i < 5)
				std::cout << "cut:" << labels[i] << std::endl;

			if (i < 5)
				for (size_t j = 0; j < range; j++)
				{
					std::cout << " " << labels[cut_graph_[i * range + j].id];
				}
			if (i < 5) {
				std::cout << std::endl;
			}
		}


		final_graph_.resize(nd_);

		unsigned max = 0, min = 1e6, avg = 0, cnt = 0;
		for (size_t i = 0; i < nd_; i++) {
			SimpleNeighbor* pool = cut_graph_ + i * (size_t)range;
			unsigned pool_size = 0;
			for (unsigned j = 0; j < range; j++) {
				if (pool[j].distance == -1)break;
				pool_size = j;
			}
			pool_size++;

			max = max < pool_size ? pool_size : max;
			min = min > pool_size ? pool_size : min;
			avg += pool_size;
			if (pool_size < 2)cnt++;

			final_graph_[i].resize(pool_size);
			for (unsigned j = 0; j < pool_size; j++) {
				final_graph_[i][j] = pool[j].id;

				if (i < 5)
					std::cout << pool[j].id << " ";
			}
			if (i < 5)
				std::cout << std::endl;
		}
		avg /= 1.0 * nd_;

		//std::cout << max << ":" << avg << ":" << min << ":" << cnt << "\n";
		//tree_grow(parameters);
		//max = 0;
		//for (unsigned i = 0; i < nd_; i++) {
		//    max = max < final_graph_[i].size() ? final_graph_[i].size() : max;
		//}
		//if(max > width)width = max;
		has_built = true;
	}

	void IndexNSG::Search(
		const float *query,
		const float *x,
		size_t K,
		const Parameters &parameters,
		unsigned *indices) {
		const unsigned L = parameters.Get<unsigned>("L_search");
		data_ = x;
		std::vector <Neighbor> retset(L + 1);
		std::vector<unsigned> init_ids(L);
		boost::dynamic_bitset<> flags{ nd_, 0 };
		//std::mt19937 rng(rand());
		//GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

		unsigned tmp_l = 0;
		for (; tmp_l < L && tmp_l < final_graph_[ep_].size(); tmp_l++) {
			init_ids[tmp_l] = final_graph_[ep_][tmp_l];
			flags[init_ids[tmp_l]] = true;
		}

		while (tmp_l < L) {
			unsigned id = rand() % nd_;
			if (flags[id])continue;
			flags[id] = true;
			init_ids[tmp_l] = id;
			tmp_l++;
		}


		for (unsigned i = 0; i < init_ids.size(); i++) {
			unsigned id = init_ids[i];
			float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned)dimension_);
			retset[i] = Neighbor(id, dist, true);
			//flags[id] = true;
		}

		std::sort(retset.begin(), retset.begin() + L);
		int k = 0;
		while (k < (int)L) {
			int nk = L;

			if (retset[k].flag) {
				retset[k].flag = false;
				unsigned n = retset[k].id;

				for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
					unsigned id = final_graph_[n][m];
					if (flags[id])continue;
					flags[id] = 1;
					float dist = distance_->compare(query, data_ + dimension_ * id, (unsigned)dimension_);
					if (dist >= retset[L - 1].distance)continue;
					Neighbor nn(id, dist, true);
					int r = InsertIntoPool(retset.data(), L, nn);

					if (r < nk)nk = r;
				}
			}
			if (nk <= k)k = nk;
			else ++k;
		}
		for (size_t i = 0; i < K; i++) {
			indices[i] = retset[i].id;
		}
	}

	void IndexNSG::SearchWithOptGraph(
		const float *query,
		size_t K,
		const Parameters &parameters,
		unsigned *indices) {
		unsigned L = parameters.Get<unsigned>("L_search");
		unsigned P = parameters.Get<unsigned>("P_search");
		DistanceFastL2* dist_fast = (DistanceFastL2*)distance_;

		P = P > K ? P : K;
		std::vector <Neighbor> retset(P + 1);
		std::vector<unsigned> init_ids(L);
		//std::mt19937 rng(rand());
		//GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

		boost::dynamic_bitset<> flags{ nd_, 0 };
		unsigned tmp_l = 0;
		unsigned *neighbors = (unsigned*)(opt_graph_ + node_size * ep_ + data_len);
		unsigned MaxM_ep = *neighbors;
		neighbors++;

		for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
			init_ids[tmp_l] = neighbors[tmp_l];
			flags[init_ids[tmp_l]] = true;
		}

		while (tmp_l < L) {
			unsigned id = rand() % nd_;
			if (flags[id])continue;
			flags[id] = true;
			init_ids[tmp_l] = id;
			tmp_l++;
		}

		for (unsigned i = 0; i < init_ids.size(); i++) {
			unsigned id = init_ids[i];
			if (id >= nd_)continue;
			_mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
		}
		L = 0;
		for (unsigned i = 0; i < init_ids.size(); i++) {
			unsigned id = init_ids[i];
			if (id >= nd_)continue;
			float *x = (float*)(opt_graph_ + node_size * id);
			float norm_x = *x; 
			x++;
			float dist = dist_fast->compare(x, query, norm_x, (unsigned)dimension_);
			//std::cout << "dist" << dist << std::endl;
			//system("pause");
			retset[i] = Neighbor(id, dist, true);
			flags[id] = true;
			L++;
		}
		//std::cout<<L<<std::endl;

		std::sort(retset.begin(), retset.begin() + L);
		int k = 0;
		while (k < (int)L) {
			int nk = L;

			if (retset[k].flag) {
				retset[k].flag = false;
				unsigned n = retset[k].id;

				_mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
				unsigned *neighbors = (unsigned*)(opt_graph_ + node_size * n + data_len);
				unsigned MaxM = *neighbors;
				neighbors++;
				for (unsigned m = 0; m < MaxM; ++m)
					_mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
				for (unsigned m = 0; m < MaxM; ++m) {
					unsigned id = neighbors[m];
					if (flags[id])continue;
					flags[id] = 1;
					float *data = (float*)(opt_graph_ + node_size * id);
					float norm = *data; data++;
					float dist = dist_fast->compare(query, data, norm, (unsigned)dimension_);
					if (dist >= retset[L - 1].distance)continue;
					Neighbor nn(id, dist, true);
					int r = InsertIntoPool(retset.data(), L, nn);

					//if(L+1 < retset.size()) ++L;
					if (r < nk)nk = r;
				}

			}
			if (nk <= k)k = nk;
			else ++k;
		}
		for (size_t i = 0; i < K; i++) {
			indices[i] = retset[i].id;
		}
	}

	void IndexNSG::OptimizeGraph(float* data) {//use after build or load

		data_ = data;
		data_len = (dimension_ + 1) * sizeof(float);
		neighbor_len = (width + 1) * sizeof(unsigned);
		node_size = data_len + neighbor_len;
		opt_graph_ = (char*)malloc(node_size * nd_);
		DistanceFastL2* dist_fast = (DistanceFastL2*)distance_;

		std::cout << " width is:" << width << std::endl;

		for (unsigned i = 0; i < nd_; i++) {
			char* cur_node_offset = opt_graph_ + i * node_size;
			float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);

			//std::cout << "cur_norm: " << cur_norm << std::endl;
			//system("pause");
			std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
			std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_, data_len - sizeof(float));

			cur_node_offset += data_len;
			unsigned k = final_graph_[i].size();
			std::memcpy(cur_node_offset, &k, sizeof(unsigned));
;
			std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(), k * sizeof(unsigned));
			std::vector<unsigned>().swap(final_graph_[i]);
		}
		free(data);
		data_ = nullptr;
		CompactGraph().swap(final_graph_);

		/*std::string label1 = "D:\\work_space\\work_code\\nsg\\source_code\\kgraph\\mnist\\train-labels.idx1-ubyte";
		std::vector<double>labels;
		read_Mnist_Label(label1, labels);
		Show(labels);*/

	}

	void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt) {
		unsigned tmp = root;
		std::stack<unsigned> s;
		s.push(root);
		if (!flag[root])cnt++;
		flag[root] = true;
		while (!s.empty()) {

			unsigned next = nd_ + 1;
			for (unsigned i = 0; i < final_graph_[tmp].size(); i++) {
				if (flag[final_graph_[tmp][i]] == false) {
					next = final_graph_[tmp][i];
					break;
				}
			}
			//std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
			if (next == (nd_ + 1)) {
				s.pop();
				if (s.empty())break;
				tmp = s.top();
				continue;
			}
			tmp = next;
			flag[tmp] = true; s.push(tmp); cnt++;
		}
	}

	void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter) {
		unsigned id;
		for (unsigned i = 0; i < nd_; i++) {
			if (flag[i] == false) {
				id = i;
				break;
			}
		}
		std::vector <Neighbor> tmp, pool;
		get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
		std::sort(pool.begin(), pool.end());

		unsigned found = 0;
		for (unsigned i = 0; i < pool.size(); i++) {
			if (flag[pool[i].id]) {
				//std::cout << pool[i].id << '\n';
				root = pool[i].id;
				found = 1;
				break;
			}
		}
		if (found == 0) {
			while (true) {
				unsigned rid = rand() % nd_;
				if (flag[rid]) {
					root = rid;
					//std::cout << root << '\n';
					break;
				}
			}
		}
		final_graph_[root].push_back(id);

	}
	void IndexNSG::tree_grow(const Parameters &parameter) {
		unsigned root = ep_;
		boost::dynamic_bitset<> flags{ nd_, 0 };
		unsigned unlinked_cnt = 0;
		while (unlinked_cnt < nd_) {
			DFS(flags, root, unlinked_cnt);
			//std::cout << unlinked_cnt << '\n';
			if (unlinked_cnt >= nd_)break;
			findroot(flags, root, parameter);
			//std::cout << "new root"<<":"<<root << '\n';
		}
	}

}
