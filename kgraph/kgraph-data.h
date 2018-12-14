#ifndef WDONG_KGRAPH_DATA
#define WDONG_KGRAPH_DATA

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstring>
#include <malloc.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <boost/assert.hpp>

#ifdef __GNUC__
#ifdef __AVX__
#define KGRAPH_MATRIX_ALIGN 32
#else
#ifdef __SSE2__
#define KGRAPH_MATRIX_ALIGN 16
#else
#define KGRAPH_MATRIX_ALIGN 4
#endif
#endif
#endif

#define KGRAPH_MATRIX_ALIGN 4

namespace kgraph {

    /// L2 square distance with AVX instructions.
    /** AVX instructions have strong alignment requirement for t1 and t2.
     */
    extern float float_l2sqr_avx (float const *t1, float const *t2, unsigned dim);
    /// L2 square distance with SSE2 instructions.
    extern float float_l2sqr_sse2 (float const *t1, float const *t2, unsigned dim);
    extern float float_l2sqr_sse2 (float const *, unsigned dim);
    extern float float_dot_sse2 (float const *, float const *, unsigned dim);
    /// L2 square distance for uint8_t with SSE2 instructions (for SIFT).
    extern float uint8_l2sqr_sse2 (uint8_t const *t1, uint8_t const *t2, unsigned dim);

    extern float float_l2sqr (float const *, float const *, unsigned dim);
    extern float float_l2sqr (float const *, unsigned dim);
    extern float float_dot (float const *, float const *, unsigned dim);


    using std::vector;

    /// namespace for various distance metrics.
    namespace metric {
        /// L2 square distance.
        struct l2sqr {
            template <typename T>
            /// L2 square distance.
            static float apply (T const *t1, T const *t2, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]) - float(t2[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }

            /// inner product.
            template <typename T>
            static float dot (T const *t1, T const *t2, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    r += float(t1[i]) *float(t2[i]);
                }
                return r;
            }

            /// L2 norm.
            template <typename T>
            static float norm2 (T const *t1, unsigned dim) {
                float r = 0;
                for (unsigned i = 0; i < dim; ++i) {
                    float v = float(t1[i]);
                    v *= v;
                    r += v;
                }
                return r;
            }
        };

        struct l2 {
            template <typename T>
            static float apply (T const *t1, T const *t2, unsigned dim) {
                return sqrt(l2sqr::apply<T>(t1, t2, dim));
            }
        };
    }

    /// Matrix data.
    template <typename T, unsigned A = KGRAPH_MATRIX_ALIGN>
    class Matrix {
        unsigned col;
        unsigned row;
        size_t stride;
        char *data;

        void reset (unsigned r, unsigned c) {
            row = r;
            col = c;
			std::cout << " T size is:" << sizeof(T) << std::endl;
			//stride = (sizeof(T) * c + A - 1) / A * A;
			stride = sizeof(T) * c;
            /*
            data.resize(row * stride);
            */
            if (data) delete(data);
			std::cout << " A:" << A << " row:" << row << " stride:" << stride << std::endl;
            //data = (char *)memalign(A, row * stride); // SSE instruction needs data to be aligned
			//data = (char *)_aligned_malloc(A, row * stride);
			//data = (char *)malloc(row * stride);
			data = new char[r * stride];
            //if (!data) throw runtime_error("memalign");
        }
    public:
        Matrix (): col(0), row(0), stride(0), data(0) {}
        Matrix (unsigned r, unsigned c): data(0) {
            reset(r, c);
        }
        ~Matrix () {
            if (data) free(data);
        }
        unsigned size () const {
            return row;
        }
        unsigned dim () const {
            return col;
        }
        size_t step () const {
            return stride;
        }
        void resize (unsigned r, unsigned c) {
            reset(r, c);
        }
        T const *operator [] (unsigned i) const {
            return reinterpret_cast<T const *>(&data[stride * i]);
        }
        T *operator [] (unsigned i) {
            return reinterpret_cast<T *>(&data[stride * i]);
        }
        void zero () {
            memset(data, 0, row * stride);
        }

        void normalize2 () {
#pragma omp parallel for
            for (int i = 0; i < row; ++i) {
                T *p = operator[](i);
                double sum = metric::l2sqr::norm2(p, col);
                sum = std::sqrt(sum);
                for (unsigned j = 0; j < col; ++j) {
                    p[j] /= sum;
                }
            }
        }
        
        void load (const std::string &path, unsigned dim, unsigned skip = 0, unsigned gap = 0) {
            std::ifstream is(path.c_str(), std::ios::binary);
            //if (!is) throw io_error(path);
            is.seekg(0, std::ios::end);
            size_t size = is.tellg();
            size -= skip;
			
            unsigned line = sizeof(T) * dim + gap;

			std::cout <<"size:"<< size << " line:" << line <<" gap:"<<gap<< std::endl;

            unsigned N =  size / line;
			
            reset(N, dim);
            zero();
            is.seekg(0, std::ios::beg);
            for (unsigned i = 0; i < N; ++i) {
				is.seekg(gap, std::ios::cur);
                is.read(&data[stride * i], sizeof(T) * dim);
            }
            //if (!is) throw io_error(path);
        }

		int ReverseInt(int i)
		{
			unsigned char ch1, ch2, ch3, ch4;
			ch1 = i & 255;
			ch2 = (i >> 8) & 255;
			ch3 = (i >> 16) & 255;
			ch4 = (i >> 24) & 255;
			return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
		}
		void load_mnist_data(std::string filename, int dim) {
			ifstream file(filename, ios::binary);
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
				reset(number_of_images, dim);
				zero();
				for (int i = 0; i < number_of_images; i++)
				{
					//cv::Mat dst(28, 28, cv::CV_8UC1);
					//cv::Mat dst(28, 28, CV_8UC1);
					for (int r = 0; r < n_rows; r++)
					{
						for (int c = 0; c < n_cols; c++)
						{
							unsigned char image = 0;
							file.read((char*)&image, sizeof(uchar));
							float im = (float)image;
							char * data_img = (char*)&im;
					        //data[dim * i + r*n_cols + c] = static_cast<float>(image);
							memcpy(data + sizeof(T)*(dim * i + r*n_cols + c), data_img, sizeof(T));
							//dst.at<uchar>(r, c) = image;
						}
					}
					//float * d = (float *)data;
					//std::cout << d[i*28*28+500] << " " << d[i * 28 * 28 + 600] << std::endl;
					//cv::imshow("img", dst);
					//cv::waitKey(0);
				}
			}
		}

        void load_lshkit (std::string const &path) {
            static const unsigned LSHKIT_HEADER = 3;
            std::ifstream is(path.c_str(), std::ios::binary);
            unsigned header[LSHKIT_HEADER]; /* entry size, row, col */
            is.read((char *)header, sizeof header);
            //if (!is) throw io_error(path);
            //if (header[0] != sizeof(T)) throw io_error(path);
            is.close();
            unsigned D = header[2];
            unsigned skip = LSHKIT_HEADER * sizeof(unsigned);
            unsigned gap = 0;
            load(path, D, skip, gap);
        }

        void save_lshkit (std::string const &path) {
            std::ofstream os(path.c_str(), std::ios::binary);
            unsigned header[3];
            assert(sizeof header == 3*4);
            header[0] = sizeof(T);
            header[1] = row;
            header[2] = col;
            os.write((const char *)header, sizeof(header));
            for (unsigned i = 0; i < row; ++i) {
                os.write(&data[stride * i], sizeof(T) * col);
            }
        }
    };

    /// Matrix proxy to interface with 3rd party libraries (FLANN, OpenCV, NumPy).
    template <typename DATA_TYPE, unsigned A = KGRAPH_MATRIX_ALIGN>
    class MatrixProxy {
        unsigned rows;
        unsigned cols;      // # elements, not bytes, in a row, 
        size_t stride;    // # bytes in a row, >= cols * sizeof(element)
        uint8_t const *data;
    public:
        MatrixProxy (Matrix<DATA_TYPE> const &m)
            : rows(m.size()), cols(m.dim()), stride(m.step()), data(reinterpret_cast<uint8_t const *>(m[0])) {
        }

#ifndef __AVX__
#ifdef FLANN_DATASET_H_
        /// Construct from FLANN matrix.
        MatrixProxy (flann::Matrix<DATA_TYPE> const &m)
            : rows(m.rows), cols(m.cols), stride(m.stride), data(m.data) {
            if (stride % A) throw invalid_argument("bad alignment");
        }
#endif
#ifdef __OPENCV_CORE_HPP__
        /// Construct from OpenCV matrix.
        MatrixProxy (cv::Mat const &m)
            : rows(m.rows), cols(m.cols), stride(m.step), data(m.data) {
            if (stride % A) throw invalid_argument("bad alignment");
        }
#endif
#ifdef NPY_NDARRAYOBJECT_H
        /// Construct from NumPy matrix.
        MatrixProxy (PyArrayObject *obj) {
            if (!obj || (obj->nd != 2)) throw invalid_argument("bad array shape");
            rows = obj->dimensions[0];
            cols = obj->dimensions[1];
            stride = obj->strides[0];
            data = reinterpret_cast<uint8_t const *>(obj->data);
            if (obj->descr->elsize != sizeof(DATA_TYPE)) throw invalid_argument("bad data type size");
            if (stride % A) throw invalid_argument("bad alignment");
            if (!(stride >= cols * sizeof(DATA_TYPE))) throw invalid_argument("bad stride");
        }
#endif
#endif
        unsigned size () const {
            return rows;
        }
        unsigned dim () const {
            return cols;
        }
        DATA_TYPE const *operator [] (unsigned i) const {
            return reinterpret_cast<DATA_TYPE const *>(data + stride * i);
        }
        DATA_TYPE *operator [] (unsigned i) {
            return const_cast<DATA_TYPE *>(reinterpret_cast<DATA_TYPE const *>(data + stride * i));
        }
    };

    /// Oracle for Matrix or MatrixProxy.
    /** DATA_TYPE can be Matrix or MatrixProxy,
    * DIST_TYPE should be one class within the namespace kgraph.metric.
    */
    template <typename DATA_TYPE, typename DIST_TYPE>
    class MatrixOracle: public kgraph::IndexOracle {
        MatrixProxy<DATA_TYPE> proxy;
    public:
        class SearchOracle: public kgraph::SearchOracle {
            MatrixProxy<DATA_TYPE> proxy;
            DATA_TYPE const *query;
        public:
            SearchOracle (MatrixProxy<DATA_TYPE> const &p, DATA_TYPE const *q): proxy(p), query(q) {
            }
            virtual unsigned size () const {
                return proxy.size();
            }
            virtual float operator () (unsigned i) const {
                return DIST_TYPE::apply(proxy[i], query, proxy.dim());
            }
        };
        template <typename MATRIX_TYPE>
        MatrixOracle (MATRIX_TYPE const &m): proxy(m) {
        }
        virtual unsigned size () const {
            return proxy.size();
        }
        virtual float operator () (unsigned i, unsigned j) const {
            return DIST_TYPE::apply(proxy[i], proxy[j], proxy.dim());
        }
        SearchOracle query (DATA_TYPE const *query) const {
            return SearchOracle(proxy, query);
        }
    };

    inline float AverageRecall (Matrix<float> const &gs, Matrix<float> const &result, unsigned K = 0) {
        if (K == 0) {
            K = result.dim();
        }
       /* if (!(gs.dim() >= K)) throw invalid_argument("gs.dim() >= K");
        if (!(result.dim() >= K)) throw invalid_argument("result.dim() >= K");
        if (!(gs.size() >= result.size())) throw invalid_argument("gs.size() > result.size()");*/
        float sum = 0;
        for (unsigned i = 0; i < result.size(); ++i) {
            float const *gs_row = gs[i];
            float const *re_row = result[i];
            // compare
            unsigned found = 0;
            unsigned gs_n = 0;
            unsigned re_n = 0;
            while ((gs_n < K) && (re_n < K)) {
                if (gs_row[gs_n] < re_row[re_n]) {
                    ++gs_n;
                }
                else if (gs_row[gs_n] == re_row[re_n]) {
                    ++found;
                    ++gs_n;
                    ++re_n;
                }
                else {
                    //throw runtime_error("distance is unstable");
                }
            }
            sum += float(found) / K;
        }
        return sum / result.size();
    }


}

#ifndef KGRAPH_NO_VECTORIZE
#ifdef __GNUC__
#ifdef __AVX__
#if 0
namespace kgraph { namespace metric {
        template <>
        inline float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
            return float_l2sqr_avx(t1, t2, dim);
        }
}}
#endif
#else
#ifdef __SSE2__
namespace kgraph { namespace metric {
        template <>
        inline float l2sqr::apply<float> (float const *t1, float const *t2, unsigned dim) {
            return float_l2sqr_sse2(t1, t2, dim);
        }
        template <>
        inline float l2sqr::dot<float> (float const *t1, float const *t2, unsigned dim) {
            return float_dot_sse2(t1, t2, dim);
        }
        template <>
        inline float l2sqr::norm2<float> (float const *t1, unsigned dim) {
            return float_l2sqr_sse2(t1, dim);
        }
        template <>
        inline float l2sqr::apply<uint8_t> (uint8_t const *t1, uint8_t const *t2, unsigned dim) {
            return uint8_l2sqr_sse2(t1, t2, dim);
        }
}}
#endif
#endif
#endif
#endif



#endif

