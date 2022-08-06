#ifndef H_BENCHMARK
#define H_BENCHMARK

#include <vector>
#include <opencv2/opencv.hpp>
#include <random>
#include <math.h>

struct Stroke
{
	int r;
	cv::Vec3b color;
	std::vector<cv::Point> points;
};

namespace benchmark
{
    class Benchmark
    {
        public:
            Benchmark(int w, int h, int stride, int len, int r, int step, float curl, float chaos, int use_pseudo) : width(w), height(h)
            {
                printf("generate benchmark[size = %d * %d, stride = %d, len = %2d, radius = %d, chaos = %.2f]\n", w, h, stride, len, r, chaos);

                std::mt19937 mt;
                if(use_pseudo)
                {
                    mt = std::mt19937(use_pseudo);
                }
                else
                {
                    std::random_device rd;
                    mt = std::mt19937(rd());
                }

                std::default_random_engine generator(0);
                std::normal_distribution<double> curl_dis(curl, 0.01);
                std::normal_distribution<double> chaos_dis(0.0, chaos);
                for(int j = r; j < h; j += stride)
                {
                    for(int i = r; i < w; i += stride)
                    {
                        int x = i;
                        int y = j;
                        Stroke stroke;
                        stroke.r = r;
                        stroke.color = cv::Vec3b(mt() % 255, mt() % 255, mt() % 255);
                        float base_dir = (static_cast<float>(mt() % 1000) / 500.0f - 1.0f ) * M_PI;
                        float curl_dir = curl_dis(generator);

                        for(int k = 0; k < len; k++)
                        {
                            if(x < r || y < r || x > w - 1 - r || y > h - 1 - r)
                                break;
                            
                            stroke.points.push_back(cv::Point(x,y));
                            
                            x += step * cos(base_dir);
                            y += step * sin(base_dir);
                            base_dir += curl_dir + chaos_dis(generator);
                        }
                        if(stroke.points.size() >= 2)
                        {
                            strokes.push_back(stroke);
                        }
                    }
                }
            }

            std::vector<Stroke>& GetStrokes()
            {
                return strokes;
            }

            int W()
            {
                return width;
            }

            int H()
            {
                return height;
            }

            int StrokeNum()
            {
                return strokes.size();
            }
        private:
            std::vector<Stroke> strokes;
            int width;
            int height;
    };
}

#endif