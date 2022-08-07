#include "benchmark.hpp"
#include "cmdline.h"
#include <omp.h>
#include <chrono>

//#define MAX(a,b) ((a) > (b) ? (a) : (b))
//#define MIN(a,b) ((a) < (b) ? (a) : (b))

#define PAPER_HEIGHT_PATH "/home/jiamian/img/paper/paper_4k_test5.png"
#define PAPER_IMAGE_PATH  "/home/jiamian/img/paper/paper_4k_test5_rendered_modified_2.png"
#define paper_scale 0.37

#define TIME_BEGIN(a) auto time_begin_##a = std::chrono::high_resolution_clock::now()

#define TIME_END(a)   auto time_end_##a = std::chrono::high_resolution_clock::now();\
					  auto elapse_##a = std::chrono::duration_cast<std::chrono::nanoseconds>(time_end_##a - time_begin_##a);\
                      printf("[%s time measured : %.6f s == %.4f ms == %.2f us]\n", #a, elapse_##a.count() * 1e-9, elapse_##a.count() * 1e-6, elapse_##a.count() * 1e-3)

#define CTIME_ENABLE
#ifdef CTIME_ENABLE
    #define CTIME_BEGIN(a) a.begin()
    #define CTIME_END(a) a.end()
#else
    #define CTIME_BEGIN(a)
    #define CTIME_END(a)
#endif

namespace Timer
{
    class Cumulative_Timer
    {
        enum class TimeUnit
        {
            s = 0,
            ms,
            us,
            ns
        };
        public:
        Cumulative_Timer(std::string _name, TimeUnit _unit = TimeUnit::ms) : name(_name), unit(_unit)
        {
            reset();
        }
        void reset()
        {
            cumulative_time = 0.0f;
        }
        void begin()
        {
            time1 = std::chrono::high_resolution_clock::now();
        }
        float end()
        {
            cudaDeviceSynchronize();
            auto time2 = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1);
            float duration_time = dur.count() * powf(0.1, 3 * (3 - (int)unit));
            cumulative_time += duration_time;
            return duration_time;
        }
        void show()
        {
            std::string unit_name = "";
            if(unit == TimeUnit::s) unit_name = "s";
            else if(unit == TimeUnit::ms) unit_name = "ms";
            else if(unit == TimeUnit::us) unit_name = "us";
            else if(unit == TimeUnit::ns) unit_name = "ns";
            printf("[%s used cumulative time = %.4f %s]\n", name.c_str(), cumulative_time, unit_name.c_str());
        }
        float get()
        {
            return cumulative_time;
        }
        private:
        std::string name;
        decltype(std::chrono::high_resolution_clock::now()) time1;
        float cumulative_time;
        TimeUnit unit;
    };
}


#define SWE_dx 1
#define SWE_dy SWE_dx
#define dt_max 0.1
#define CFL_a 2
#define SWE_f 0.4  //0.075
#define SWE_g 9.8
#define SWE_incline_x 0
#define SWE_incline_y 0.00
#define SWE_gamma 0.8
#define SWE_rou   0.4
#define SWE_omega 15
#define dt dt_max

#define SWE_eta_water 0.005  //0.005
#define SWE_eta_water_const 0.1 //0.1
#define SWE_EVAP_FREE 0.0003
#define SWE_EVAP_DAMP 0.00000015

#define ABSORB 0.000085
#define C_MAX  0.01
#define C_MIN  0.0028
#define CAP_epsilon 0.008
#define CAP_delta 0.00001
#define CAP_sigma 0.001

#define WATER_H 0.22  //0.135
#define PIGMENT_THICK 0.5

#define BLOCK 0
#define WATER 1
#define DAMP 2
#define FREE 3

#define SAVE_SIMD 1

struct BGRu8
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
};


#define SAVE(path, img) cv::imwrite(std::string("../output/") + path + ".png", img)
#define DEFAULT_COLOR 255

void ShowBGRImg(int width, int height, BGRu8* img, const std::string& name)
{
	cv::Mat show = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

    #pragma omp parallel for num_threads(32)
    for(int i = 0; i < 32; i++)
    {
        int chunk_size = height * width / 32;
	    memcpy(show.data + i * chunk_size * sizeof(BGRu8), img + i * chunk_size, chunk_size * sizeof(BGRu8));
    }

	SAVE(name,show);
}

void DrawStrokes(benchmark::Benchmark& bench, cv::Mat& mat)
{
    std::vector<Stroke> strokes = bench.GetStrokes();
    mat = cv::Mat::zeros(cv::Size(bench.W(), bench.H()), CV_8UC3);
    mat.setTo(cv::Scalar(DEFAULT_COLOR, DEFAULT_COLOR, DEFAULT_COLOR));

    int stroke_num = strokes.size();
    #pragma omp parallel
    for(int sidx = 0; sidx < stroke_num; sidx++)
    {
        Stroke& stroke = strokes.at(sidx);
        for(auto point : stroke.points)
        {
            cv::circle(mat, point, stroke.r, stroke.color, -1);
        }
    }
}

__global__ void UpdateH(
    int height, 
    int width,
    int tank_dim_x,
    int tank_dim_y,
    float* h_new,
    float* h_old,
    float* u_new,
    float* u_old,
    float* v_new,
    float* v_old,
    unsigned char* cell
);

__global__ void UpdateUV(
    int height, 
    int width,
    int tank_dim_x,
    int tank_dim_y,
    float* h_new,
    float* h_old,
    float* u_new,
    float* u_old,
    float* v_new,
    float* v_old,
    unsigned char* cell,
    float* s,
    float* s_temp,
    float* z
);

__global__ void EvaporateY(
    int height,
    int width,
	unsigned char* water_region_flag,
    float* evap
);

__global__ void EvaporateXAbsorb(
    int height,
    int width,
    unsigned char* cell,
	float* h_new,
	float* s,
	float* z,
    float* evap,
	unsigned char* water_region_flag
);

__global__ void Save(
    int height, 
    int width,
    float* h_new,
    float* h_old,
    float* u_new,
    float* u_old,
    float* v_new,
    float* v_old,
    float* s,
    float* s_temp
);

__global__ void UpdateCell(
    int height,
    int width,
    float* h_new,
    float* s,
    unsigned char* cell,
    unsigned char* water_region_flag
);

#define SWE_TANK_WIDTH 32
#define SWE_TANK_HEIGHT 64
#define SWE_BLOCK_WIDTH 32
#define SWE_BLOCK_HEIGHT 16

#define SAVE_BLOCK_SIZE (32 * 8)
#define SAVE_BLOCK_NUM -1

#define EVAP_BLOCK_WIDTH 32
#define EVAP_BLOCK_HEIGHT 16


int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<int>("width", 'w', "width of canvas", false, 1664);
    cmd.add<int>("height", 'h', "height of canvas", false, 1664);
    cmd.add<int>("stride", 's', "stride of stroke distribution", false, 100);
    cmd.add<int>("len", 'l', "length of each stroke", false, 20);
    cmd.add<int>("radius", 'r', "radius of stroke bursh", false, 50);
    cmd.add<int>("move_step", 'm', "move step of brush", false, 20);
    cmd.add<float>("curl", 'u', "curl of stroke appearance", false, 0.05);
    cmd.add<float>("chaos", 'c', "chaos of stroke appearance", false, 0.5);
    cmd.add<int>("pseudo", 'p', "use pseudo-random benchmark", false, 0);
    cmd.add<int>("gpu_id", 'g', "gpu id", false, 0);
    cmd.add<int>("swe_block_num_want", ' ', "swe_block_num_want", false, 64);
    cmd.add<int>("time", 't', "maximum time", false, 4500);
	cmd.parse_check(argc, argv);

    int width = cmd.get<int>("width");
    int height = cmd.get<int>("height");
    int stride = cmd.get<int>("stride");
    int len = cmd.get<int>("len");
    int radius = cmd.get<int>("radius");
    int step = cmd.get<int>("move_step");
    float curl = cmd.get<float>("curl");
    float chaos = cmd.get<float>("chaos");
	int use_pseudo = cmd.get<int>("pseudo");
    int gpu_id = cmd.get<int>("gpu_id");
    int max_time = cmd.get<int>("time");
    int swe_block_num_want = cmd.get<int>("swe_block_num_want");
	

    benchmark::Benchmark bench1(width, height, stride, len, radius, step, curl, chaos, use_pseudo);
    std::cout<<"generated "<<bench1.StrokeNum()<<" strokes"<<std::endl;

    std::cout<<"use gpu"<<gpu_id<<std::endl;
    cudaSetDevice(gpu_id);

    //warm-up

    TIME_BEGIN(save_stroke_sample);
    cv::Mat stroke_sample;
    DrawStrokes(bench1, stroke_sample);
    SAVE("stroke_sample", stroke_sample);
    TIME_END(save_stroke_sample);

    TIME_BEGIN(prepare_paper);
    cv::Mat paper_height = cv::imread(PAPER_HEIGHT_PATH, cv::IMREAD_GRAYSCALE);
	cv::resize(paper_height, paper_height, cv::Size(paper_height.size().width * paper_scale, paper_height.size().height * paper_scale));
	paper_height = paper_height(cv::Rect(0, 0, width, height));
    SAVE("paper_height",paper_height);
	cv::normalize(paper_height, paper_height, 255, 0, cv::NORM_MINMAX);
    cv::Mat paper_height_norm;
    paper_height.convertTo(paper_height_norm, CV_32FC1, 1 / 255.0);
	TIME_END(prepare_paper);

    unsigned char* cell;
    unsigned char* water_region_flag;
    float* h_new;
    float* h_old;
    float* u_new;
    float* u_old;
    float* v_new;
    float* v_old;
    float* s;
    float* s_temp;
    float* z;
    float* evap1;
    float* evap2;

    cudaMalloc((void**)&cell, height * width * sizeof(unsigned char));
    cudaMalloc((void**)&water_region_flag, height * width * sizeof(unsigned char));
    cudaMalloc((void**)&h_new, height * width * sizeof(float));
    cudaMalloc((void**)&h_old, height * width * sizeof(float));
    cudaMalloc((void**)&u_new, height * (width + 1) * sizeof(float));
    cudaMalloc((void**)&u_old, height * (width + 1) * sizeof(float));
    cudaMalloc((void**)&v_new, (height + 1) * width * sizeof(float));
    cudaMalloc((void**)&v_old, (height + 1) * width * sizeof(float));
    cudaMalloc((void**)&s, height * width * sizeof(float));
    cudaMalloc((void**)&s_temp, height * width * sizeof(float));
    cudaMalloc((void**)&z, height * width * sizeof(float));
    cudaMalloc((void**)&evap1, height * width * sizeof(float));
    cudaMalloc((void**)&evap2, height * width * sizeof(float));
    
    cudaMemcpy(z, ((float*)(paper_height_norm.data)), height * width * sizeof(float), cudaMemcpyHostToDevice);

    float* s_cpu = new float[height * width];
    float* h_cpu = new float[height * width];
    unsigned char* cell_cpu = new unsigned char[height * width];
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            if(powf(j - height / 2, 2) + powf(i - width / 2, 2) <= 800*800)
            {
                h_cpu[j * width + i] = WATER_H;
                cell_cpu[j * width + i] = WATER;
            }
            else
            {
                h_cpu[j * width + i] = 0;
                cell_cpu[j * width + i] = BLOCK;
            }
            s_cpu[j * width + i] = CAP_sigma * 0.6;
        }
    }
    cudaMemcpy(h_old, h_cpu, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s_temp, s_cpu, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cell, cell_cpu, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cv::Mat h_mat = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            h_mat.at<unsigned char>(cv::Point(i, j)) = h_cpu[j * width + i] * 255;
        }
    }
    SAVE("h_old_init", h_mat);


    dim3 swe_block_size(SWE_BLOCK_WIDTH, SWE_BLOCK_HEIGHT);
    int swe_tank_dim_x = (width + SWE_TANK_WIDTH - 1) / SWE_TANK_WIDTH;
    int swe_tank_dim_y = (height + SWE_TANK_HEIGHT - 1) / SWE_TANK_HEIGHT;
    int swe_tank_num = swe_tank_dim_x * swe_tank_dim_y;
    int swe_block_num = min(swe_block_num_want == -1 ? swe_tank_num : swe_block_num_want, swe_tank_num);
    printf("SWE kernel : total=%d, have=%d, real=%d\n", swe_tank_num, swe_block_num_want, swe_block_num);

    int save_block_num_need = (height * width + SAVE_BLOCK_SIZE - 1) / SAVE_BLOCK_SIZE;
    int save_block_num = min(SAVE_BLOCK_NUM == -1 ? save_block_num_need : SAVE_BLOCK_NUM, save_block_num_need);
    printf("Save kernel : total=%d, have=%d, real=%d\n", save_block_num_need, SAVE_BLOCK_NUM, save_block_num);

    dim3 evap_grid_size((width + EVAP_BLOCK_WIDTH - 1) / EVAP_BLOCK_WIDTH, (height + EVAP_BLOCK_HEIGHT - 1) / EVAP_BLOCK_HEIGHT);
    dim3 evap_block_size(EVAP_BLOCK_WIDTH, EVAP_BLOCK_HEIGHT);

    /*unsigned char* water_region_flag_cpu = new unsigned char[width * height];
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            if(powf(i - width / 2, 2) + powf(j - height / 2, 2) <= 600 * 600)
            {
                water_region_flag_cpu[j * width + i] = 1;
            }
            else
            {
                water_region_flag_cpu[j * width + i] = 0;
            }
        }
    }
    cudaMemcpy(water_region_flag, water_region_flag_cpu, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);
    float* debug;
    cudaMalloc((void**)&debug, height * width * sizeof(float));
    
    TIME_BEGIN(gauss);
    EvaporateY<<<evap_grid_size, evap_block_size>>>(
        height,
        width,
        water_region_flag,
        evap1
    );
    EvaporateXAbsorb<<<evap_grid_size, evap_block_size>>>(
        height,
        width,
        cell,
        h_new,
        debug,
        z,
        evap1,
        water_region_flag
    );
    TIME_END(gauss);
    cudaDeviceSynchronize();
    float* debug_cpu = new float[width * height];
    cudaMemcpy(debug_cpu, debug, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat gauss_debug = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            gauss_debug.at<unsigned char>(cv::Point(i, j)) = debug_cpu[j * width + i] * 255;
        }
    }
    SAVE("gauss_debug",gauss_debug);
    delete[] debug_cpu;
    delete[] water_region_flag_cpu;*/

    Timer::Cumulative_Timer ctime_h("update H");
    Timer::Cumulative_Timer ctime_uv("update UV");
    Timer::Cumulative_Timer ctime_evap("evaporation");
    Timer::Cumulative_Timer ctime_save("save t-1");
    Timer::Cumulative_Timer ctime_cell("update cells");

    int t = 0;
    std::cout<<"simulation start"<<std::endl;
    TIME_BEGIN(simulation);
    while(t < max_time)
    {
        //BGRu8* debug;
        //cudaMalloc((void**)&debug, height * width * sizeof(BGRu8));
        //cudaMemset(debug, 0, height * width * sizeof(BGRu8));
        CTIME_BEGIN(ctime_h);
        UpdateH<<<swe_block_num, swe_block_size>>>(
            height,
            width,
            swe_tank_dim_x,
            swe_tank_dim_y,
            h_new,
            h_old,
            u_new,
            u_old,
            v_new,
            v_old,
            cell
        );
        CTIME_END(ctime_h);
        //cudaMemcpy(s_temp, s, height * width * sizeof(float), cudaMemcpyDeviceToDevice);
        CTIME_BEGIN(ctime_uv);
        UpdateUV<<<swe_block_num, swe_block_size>>>(
            height,
            width,
            swe_tank_dim_x,
            swe_tank_dim_y,
            h_new,
            h_old,
            u_new,
            u_old,
            v_new,
            v_old,
            cell,
            s,
            s_temp,
            z
        );
        CTIME_END(ctime_uv);

        CTIME_BEGIN(ctime_evap);
        EvaporateY<<<evap_grid_size, evap_block_size>>>(
            height,
            width,
            water_region_flag,
            evap1
        );
        
        EvaporateXAbsorb<<<evap_grid_size, evap_block_size>>>(
            height,
            width,
            cell,
            h_new,
            s,
            z,
            evap1,
            water_region_flag
        );
        CTIME_END(ctime_evap);

        //cudaMemcpy(h_old, h_new, height * width * sizeof(float), cudaMemcpyDeviceToDevice);
        //cudaMemcpy(u_old, u_new, height * (width + 1) * sizeof(float), cudaMemcpyDeviceToDevice);
        //cudaMemcpy(v_old, v_new, (height + 1) * width * sizeof(float), cudaMemcpyDeviceToDevice);

        CTIME_BEGIN(ctime_save);
        Save<<<save_block_num, SAVE_BLOCK_SIZE>>>(
            height, 
            width,
            h_new,
            h_old,
            u_new,
            u_old,
            v_new,
            v_old,
            s,
            s_temp
        );
        CTIME_END(ctime_save);

        CTIME_BEGIN(ctime_cell);
        UpdateCell<<<evap_grid_size, evap_block_size>>>(
            height,
            width,
            h_new,
            s,
            cell,
            water_region_flag
        );
        CTIME_END(ctime_cell);

        //BGRu8* debug_cpu = new BGRu8[width * height];
        //cudaMemcpy(debug_cpu, debug, height * width * sizeof(BGRu8), cudaMemcpyDeviceToHost);
        //cv::Mat debug_mat = cv::Mat(cv::Size(width, height), CV_8UC3, (unsigned char*)debug_cpu);
        //SAVE("debug", debug_mat);

        t++;
    }
    cudaDeviceSynchronize();
    TIME_END(simulation);
    std::cout<<"simulation end"<<std::endl;

    cudaMemcpy(h_cpu, h_old, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat h_mat2 = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            h_mat2.at<unsigned char>(cv::Point(i, j)) = h_cpu[j * width + i] * 255;
        }
    }
    SAVE("h_old_end", h_mat2);
    
    cudaMemcpy(s_cpu, s_temp, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat s_end = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            if(s_cpu[j * width + i] != 0)
                s_end.at<unsigned char>(cv::Point(i, j)) = s_cpu[j * width + i] * 100000;
        }
    }
    SAVE("s_end", s_end);

    cudaMemcpy(cell_cpu, cell, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cv::Mat cell_end = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
    for(int j = 0; j < height; j++)
    {
        for(int i = 0; i < width; i++)
        {
            if(cell_cpu[j * width + i] == BLOCK)
                cell_end.at<cv::Vec3b>(cv::Point(i, j)) = cv::Vec3b(0,0,0);
            else if(cell_cpu[j * width + i] == WATER)
                cell_end.at<cv::Vec3b>(cv::Point(i, j)) = cv::Vec3b(255,0,0);
            else if(cell_cpu[j * width + i] == DAMP)
                cell_end.at<cv::Vec3b>(cv::Point(i, j)) = cv::Vec3b(255,127,127);
            else if(cell_cpu[j * width + i] == FREE)
                cell_end.at<cv::Vec3b>(cv::Point(i, j)) = cv::Vec3b(127,127,127);
            else
                cell_end.at<cv::Vec3b>(cv::Point(i, j)) = cv::Vec3b(0,0,127);
        }
    }
    SAVE("cell_end", cell_end);

    ctime_h.show();
    ctime_uv.show();
    ctime_evap.show();
    ctime_save.show();
    ctime_cell.show();

    cudaFree(h_new);
    cudaFree(h_old);
    cudaFree(u_new);
    cudaFree(u_old);
    cudaFree(v_new);
    cudaFree(v_old);
    cudaFree(s);
    cudaFree(s_temp);
    cudaFree(z);
    cudaFree(evap1);
    cudaFree(evap2);

    delete[] h_cpu;
    delete[] s_cpu;
    delete[] cell_cpu;

    return 0;
}


__global__ void UpdateH(
    int height, 
    int width,
    int tank_dim_x,
    int tank_dim_y,
    float* h_new,
    float* h_old,
    float* u_new,
    float* u_old,
    float* v_new,
    float* v_old,
    unsigned char* cell
)
{
    int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    //unsigned char rand1 = static_cast<int>(sinf(block_idx) * 10000) % 255;
    //unsigned char rand2 = static_cast<int>(cosf(block_idx) * 10000) % 255;
    //unsigned char rand3 = static_cast<int>(tanf(block_idx) * 10000) % 255;
    for(int tank_idx = block_idx; tank_idx < tank_dim_x * tank_dim_y; tank_idx += gridDim.x)
    {
        int init_grid_idx_x = tank_idx % tank_dim_x * SWE_TANK_WIDTH;
        int init_grid_idx_y = tank_idx / tank_dim_x * SWE_TANK_HEIGHT;
        for(int grid_idx = thread_idx; grid_idx < SWE_TANK_HEIGHT * SWE_TANK_WIDTH; grid_idx += SWE_BLOCK_HEIGHT * SWE_BLOCK_WIDTH)
        {
            int grid_idx_x = init_grid_idx_x + grid_idx % SWE_TANK_WIDTH + 1;  //one offset
            int grid_idx_y = init_grid_idx_y + grid_idx / SWE_TANK_WIDTH + 1;
            if(grid_idx_x < width - 1 && grid_idx_y < height - 1)
            {
                //debug[grid_idx_y * width + grid_idx_x] = {rand1,rand2,rand3};

                //update h_new
                if(cell[grid_idx_y * width + grid_idx_x] == WATER || cell[grid_idx_y * width + grid_idx_x] == FREE)
                {
                    float p_i_p_half_j_old = u_old[grid_idx_y * (width + 1) + grid_idx_x + 1] * (u_old[grid_idx_y * (width + 1) + grid_idx_x + 1] >= 0 ? h_old[grid_idx_y * width + grid_idx_x] : h_old[grid_idx_y * width + grid_idx_x + 1]);
                    float p_i_n_half_j_old = u_old[grid_idx_y * (width + 1) + grid_idx_x] * (u_old[grid_idx_y * (width + 1) + grid_idx_x] >= 0 ? h_old[grid_idx_y * width + grid_idx_x - 1] : h_old[grid_idx_y * width + grid_idx_x]);
                    float q_i_j_p_half_old = v_old[(grid_idx_y + 1) * width + grid_idx_x] * (v_old[(grid_idx_y + 1) * width + grid_idx_x] >= 0 ? h_old[grid_idx_y * width + grid_idx_x] : h_old[(grid_idx_y + 1) * width + grid_idx_x]);
                    float q_i_j_n_half_old = v_old[grid_idx_y * width + grid_idx_x] * (v_old[grid_idx_y * width + grid_idx_x] >= 0 ? h_old[(grid_idx_y - 1) * width + grid_idx_x] : h_old[grid_idx_y * width + grid_idx_x]);
                    h_new[grid_idx_y * width + grid_idx_x] = h_old[grid_idx_y * width + grid_idx_x] - dt / SWE_dx * (p_i_p_half_j_old - p_i_n_half_j_old) - dt / SWE_dy * (q_i_j_p_half_old - q_i_j_n_half_old);
                }
            }
        }
    }
}

__global__ void UpdateUV(
    int height, 
    int width,
    int tank_dim_x,
    int tank_dim_y,
    float* h_new,
    float* h_old,
    float* u_new,
    float* u_old,
    float* v_new,
    float* v_old,
    unsigned char* cell,
    float* s,
    float* s_temp,
    float* z
)
{
    //int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    //int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    for(int tank_idx = blockIdx.y * gridDim.x + blockIdx.x; tank_idx < tank_dim_x * tank_dim_y; tank_idx += gridDim.x)
    {
        for(int grid_idx = threadIdx.y * blockDim.x + threadIdx.x; grid_idx < SWE_TANK_HEIGHT * SWE_TANK_WIDTH; grid_idx += SWE_BLOCK_HEIGHT * SWE_BLOCK_WIDTH)
        {
            int grid_idx_x = tank_idx % tank_dim_x * SWE_TANK_WIDTH + grid_idx % SWE_TANK_WIDTH + 1;
            int grid_idx_y = tank_idx / tank_dim_x * SWE_TANK_HEIGHT + grid_idx / SWE_TANK_WIDTH + 1;
            if(grid_idx_x < width - 1 && grid_idx_y < height - 1)
            {
                unsigned char c1 = cell[grid_idx_y * width + grid_idx_x];
                unsigned char c2 = cell[grid_idx_y * width + grid_idx_x - 1];
                unsigned char c3 = cell[(grid_idx_y - 1) * width + grid_idx_x];
                unsigned char update_flag = 0x03;  // u,v
                if(c1 == BLOCK || c1 == DAMP)
                {
                    u_new[grid_idx_y * (width + 1) + grid_idx_x] = 0;
                    v_new[grid_idx_y * width + grid_idx_x] = 0;
                    update_flag &= ~0x03;
                }
                else
                {
                    if(c2 == BLOCK || c2 == DAMP)
                    {
                        u_new[grid_idx_y * (width + 1) + grid_idx_x] = 0;
                        update_flag &= ~0x02;
                    }
                    if(c3 == BLOCK || c3 == BLOCK)
                    {
                        v_new[grid_idx_y * width + grid_idx_x] = 0;
                        update_flag &= ~0x01;
                    }
                }

                if(grid_idx_x == 1)
                    update_flag &= ~0x02;
                if(grid_idx_y == 1)
                    update_flag &= ~0x01;

                if(update_flag)
                {
                    float h1 = h_old[grid_idx_y * width + grid_idx_x];
                    float h2 = h_old[grid_idx_y * width + grid_idx_x - 1];
                    float h3 = h_old[(grid_idx_y - 1) * width + grid_idx_x];
                    float h4 = h_old[(grid_idx_y - 1) * width + grid_idx_x - 1];
                    float h5 = h_old[(grid_idx_y + 1) * width + grid_idx_x];
                    float h6 = h_old[grid_idx_y * width + grid_idx_x + 1];
                    float u1 = u_old[grid_idx_y * (width + 1) + grid_idx_x];
                    float u2 = u_old[grid_idx_y * (width + 1) + grid_idx_x + 1];
                    float u3 = u_old[(grid_idx_y - 1) * (width + 1) + grid_idx_x];
                    float v1 = v_old[grid_idx_y * width + grid_idx_x];
                    float v2 = v_old[(grid_idx_y + 1) * width + grid_idx_x];
                    float v3 = v_old[grid_idx_y * width + grid_idx_x - 1];

                    float h1n = h_new[grid_idx_y * width + grid_idx_x];
                    float h2n = h_new[grid_idx_y * width + grid_idx_x - 1];
                    float h3n = h_new[(grid_idx_y - 1) * width + grid_idx_x];

                    if(update_flag & 0x02) //update u
                    {
                        if(((h1n + h2n) / 2.0f) == 0)
                        {
                            u_new[grid_idx_y * (width + 1) + grid_idx_x] = 0;
                        }
                        else
                        {
                            float q_i_p_1_j_p_half_old = v2 * (v2 >= 0 ? h1 : h5);
                            float q_i_j_p_half_old = v_old[(grid_idx_y + 1) * width + grid_idx_x - 1] * (v_old[(grid_idx_y + 1) * width + grid_idx_x - 1] >= 0 ? h2 : h_old[(grid_idx_y + 1) * width + grid_idx_x - 1]);
                            float q_i_p_1_j_n_half_old = v1 * (v1 >=0 ? h3 : h1);
                            float q_i_j_n_half_old = v3 * (v3 >= 0 ? h4 : h2);
                            float q_i_p_half_j_p_half_old = (q_i_p_1_j_p_half_old + q_i_j_p_half_old) / 2;
						    float q_i_p_half_j_n_half_old = (q_i_p_1_j_n_half_old + q_i_j_n_half_old) / 2;
                            float quy = q_i_p_half_j_p_half_old * (q_i_p_half_j_p_half_old >= 0 ? u1 : u_old[(grid_idx_y + 1) * (width + 1) + grid_idx_x])
							          - q_i_p_half_j_n_half_old * (q_i_p_half_j_n_half_old >= 0 ? u3 : u1);

                            float p_i_p_3half_j_old = u2 * (u2 >= 0 ? h1 : h6);
                            float p_i_p_half_j_old = u1 * (u2 >= 0 ? h2 : h1);
                            float p_i_n_half_j_old = u_old[grid_idx_y * (width + 1) + grid_idx_x - 1] * (u_old[grid_idx_y * (width + 1) + grid_idx_x - 1] >= 0 ? h_old[grid_idx_y * width + grid_idx_x - 2] : h2);
                            float p_i_p_1_j_old = (p_i_p_3half_j_old + p_i_p_half_j_old) / 2;
                            float p_i_j_old = (p_i_p_half_j_old + p_i_n_half_j_old) / 2;
                            float pux = p_i_p_1_j_old * (p_i_p_1_j_old >= 0 ? u1 : u2)
                                      - p_i_j_old * (p_i_j_old >= 0 ? u_old[grid_idx_y * (width + 1) + grid_idx_x - 1] : u1);

						    u_new[grid_idx_y * (width + 1) + grid_idx_x] = (
                                                            ((h2 + h1) / 2) * u1 
                                                            - dt / SWE_dy * quy 
                                                            - dt / SWE_dx * (pux + 0.5 * SWE_g * (powf(h1n, 2) - powf(h2n, 2))) 
                                                            - dt / SWE_dx * SWE_g * ((h1n + h2n) / 2.0f) * SWE_incline_x
                                                        ) / (((h1n + h2n) / 2.0f) + dt * SWE_f / 8 * abs(u1));
                        }
                    }

                    if(update_flag & 0x01) //update v
                    {
                        if(((h1n + h3n) / 2.0f) == 0)
                        {
                            v_new[grid_idx_y * width + grid_idx_x] = 0;
                        }
                        else
                        {
                            float p_i_p_half_j_p_1_old = u2 * (u2 >= 0 ? h1 : h6);
						    float p_i_p_half_j_old = u_old[(grid_idx_y - 1) * (width + 1) + grid_idx_x + 1] * (u_old[(grid_idx_y - 1) * (width + 1) + grid_idx_x + 1] >= 0 ? h3 : h_old[(grid_idx_y - 1) * width + grid_idx_x + 1]);
						    float p_i_n_half_j_p_1_old = u1 * (u1 >= 0 ? h2 : h1);
						    float p_i_n_half_j_old = u3 * (u3 >= 0 ? h4 : h3);
						    float p_i_p_half_j_p_half_old = (p_i_p_half_j_p_1_old + p_i_p_half_j_old) / 2;
						    float p_i_n_half_j_p_half_old = (p_i_n_half_j_p_1_old + p_i_n_half_j_old) / 2;
                            float pvx = p_i_p_half_j_p_half_old * (p_i_p_half_j_p_half_old >= 0 ? v1 : v_old[grid_idx_y * width + grid_idx_x + 1])
							          - p_i_n_half_j_p_half_old * (p_i_n_half_j_p_half_old >= 0 ? v3 : v1);
                            
                            float q_i_j_p_3half_old = v2 * (v2 >= 0 ? h1 : h5);
						    float q_i_j_p_half_old = v1 * (v1 >= 0 ? h3 : h1);
						    float q_i_j_n_half_old = v_old[(grid_idx_y - 1) * width + grid_idx_x] * (v_old[(grid_idx_y - 1) * width + grid_idx_x] >= 0 ? h_old[(grid_idx_y - 2) * width + grid_idx_x] : h3);
						    float q_i_j_p_1_old = (q_i_j_p_3half_old + q_i_j_p_half_old) / 2;
						    float q_i_j_old = (q_i_j_p_half_old + q_i_j_n_half_old) / 2;
						    float qvy = q_i_j_p_1_old * (q_i_j_p_1_old >= 0 ? v1 : v2)
							          - q_i_j_old * (q_i_j_old >= 0 ? v_old[(grid_idx_y - 1) * width + grid_idx_x] : v1);

                            v_new[grid_idx_y * width + grid_idx_x] = (
                                                ((h3 + h1) / 2) * v1
                                                    - dt / SWE_dx * pvx 
                                                    - dt / SWE_dy * (qvy + 0.5 * SWE_g * (powf(h1n, 2) - powf(h3n, 2)))
                                                    - dt / SWE_dy * SWE_g * ((h1n + h3n) / 2.0f) * SWE_incline_y
                                                ) / (((h1n + h3n) / 2.0f) + dt * SWE_f / 8 * abs(v1));
                        }
                    }
                }

                //update s
                float s_this = s_temp[grid_idx_y * width + grid_idx_x];
                float s_that = s_temp[(grid_idx_y - 1) * width + grid_idx_x];
                float z_this = z[grid_idx_y * width + grid_idx_x];
                float z_that = z[(grid_idx_y - 1) * width + grid_idx_x];
                float s_this_new = s_this;
                if(s_this >= CAP_epsilon && s_this > s_that && s_that >= CAP_delta)
                {
                    s_this_new -= MAX(0, MIN(s_this - s_that, z_that * (C_MAX - C_MIN) + C_MIN - s_that) / 4.0);
                }
                else if(s_that >= CAP_epsilon && s_that > s_this && s_this >= CAP_delta)
                {
                    s_this_new += MAX(0, MIN(s_that - s_this, z_this * (C_MAX - C_MIN) + C_MIN - s_this) / 4.0);
                }
                s_that = s_temp[(grid_idx_y + 1) * width + grid_idx_x];
                z_that = z[(grid_idx_y + 1) * width + grid_idx_x];
                if(s_this >= CAP_epsilon && s_this > s_that && s_that >= CAP_delta)
                {
                    s_this_new -= MAX(0, MIN(s_this - s_that, z_that * (C_MAX - C_MIN) + C_MIN - s_that) / 4.0);
                }
                else if(s_that >= CAP_epsilon && s_that > s_this && s_this >= CAP_delta)
                {
                    s_this_new += MAX(0, MIN(s_that - s_this, z_this * (C_MAX - C_MIN) + C_MIN - s_this) / 4.0);
                }
                s_that = s_temp[grid_idx_y * width + grid_idx_x - 1];
                z_that = z[grid_idx_y * width + grid_idx_x - 1];
                if(s_this >= CAP_epsilon && s_this > s_that && s_that >= CAP_delta)
                {
                    s_this_new -= MAX(0, MIN(s_this - s_that, z_that * (C_MAX - C_MIN) + C_MIN - s_that) / 4.0);
                }
                else if(s_that >= CAP_epsilon && s_that > s_this && s_this >= CAP_delta)
                {
                    s_this_new += MAX(0, MIN(s_that - s_this, z_this * (C_MAX - C_MIN) + C_MIN - s_this) / 4.0);
                }
                s_that = s_temp[grid_idx_y * width + grid_idx_x + 1];
                z_that = z[grid_idx_y * width + grid_idx_x + 1];
                if(s_this >= CAP_epsilon && s_this > s_that && s_that >= CAP_delta)
                {
                    s_this_new -= MAX(0, MIN(s_this - s_that, z_that * (C_MAX - C_MIN) + C_MIN - s_that) / 4.0);
                }
                else if(s_that >= CAP_epsilon && s_that > s_this && s_this >= CAP_delta)
                {
                    s_this_new += MAX(0, MIN(s_that - s_this, z_this * (C_MAX - C_MIN) + C_MIN - s_this) / 4.0);
                }
                s[grid_idx_y * width + grid_idx_x] = s_this_new;

                //debug[grid_idx_y * width + grid_idx_x] = {255,0,0};
            }
        }
    }   
}

__global__ void EvaporateY(
    int height,
    int width,
	unsigned char* water_region_flag,
    float* evap
)
{
    __shared__ float evap1[EVAP_BLOCK_WIDTH + 8][EVAP_BLOCK_HEIGHT + 8];

    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    
    for(int i = thread_idx; i < (EVAP_BLOCK_WIDTH + 8) * (EVAP_BLOCK_HEIGHT + 8); i += EVAP_BLOCK_WIDTH * EVAP_BLOCK_HEIGHT)
    {
        if(i < (EVAP_BLOCK_WIDTH + 8) * (EVAP_BLOCK_HEIGHT + 8))
        {
            int pixel_idx_x = blockIdx.x * blockDim.x - 4 + i % (EVAP_BLOCK_WIDTH + 8);
            int pixel_idx_y = (blockIdx.y * blockDim.y - 4 + i / (EVAP_BLOCK_WIDTH + 8));
            if(pixel_idx_x >= 0 && pixel_idx_x <= width - 1 && pixel_idx_y >= 0 && pixel_idx_y <= height - 1)
                evap1[i / (EVAP_BLOCK_WIDTH + 8)][i % (EVAP_BLOCK_WIDTH + 8)] = (float)water_region_flag[pixel_idx_y * width + pixel_idx_x];
            else
                evap1[i / (EVAP_BLOCK_WIDTH + 8)][i % (EVAP_BLOCK_WIDTH + 8)] = 0.0f;
        }
    }
    __syncthreads();
    evap[(blockIdx.y * blockDim.y + threadIdx.y) * width + blockIdx.x * blockDim.x + threadIdx.x] = 
                                            evap1[threadIdx.y + 4 - 4][threadIdx.x + 4] * 0.1109038212717001f +
                                            evap1[threadIdx.y + 4 - 3][threadIdx.x + 4] * 0.1110591953579631f + 
                                            evap1[threadIdx.y + 4 - 2][threadIdx.x + 4] * 0.1111703101014332f +
                                            evap1[threadIdx.y + 4 - 1][threadIdx.x + 4] * 0.1112370323021526f +
                                            evap1[threadIdx.y + 4    ][threadIdx.x + 4] * 0.1112592819335020f + 
                                            evap1[threadIdx.y + 4 + 1][threadIdx.x + 4] * 0.1112370323021526f + 
                                            evap1[threadIdx.y + 4 + 2][threadIdx.x + 4] * 0.1111703101014332f + 
                                            evap1[threadIdx.y + 4 + 3][threadIdx.x + 4] * 0.1110591953579631f +
                                            evap1[threadIdx.y + 4 + 4][threadIdx.x + 4] * 0.1109038212717001f;
}

__global__ void EvaporateXAbsorb(
    int height,
    int width,
    unsigned char* cell,
	float* h_new,
	float* s,
	float* z,
    float* evap,
	unsigned char* water_region_flag
)
{
    __shared__ float evap1[EVAP_BLOCK_WIDTH + 8][EVAP_BLOCK_HEIGHT + 8];

    int pixel_idx = (blockIdx.y * blockDim.y + threadIdx.y) * width + blockIdx.x * blockDim.x + threadIdx.x; 
    int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    
    for(int i = thread_idx; i < (EVAP_BLOCK_WIDTH + 8) * (EVAP_BLOCK_HEIGHT + 8); i += EVAP_BLOCK_WIDTH * EVAP_BLOCK_HEIGHT)
    {
        if(i < (EVAP_BLOCK_WIDTH + 8) * (EVAP_BLOCK_HEIGHT + 8))
        {
            int pixel_idx_x = blockIdx.x * blockDim.x - 4 + i % (EVAP_BLOCK_WIDTH + 8);
            int pixel_idx_y = (blockIdx.y * blockDim.y - 4 + i / (EVAP_BLOCK_WIDTH + 8));
            if(pixel_idx_x >= 0 && pixel_idx_x <= width - 1 && pixel_idx_y >= 0 && pixel_idx_y <= height - 1)
                evap1[i / (EVAP_BLOCK_WIDTH + 8)][i % (EVAP_BLOCK_WIDTH + 8)] = evap[pixel_idx_y * width + pixel_idx_x];
            else
                evap1[i / (EVAP_BLOCK_WIDTH + 8)][i % (EVAP_BLOCK_WIDTH + 8)] = 0.0f;
        }
    }
    __syncthreads();
    float evap_this = evap1[threadIdx.y + 4][threadIdx.x + 4 - 4] * 0.1109038212717001f +
                      evap1[threadIdx.y + 4][threadIdx.x + 4 - 3] * 0.1110591953579631f + 
                      evap1[threadIdx.y + 4][threadIdx.x + 4 - 2] * 0.1111703101014332f +
                      evap1[threadIdx.y + 4][threadIdx.x + 4 - 1] * 0.1112370323021526f +
                      evap1[threadIdx.y + 4][threadIdx.x + 4    ] * 0.1112592819335020f + 
                      evap1[threadIdx.y + 4][threadIdx.x + 4 + 1] * 0.1112370323021526f + 
                      evap1[threadIdx.y + 4][threadIdx.x + 4 + 2] * 0.1111703101014332f + 
                      evap1[threadIdx.y + 4][threadIdx.x + 4 + 3] * 0.1110591953579631f +
                      evap1[threadIdx.y + 4][threadIdx.x + 4 + 4] * 0.1109038212717001f;
    evap_this = (1 - evap_this) * water_region_flag[pixel_idx];
    evap_this = h_new[pixel_idx] * SWE_eta_water * (evap_this * (1 - SWE_eta_water_const) + SWE_eta_water_const);
    unsigned char this_cell = cell[pixel_idx];
    if(this_cell == WATER)
    {
        float ds = ABSORB / (evap_this + ABSORB) * MIN(evap_this + ABSORB, h_new[pixel_idx]);
        s[pixel_idx] += MIN(z[pixel_idx] * (C_MAX - C_MIN) + C_MIN - s[pixel_idx], ds);
        h_new[pixel_idx] -= MIN(evap_this + ABSORB, h_new[pixel_idx]);
    }
    else if(this_cell == FREE)
    {
        s[pixel_idx] -= MIN(s[pixel_idx], SWE_EVAP_FREE); 
    }
    else if(this_cell == DAMP)
    {
        s[pixel_idx] -= MIN(s[pixel_idx], SWE_EVAP_DAMP); 
    }
}

__global__ void Save(
    int height, 
    int width,
    float* h_new,
    float* h_old,
    float* u_new,
    float* u_old,
    float* v_new,
    float* v_old,
    float* s,
    float* s_temp
)
{
#if SAVE_SIMD
    int max_size = max(height * (width + 1), (height + 1) * width);
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < max_size / 4; idx += gridDim.x * blockDim.x)
    {
        if(idx < height * width / 4)
        {
            ((float4*)h_old)[idx] = ((float4*)h_new)[idx];
            ((float4*)s_temp)[idx] = ((float4*)s)[idx];
        }
        if(idx < height * (width + 1) / 4)
        {
            ((float4*)u_old)[idx] = ((float4*)u_new)[idx];
        }
        if(idx < (height + 1) * width / 4)
        {
            ((float4*)v_old)[idx] = ((float4*)v_new)[idx];
        }
    }
#else
    int max_size = max(height * (width + 1), (height + 1) * width);
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < max_size; idx += gridDim.x * blockDim.x)
    {
        if(idx < height * width)
        {
            h_old[idx] = h_new[idx];
            s_temp[idx] = s[idx];
        }
        if(idx < height * (width + 1))
            u_old[idx] = u_new[idx];
        if(idx < (height + 1) * width)
            v_old[idx] = v_new[idx];
    }
#endif
}

__global__ void UpdateCell(
    int height,
    int width,
    float* h_new,
    float* s,
    unsigned char* cell,
    unsigned char* water_region_flag
)
{
    int pixel_idx = (blockIdx.y * blockDim.y + threadIdx.y) * width + blockIdx.x * blockDim.x + threadIdx.x; 

    float h_this = h_new[pixel_idx];
    float s_this = s[pixel_idx];

    if(h_this == 0 && s_this < CAP_delta)
    {
        cell[pixel_idx] = BLOCK;
        water_region_flag[pixel_idx] = 0;
    }
    else if(h_this != 0)
    {
        cell[pixel_idx] = WATER;
        water_region_flag[pixel_idx] = 1;
    }
    else if(h_this == 0.0f && s_this >= CAP_sigma)
    {
        cell[pixel_idx] = FREE;
        water_region_flag[pixel_idx] = 0;
    }
    else if(h_this == 0.0f && s_this >= CAP_delta && s_this < CAP_sigma)
    {
        cell[pixel_idx] = DAMP;
        water_region_flag[pixel_idx] = 0;
    }
}