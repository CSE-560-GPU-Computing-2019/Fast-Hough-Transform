#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "arraylist.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

#define PI 3.14159265
#define imgchannels 3

#define KERNEL_COLUMNS 5
#define KERNEL_ROWS 5
#define BLOCK_SIZE_2D 32
#define BLOCK_SIZE_3D 8
#define SHARED_MEM_WIDTH (BLOCK_SIZE_2D + KERNEL_COLUMNS)

using namespace std;
// __device__ ArrayList* rtable[180];
double* rtable[181];
int pointer[181];


void convert_to_grayscale(const unsigned char*inputImage, unsigned char * outputImageData, int height, int width, int channels)
{
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j)
      {
        float R = (float) inputImage[(width * i + j)*channels + 0];
        float G = (float) inputImage[(width * i + j)*channels + 1];
        float B = (float) inputImage[(width * i + j)*channels + 2];
        float gray_val = 0.299*R + 0.587*G + 0.114*B;
        outputImageData[(width * i + j)] = gray_val;
      }
    } 
}

__global__ void grayscaleKernel(const unsigned char*inputImage, unsigned char * outputImageData, int height, int width, int channels)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i>=0 && i<height && j>=0 && j<width)
  {
    float R = (float) inputImage[(width * i + j)*channels + 0];
    float G = (float) inputImage[(width * i + j)*channels + 1];
    float B = (float) inputImage[(width * i + j)*channels + 2];
    float gray_val = 0.299*R + 0.587*G + 0.114*B;
    outputImageData[(width * i + j)] = gray_val;
  }
}

__global__ void thresholdKernel(const unsigned char*InputImageData, unsigned char*OutputImageData, int height, int width, int channels, int threshold)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i>=0 && i<height && j>=0 && j<width)
  {
    for(int k=0; k<channels; k++)
    {
      float val = (float) InputImageData[(width * i + j)*channels + k];
      if(val>=threshold)
      {
        OutputImageData[(width * i + j)*channels + k] = 255;
      }
      else
      {
        OutputImageData[(width * i + j)*channels + k] = 0;
      }
    }
  }
}

__global__ void sobelKernel(const unsigned char*InputImageData, unsigned char * outputImageData, unsigned int * outputGradient, int height, int width)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  int allHorizontalX[2] = {-1, 1}; 
  int allHorizontalY[3] = {-1, 0, 1};
  int allCoefficientsX[6] = {-1, -2, -1, 1, 2, 1};
    
  int allVerticalX[3] = {-1, 0, 1}; 
  int allVerticalY[2] = {-1, 1};
  int allCoefficientsY[6] = {1, -1, 2, -2, 1, -1};

  if(x>=0 && x<width && y>=0 && y<height)
    {
        float h = 0;
        int k = 0;
        for(int i=0;i<2;i+=1)
        {
          for(int j=0;j<3;j+=1)
          {
            int index = (y + allHorizontalY[j]) * width + (x + allHorizontalX[i]);
            h += allCoefficientsX[k] * InputImageData[index];
            k+=1;
          }
        }

        float v = 0;
        k = 0;
        for(int i=0;i<3;i+=1)
        {
          for(int j=0;j<2;j+=1)
          {
            int index = (y + allVerticalY[j]) * width + (x + allVerticalX[i]);
            v += allCoefficientsY[k] * InputImageData[index];
            k+=1;
          }
        }

        h /= 5;
        v /= 5;
        float val = (float)sqrt((h*h) + (v*v));
        float gradient = atan(v / (h+0.000001));
        
        outputImageData[y*width+x] = val;
        outputGradient[y*width+x] = (gradient * 180 / PI) + 90;
    }
}

__global__ void convKernel(unsigned char*inputImage, float * kernel, unsigned char * outputImageData, int image_height, int image_width) {
    __shared__ float shared_mem[SHARED_MEM_WIDTH][SHARED_MEM_WIDTH];
    __shared__ float shared_kernel[KERNEL_ROWS * KERNEL_COLUMNS];

    int half_kernel = (int) (KERNEL_ROWS - 1) / 2;

    if (threadIdx.x == 0 and threadIdx.y == 0) {
        for (int i = 0; i < KERNEL_COLUMNS * KERNEL_ROWS; ++i)
            shared_kernel[i] = kernel[i];   
        int block_coord_x = blockDim.x * blockIdx.x;
        int block_coord_y = blockDim.y * blockIdx.y;
        int src_x, src_y;
        for (int i=0; i<SHARED_MEM_WIDTH; i++) {
            for (int j=0; j<SHARED_MEM_WIDTH; j++) {
                src_x = block_coord_x - half_kernel + i;
                src_y = block_coord_y - half_kernel + j;
                if(src_y >= 0 && src_y < image_height && src_x >=0 && src_x < image_width) {
                    shared_mem[j][i] = inputImage[src_x + image_width * src_y];
                } else {
                    shared_mem[j][i] = 0;
                }
            }
        }
    }

    __syncthreads();

    float sum = 0;
    int y, x;

    for (y= 0; y < KERNEL_COLUMNS; y++)
        for(x = 0; x<KERNEL_ROWS; x++)
            sum += shared_mem[threadIdx.y + y][threadIdx.x + x] * shared_kernel[y * KERNEL_COLUMNS + x];

    y = blockIdx.y * blockDim.y + threadIdx.y;
    x = blockIdx.x * blockDim.x + threadIdx.x;

    if(y < image_height && x < image_width)
        outputImageData[y * image_width + x] = sum;

}

float * get_avg_kernel() {
  float kernel_data[KERNEL_ROWS*KERNEL_COLUMNS];
  for(int i=0; i< KERNEL_ROWS*KERNEL_COLUMNS; i++){
      kernel_data[i] = 1.0/(KERNEL_ROWS*KERNEL_COLUMNS);
  }
  float * h_kernel_data = (float *) malloc(KERNEL_ROWS*KERNEL_COLUMNS*sizeof(float));
  memcpy(h_kernel_data, kernel_data, KERNEL_ROWS*KERNEL_COLUMNS*sizeof(float));
  return h_kernel_data;
}

__global__ void rtable_init_kernel(double** rtable, int *pointer, int initSize) {

  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  rtable[tidx] = (double*) malloc(initSize * sizeof(double));
  memset(rtable[tidx], 0, initSize * sizeof(double));
  pointer[tidx] = 0;
}

__global__ void rtable_create_kernel(double**rtable, int* pointer, unsigned char * template_edge, unsigned int * template_orientation, int height, int width, double centroid_x, double centroid_y) {
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int orientation = template_orientation[tidx];
  unsigned int edge = template_edge[tidx];
  if (edge != 0) {
    rtable[orientation][pointer[orientation]] = (tidx % width) - centroid_x;
    rtable[orientation][pointer[orientation]+1] = (tidx / width) - centroid_y;
    pointer[orientation]+=2;
  }
}

__global__ void rtable_print_kernel(double** rtable, int* pointer) {
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  for(int index =0; index<pointer[tidx]/2; index++) {
    printf("(%.3f, %.3f) ", rtable[tidx][2*index], rtable[tidx][2*index+1]);
  }
  printf("\n");
  }

__global__ void accumulator_general(unsigned char* image, int * accum, int height, int width, int num_angle, int* pointer, double** rtable) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int theta = blockIdx.z * blockDim.z + threadIdx.z;

  if(image[y*width+x] != 0) {
    double *list = rtable[theta];
    for(int index =0; index<pointer[theta]/2; index++) {
      float vec_x = list[2*index];
      float vec_y = list[2*index+1];
      int x0 = (int) (x + vec_x);
      int y0 = (int) (y + vec_y);
      if (x0 >= 0 and x0 < width && y0 >=0 && y0 < height) {
        atomicAdd(&accum[(y0*width+x0)*num_angle+theta], 1);
      }
    }
  }
}


__global__ void accumulator_circle(unsigned char* image, int * accum, int height, int width, int num_radii) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int r = blockIdx.z * blockDim.z + threadIdx.z;

  if(image[y*width+x] != 0) {
    for(int theta =0; theta<360; theta++) {
      float r_cos_theta = r * cos((float) theta * PI/180);
      float r_sin_theta = r * sin((float) theta * PI/180);
      int x0 = (int) (x - r_cos_theta);
      int y0 = (int) (y - r_sin_theta);
      if (x0 >= 0 and x0 < width && y0 >=0 && y0 < height) {
        atomicAdd(&accum[(y0*width+x0)*num_radii+r], 1);
      }
    }
  }
}

unsigned char* get_image_accum_radii(int * accum, int height, int width, int radii, int num_radii) {
  unsigned char * image = (unsigned char *) malloc(width*height*sizeof(unsigned char));
  for (int y=0; y<height; y++) {
    for (int x=0; x<width; x++) {
      image[y*width+x] = (unsigned char) accum[(y*width+x)*num_radii+radii];
    }
  }
  return image;
}

__global__ void threshold(int * accum, int height, int width, int num_radii, int threshold_val) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int r = blockIdx.z * blockDim.z + threadIdx.z;

  if(accum[(y*width+x)*num_radii+r] < threshold_val) {
    accum[(y*width+x)*num_radii+r] = 0;
  }
}

int main()
{

  cudaEvent_t start_kernel, stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);


  int width, height, bpp;

  string filename = "input3.png";
  string outname = "output/hi.png";
  unsigned char *img;
  const unsigned char* image = stbi_load(filename.c_str(), &width, &height, &bpp, imgchannels);
  img = (unsigned char*)malloc(width*height*sizeof(unsigned char));
  cout << height << " " << width << " " << bpp << endl;
  convert_to_grayscale(image, img, height, width, imgchannels);
  stbi_write_png(outname.c_str(), width, height, 1, img, 0);

  /**********************CUDA-CODE*************************************/
  string gpu_out_edge = "output/hi_gpu_edge.png";
  string gpu_out_edge_thresholded = "output/hi_gpu_edge_thresholded.png";
  // string gpu_out_orientation = "output/hi_gpu_orientation.png";
  string gpu_blurred_image = "output/hi_gpu_blurred_image.png";
  string gpu_accum_image = "output/hi_gpu_accum_image.png";

  unsigned char * d_input;
  cudaMalloc(&d_input, width*height*imgchannels*sizeof(unsigned char));
  cudaMemcpy(d_input, image, width*height*imgchannels*sizeof(unsigned char), cudaMemcpyHostToDevice);

  unsigned char * d_output;
  cudaMalloc(&d_output, width*height*sizeof(unsigned char));

  unsigned char * d_output_edge;
  cudaMalloc(&d_output_edge, width*height*sizeof(unsigned char));

  unsigned char * d_output_edge_thresholded;
  cudaMalloc(&d_output_edge_thresholded, width*height*sizeof(unsigned char));

  unsigned int * d_output_orientation;
  cudaMalloc(&d_output_orientation, width*height*sizeof(unsigned int));

  unsigned char * d_blurred_image;
  cudaMalloc(&d_blurred_image, width*height*sizeof(unsigned char));

  float * d_kernel_data;
  cudaMalloc(&d_kernel_data, KERNEL_ROWS*KERNEL_COLUMNS*sizeof(float));
  cudaMemcpy(d_kernel_data, get_avg_kernel(), KERNEL_ROWS*KERNEL_COLUMNS*sizeof(float), cudaMemcpyHostToDevice);

  const dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
  const int bx = ceil( (float) width/BLOCK_SIZE_2D);
  const int by = ceil( (float) height/BLOCK_SIZE_2D);
  const dim3 gridSize = dim3(bx, by);

  grayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, height, width, imgchannels);
  convKernel<<<gridSize, blockSize>>>(d_output, d_kernel_data, d_blurred_image, height, width);
  sobelKernel<<<gridSize, blockSize>>>(d_blurred_image, d_output_edge, d_output_orientation, height, width);
  thresholdKernel<<<gridSize, blockSize>>>(d_output_edge, d_output_edge_thresholded, height, width, 1, 10);

  int num_radii = 100;

  const dim3 blockSize_accum(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);
  const int bx_accum = ceil( (float) width/BLOCK_SIZE_3D);
  const int by_accum = ceil( (float) height/BLOCK_SIZE_3D);
  const int bz_accum = ceil( (float) num_radii/BLOCK_SIZE_3D);
  const dim3 gridSize_accum = dim3(bx_accum, by_accum, bz_accum);

  int * d_accum;
  cudaMalloc(&d_accum, width*height*num_radii*sizeof(int));  
  cudaMemset(d_accum, 0, width*height*num_radii*sizeof(int));

  // ----------------------------------------------------------------------------------------------------

  printf("1\n");
  double** d_rtable;
  int* d_pointer;
  int num_angles = 181;

  cudaMalloc(&d_rtable, num_angles*sizeof(double*));
  cudaMalloc(&d_pointer, num_angles*sizeof(int));
  rtable_init_kernel<<<1, num_angles>>>(d_rtable, d_pointer, 1000);
  rtable_create_kernel<<<gridSize, blockSize>>>(d_rtable, d_pointer, d_output_edge, d_output_orientation, height, width, width/2, height/2);
  cudaDeviceSynchronize();

  int * d_accum_general;  
  cudaMalloc(&d_accum_general, width*height*num_angles*sizeof(int));  
  cudaMemset(d_accum_general, 0, width*height*num_angles*sizeof(int));
  cudaEventRecord(start_kernel);
  accumulator_general<<<gridSize_accum, blockSize_accum>>>(d_output_edge_thresholded, d_accum_general, height, width, 181, d_pointer, d_rtable);
  cudaDeviceSynchronize();
  cudaEventRecord(stop_kernel);
  
  printf("2\n");

  cudaEventSynchronize(stop_kernel);
  float k_time ;
  cudaEventElapsedTime(&k_time, start_kernel, stop_kernel);
  cout << "GPU Time: " << k_time << " milliseconds" << endl;
  // ----------------------------------------------------------------------------------------------------

  accumulator_circle<<<gridSize_accum, blockSize_accum>>>(d_output_edge_thresholded, d_accum, height, width, num_radii);
  threshold<<<gridSize_accum, blockSize_accum>>>(d_accum_general, height, width, num_radii, 20);

  unsigned char * h_blurred_image = (unsigned char *) malloc(width*height*sizeof(unsigned char));
  cudaMemcpy(h_blurred_image, d_blurred_image, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  unsigned char * h_output_edge = (unsigned char *) malloc(width*height*sizeof(unsigned char));
  cudaMemcpy(h_output_edge, d_output_edge, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  unsigned char * h_output_edge_thresholded = (unsigned char *) malloc(width*height*sizeof(unsigned char));
  cudaMemcpy(h_output_edge_thresholded, d_output_edge_thresholded, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  unsigned int * h_output_orientation = (unsigned int *) malloc(width*height*sizeof(unsigned int));
  cudaMemcpy(h_output_orientation, d_output_orientation, width*height*sizeof(unsigned int), cudaMemcpyDeviceToHost);

  int * h_accum = (int *) malloc(width*height*num_radii*sizeof(int));
  cudaMemcpy(h_accum, d_accum, width*height*num_radii*sizeof(int), cudaMemcpyDeviceToHost);

  unsigned char * accum_image = get_image_accum_radii( h_accum, height, width, 25, num_radii);

  stbi_write_png(gpu_blurred_image.c_str(), width, height, 1, h_blurred_image, 0);
  stbi_write_png(gpu_out_edge.c_str(), width, height, 1, h_output_edge, 0);
  stbi_write_png(gpu_out_edge_thresholded.c_str(), width, height, 1, h_output_edge_thresholded, 0);
  // stbi_write_png(gpu_out_orientation.c_str(), width, height, 1, h_output_orientation, 0);
  stbi_write_png(gpu_accum_image.c_str(), width, height, 1, accum_image, 0);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_output_edge);
  cudaFree(d_output_orientation);

  free(h_output_edge);
  free(h_output_orientation);
  /**********************CUDA-CODE-ENDS*************************************/

  return 0;
}
