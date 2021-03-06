#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    // Lab 6 & 10
    int mode, param;
    float paramFloat;
    std::string inputFilename2;
    JpegInfo *inputImage2;

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    switch (lwNum) {
        case 1:
            timer.start();
            labwork.labwork1_CPU();
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork1-cpu-out.jpg");

            timer.start();
            labwork.labwork1_OpenMP();
            printf("labwork 1 OpenMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork1-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            timer.start();
            for (int i = 0; i < 100; ++i) {
                labwork.labwork3_GPU();
            }
            printf("labwork 3 GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            timer.start();
            for (int i = 0; i < 100; ++i) {
                labwork.labwork4_GPU();
            }
            printf("labwork 4 GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            timer.start();
            labwork.labwork5_CPU();
            printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-cpu-out.jpg");

            timer.start();
            for (int i = 0; i < 100; ++i) {
                labwork.labwork5_GPU();
            }
            printf("labwork 5 GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu-out.jpg");

            timer.start();
            for (int i = 0; i < 100; ++i) {
                labwork.labwork5_GPU_optimized();
            }
            printf("labwork 5 GPU optimized (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork5-gpu-optimized-out.jpg");
            break;
        case 6:
            mode = atoi(argv[3]);
            param = atoi(argv[4]);
            if (mode == 0) {
                timer.start();
                for (int i = 0; i < 100; ++i) {
                    labwork.labwork6a_GPU(param);
                }
                printf("labwork 6a GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
                labwork.saveOutputImage("labwork6a-gpu-out.jpg");
            } else if (mode == 1) {
                timer.start();
                for (int i = 0; i < 100; ++i) {
                    labwork.labwork6b_GPU(param);
                }
                printf("labwork 6b GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
                labwork.saveOutputImage("labwork6b-gpu-out.jpg");
            } else if (mode == 2) {
                paramFloat = atof(argv[4]);
                inputFilename2 = std::string(argv[5]);
                inputImage2 = labwork.loadImage(inputFilename2);
                timer.start();
                for (int i = 0; i < 100; ++i) {
                    labwork.labwork6c_GPU(paramFloat, inputImage2);
                }
                printf("labwork 6c GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
                labwork.saveOutputImage("labwork6c-gpu-out.jpg");
            }
            break;
        case 7:
            timer.start();
            for (int i = 0; i < 100; ++i) {
                labwork.labwork7_GPU();
            }
            printf("labwork 7 GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            timer.start();
            for (int i = 0; i < 100; ++i) {
                labwork.labwork8_GPU();
            }
            printf("labwork 8 GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            timer.start();
            for (int i = 0; i < 100; ++i) {
                labwork.labwork9_GPU();
            }
            printf("labwork 9 GPU (100 times) ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            param = atoi(argv[3]);
            timer.start();
            labwork.labwork10_GPU(param);
            printf("labwork 10 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

JpegInfo* Labwork::loadImage(std::string fileName) {
    return jpegLoader.load(fileName);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) { // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = ((unsigned char) inputImage->buffer[i * 3] +
                                  (unsigned char) inputImage->buffer[i * 3 + 1] +
                                  (unsigned char) inputImage->buffer[i * 3 + 2]) / 3;
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));

    #pragma omp parallel for

    for (int j = 0; j < 100; j++) { // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = ((unsigned char) inputImage->buffer[i * 3] +
                                  (unsigned char) inputImage->buffer[i * 3 + 1] +
                                  (unsigned char) inputImage->buffer[i * 3 + 2]) / 3;
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int noOfGPUs = 0;
    cudaGetDeviceCount(&noOfGPUs);

    for (int i = 0; i < noOfGPUs; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nGPU #%d:\n", i);
        printf(" - Name: %s\n", prop.name);
        printf(" - Core info:\n");
        printf("    + Clock rate: %d\n", prop.clockRate);
        printf("    + Number of cores: %d\n", getSPcores(prop));
        printf("    + Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("    + Warp size: %d\n", prop.warpSize);
        printf(" - Memory info:\n");
        printf("    + Clock rate: %d\n", prop.memoryClockRate);
        printf("    + Bus width: %d\n", prop.memoryBusWidth);
        printf("    + Bandwidth: %d\n", prop.memoryClockRate * prop.memoryBusWidth);
    }
}

__global__ void grayscale(unsigned char *input, char *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid * 3] = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2]) / 3;
    output[tid * 3 + 2] = output[tid * 3 + 1] = output[tid * 3];
}

void Labwork::labwork3_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = pixelCount / blockSize;

    grayscale<<<numBlock, blockSize>>>(devInput, devOutput);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void grayscale2D(unsigned char *input, char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    // int globalBlockIdx = blockIdx.x + gridDim.x * blockIdx.y;
    // int globalId = globalBlockIdx * blockDim.x * blockDim.y + (threadIdx.x + blockDim.x * threadIdx.y);

    output[globalId * 3] = (input[globalId * 3] + input[globalId * 3 + 1] + input[globalId * 3 + 2]) / 3;
    output[globalId * 3 + 2] = output[globalId * 3 + 1] = output[globalId * 3];
}

void Labwork::labwork4_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    grayscale2D<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = {0, 0,  1,  2,   1,  0,  0,
                    0, 3,  13, 22,  13, 3,  0,
                    1, 13, 59, 97,  59, 13, 1,
                    2, 22, 97, 159, 97, 22, 2,
                    1, 13, 59, 97,  59, 13, 1,
                    0, 3,  13, 22,  13, 3,  0,
                    0, 0,  1,  2,   1,  0,  0};
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = ((unsigned char) inputImage->buffer[tid * 3] +
                                          (unsigned char) inputImage->buffer[tid * 3 + 1] +
                                          (unsigned char) inputImage->buffer[tid * 3 + 2]) / 3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

__global__ void gaussianBlur(unsigned char *input, char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    int weights[] = {0, 0,  1,  2,   1,  0,  0,
                     0, 3,  13, 22,  13, 3,  0,
                     1, 13, 59, 97,  59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97,  59, 13, 1,
                     0, 3,  13, 22,  13, 3,  0,
                     0, 0,  1,  2,   1,  0,  0};

    int sum = 0;
    int c = 0;
    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            int i = globalIdX + x;
            int j = globalIdY + y;
            if (i < 0) continue;
            if (i >= width) continue;
            if (j < 0) continue;
            if (j >= height) continue;
            int tid = j * width + i;
            unsigned char gray = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2])/3;
            int coefficient = weights[(y+3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;

    output[globalId * 3] = output[globalId * 3 + 1] = output[globalId * 3 + 2] = sum;
}

void Labwork::labwork5_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    gaussianBlur<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void gaussianBlurOptimized(unsigned char *input, char *output, int width, int height, int* weights) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    __shared__ int sharedWeights[49];

    int localId = threadIdx.x + threadIdx.y * blockDim.x;
    if (localId < 49)
    {
        sharedWeights[localId] = weights[localId];
    }

    __syncthreads();

    int sum = 0;
    int c = 0;
    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            int i = globalIdX + x;
            int j = globalIdY + y;
            if (i < 0) continue;
            if (i >= width) continue;
            if (j < 0) continue;
            if (j >= height) continue;
            int tid = j * width + i;
            unsigned char gray = (input[tid * 3] + input[tid * 3 + 1] + input[tid * 3 + 2])/3;
            int coefficient = sharedWeights[(y+3) * 7 + x + 3];
            sum = sum + gray * coefficient;
            c += coefficient;
        }
    }
    sum /= c;
    output[globalId * 3] = sum;
    output[globalId * 3 + 1] = sum;
    output[globalId * 3 + 2] = sum;
}

void Labwork::labwork5_GPU_optimized() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    int weights[] = {0, 0,  1,  2,   1,  0,  0,
                     0, 3,  13, 22,  13, 3,  0,
                     1, 13, 59, 97,  59, 13, 1,
                     2, 22, 97, 159, 97, 22, 2,
                     1, 13, 59, 97,  59, 13, 1,
                     0, 3,  13, 22,  13, 3,  0,
                     0, 0,  1,  2,   1,  0,  0};

    int* devWeights;
    cudaMalloc(&devWeights, 49 * sizeof(int));
    cudaMemcpy(devWeights, weights, 49 * sizeof(int), cudaMemcpyHostToDevice);

    gaussianBlurOptimized<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, devWeights);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void binarization(unsigned char *input, char *output, int width, int height, int threshold) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char binary = (input[globalId * 3] + input[globalId * 3 + 1] + input[globalId * 3 + 2]) / 3;
    // binary = (binary / threshold) > 0 ? 255 : 0;
    binary = min(binary / threshold, 1) * 255;

    output[globalId * 3] = binary;
    output[globalId * 3 + 1] = binary;
    output[globalId * 3 + 2] = binary;
}

void Labwork::labwork6a_GPU(int threshold) {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    binarization<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, threshold);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void brightness(unsigned char *input, char *output, int width, int height, int brightnessChange) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char r = min(max(input[globalId * 3] + brightnessChange, 0), 255);
    unsigned char g = min(max(input[globalId * 3 + 1] + brightnessChange, 0), 255);
    unsigned char b = min(max(input[globalId * 3 + 2] + brightnessChange, 0), 255);

    output[globalId * 3] = r;
    output[globalId * 3 + 1] = g;
    output[globalId * 3 + 2] = b;
}

void Labwork::labwork6b_GPU(int brightnessChange) {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    brightness<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, brightnessChange);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void blend(unsigned char *input, unsigned char *input2, char *output, int width, int height, float ratio) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char r = input[globalId * 3] * ratio + input2[globalId * 3] * (1 - ratio);
    unsigned char g = input[globalId * 3 + 1] * ratio + input2[globalId * 3 + 1] * (1 - ratio);
    unsigned char b = input[globalId * 3 + 2] * ratio + input2[globalId * 3 + 2] * (1 - ratio);

    output[globalId * 3] = r;
    output[globalId * 3 + 1] = g;
    output[globalId * 3 + 2] = b;
}

void Labwork::labwork6c_GPU(float ratio, JpegInfo *inputImage2) {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char* devInput;
    unsigned char* devInput2;
    char* devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devInput2, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devInput2, inputImage2->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    blend<<<gridSize, blockSize>>>(devInput, devInput2, devOutput, inputImage->width, inputImage->height, ratio);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void getGreyscale(unsigned char *input, unsigned char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    output[globalId] = (input[globalId * 3] + input[globalId * 3 + 1] + input[globalId * 3 + 2]) / 3;
}

__global__ void getMaxIntensity(unsigned char *input, unsigned char *output, int count) {
    extern __shared__ unsigned char blockShareArray[];

    int blockSize = blockDim.x * blockDim.y;
    int localId = threadIdx.x + blockDim.x * threadIdx.y;
    int globalId = blockIdx.x * blockSize + localId;

    if (globalId < count) {
        blockShareArray[localId] = input[globalId];
    } else {
        blockShareArray[localId] = 0;
    }

    __syncthreads();

    for (int s = 1; s < blockSize; s *= 2) {
        if (localId % (s * 2) == 0) {
            blockShareArray[localId] = max(blockShareArray[localId], blockShareArray[localId + s]);
        }

        __syncthreads();
    }

    if (localId == 0) {
        output[blockIdx.x] = blockShareArray[0];
    }
}

__global__ void getMinIntensity(unsigned char *input, unsigned char *output, int count) {
    extern __shared__ unsigned char blockShareArray[];

    int blockSize = blockDim.x * blockDim.y;
    int localId = threadIdx.x + blockDim.x * threadIdx.y;
    int globalId = blockIdx.x * blockSize + localId;

    if (globalId < count) {
        blockShareArray[localId] = input[globalId];
    } else {
        blockShareArray[localId] = 255;
    }

    __syncthreads();

    for (int s = 1; s < blockSize; s *= 2) {
        if (localId % (s * 2) == 0) {
            blockShareArray[localId] = min(blockShareArray[localId], blockShareArray[localId + s]);
        }

        __syncthreads();
    }

    if (localId == 0) {
        output[blockIdx.x] = blockShareArray[0];
    }
}

__global__ void grayscaleStretch(unsigned char *input, char *output, unsigned char *max, unsigned char *min, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char greyStretched = ((float) (input[globalId] - min[0]) / (max[0] - min[0])) * 255;

    output[globalId * 3] = greyStretched;
    output[globalId * 3 + 1] = greyStretched;
    output[globalId * 3 + 2] = greyStretched;
}

void Labwork::labwork7_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char *devInput, *devGrey;
    char *devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devGrey, pixelCount);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    getGreyscale<<<gridSize, blockSize>>>(devInput, devGrey, inputImage->width, inputImage->height);

    // REDUCTION BEGIN ----------------------------------------------
    int threadsPerBlock = blockX * blockY;
    int reduceGridSize = (pixelCount + threadsPerBlock - 1) / threadsPerBlock;
    int swap = 0;

    unsigned char *devGlobalMaxArray1, *devGlobalMaxArray2, *devGlobalMinArray1, *devGlobalMinArray2;
    cudaMalloc(&devGlobalMaxArray1, reduceGridSize);
    cudaMalloc(&devGlobalMinArray1, reduceGridSize);

    unsigned char *maxArrayPointer[2], *minArrayPointer[2];
    maxArrayPointer[swap] = devGlobalMaxArray1;
    minArrayPointer[swap] = devGlobalMinArray1;

    getMaxIntensity<<<reduceGridSize, blockSize, threadsPerBlock>>>(devGrey, devGlobalMaxArray1, pixelCount);
    getMinIntensity<<<reduceGridSize, blockSize, threadsPerBlock>>>(devGrey, devGlobalMinArray1, pixelCount);

    int tempCount = reduceGridSize;
    reduceGridSize = (reduceGridSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&devGlobalMaxArray2, reduceGridSize);
    cudaMalloc(&devGlobalMinArray2, reduceGridSize);
    maxArrayPointer[!swap] = devGlobalMaxArray2;
    minArrayPointer[!swap] = devGlobalMinArray2;

    while (reduceGridSize > 1) {
        getMaxIntensity<<<reduceGridSize, blockSize, threadsPerBlock>>>(maxArrayPointer[swap], maxArrayPointer[!swap], tempCount);
        getMinIntensity<<<reduceGridSize, blockSize, threadsPerBlock>>>(minArrayPointer[swap], minArrayPointer[!swap], tempCount);

        tempCount = reduceGridSize;
        reduceGridSize = (reduceGridSize + threadsPerBlock - 1) / threadsPerBlock;
        swap = !swap;
    }

    getMaxIntensity<<<1, blockSize, threadsPerBlock>>>(maxArrayPointer[swap], maxArrayPointer[!swap], tempCount);
    getMinIntensity<<<1, blockSize, threadsPerBlock>>>(minArrayPointer[swap], minArrayPointer[!swap], tempCount);
    // REDUCTION END ------------------------------------------------

    grayscaleStretch<<<gridSize, blockSize>>>(devGrey, devOutput, maxArrayPointer[!swap], minArrayPointer[!swap], inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devGrey);
    cudaFree(devOutput);
    cudaFree(devGlobalMaxArray1);
    cudaFree(devGlobalMaxArray2);
    cudaFree(devGlobalMinArray1);
    cudaFree(devGlobalMinArray2);
}

__global__ void RGB2HSV(unsigned char *input, int *hue, float *saturation, float *value, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    float floatR = input[globalId * 3] / 255.0f;
    float floatG = input[globalId * 3 + 1] / 255.0f;
    float floatB = input[globalId * 3 + 2] / 255.0f;
    float maxValue = max(max(floatR, floatG), floatB);
    float delta = maxValue - min(min(floatR, floatG), floatB);

    value[globalId] = maxValue;

    if (delta == 0.0f) {
        hue[globalId] = 0;
        saturation[globalId] = 0.0f;
    } else {
        saturation[globalId] = delta / maxValue;

        if (maxValue == floatR) {
            hue[globalId] = 60.0f * fmod((floatG - floatB) / delta, 6.0f);
        } else if (maxValue == floatG) {
            hue[globalId] = 60.0f * ((floatB - floatR) / delta + 2.0f);
        } else {
            hue[globalId] = 60.0f * ((floatR - floatG) / delta + 4.0f);
        }

        if (hue[globalId] < 0) {
            hue[globalId] = 360 + hue[globalId];
        }
    }
}

__global__ void HSV2RGB(int *hue, float *saturation, float *value, char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    float c = value[globalId] * saturation[globalId];
    float x = c * (1.0f - abs(fmod(hue[globalId] / 60.0f, 2.0f) - 1.0f));
    float m = value[globalId] - c;

    if (hue[globalId] < 60) {
        output[globalId * 3] = (c + m) * 255.0f;
        output[globalId * 3 + 1] = (x + m) * 255.0f;
        output[globalId * 3 + 2] = m * 255.0f;
    } else if (hue[globalId] < 120) {
        output[globalId * 3] = (x + m) * 255.0f;
        output[globalId * 3 + 1] = (c + m) * 255.0f;
        output[globalId * 3 + 2] = m * 255.0f;
    } else if (hue[globalId] < 180) {
        output[globalId * 3] = m * 255.0f;
        output[globalId * 3 + 1] = (c + m) * 255.0f;
        output[globalId * 3 + 2] = (x + m) * 255.0f;
    } else if (hue[globalId] < 240) {
        output[globalId * 3] = m * 255.0f;
        output[globalId * 3 + 1] = (x + m) * 255.0f;
        output[globalId * 3 + 2] = (c + m) * 255.0f;
    } else if (hue[globalId] < 300) {
        output[globalId * 3] = (x + m) * 255.0f;
        output[globalId * 3 + 1] = m * 255.0f;
        output[globalId * 3 + 2] = (c + m) * 255.0f;
    } else {
        output[globalId * 3] = (c + m) * 255.0f;
        output[globalId * 3 + 1] = m * 255.0f;
        output[globalId * 3 + 2] = (x + m) * 255.0f;
    }
}

void Labwork::labwork8_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);

    unsigned char *devInput;
    int *devHue;
    float *devSaturation, *devValue;
    char *devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);
    cudaMalloc(&devHue, pixelCount * sizeof(int));
    cudaMalloc(&devSaturation, pixelCount * sizeof(float));
    cudaMalloc(&devValue, pixelCount * sizeof(float));

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    RGB2HSV<<<gridSize, blockSize>>>(devInput, devHue, devSaturation, devValue, inputImage->width, inputImage->height);

    HSV2RGB<<<gridSize, blockSize>>>(devHue, devSaturation, devValue, devOutput, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
    cudaFree(devHue);
    cudaFree(devSaturation);
    cudaFree(devValue);
}

__global__ void histogram(unsigned char *input, int *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    atomicAdd(&output[input[globalId]], 1);
}

__global__ void histogramEqualizationMap(int *histogram, unsigned char *equalizationMap, int totalPixels) {
    extern __shared__ float probabilities[];
    probabilities[threadIdx.x] = (float) histogram[threadIdx.x] / totalPixels;
    __syncthreads();

    float cdf = 0.0f;
    for (int i = 0; i <= threadIdx.x; i++) {
        cdf += probabilities[i];
    }

    equalizationMap[threadIdx.x] = cdf * 255;
}

__global__ void histogramEqualization(unsigned char *devGrey, unsigned char *equalizationMap, char *output, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    unsigned char newGrey = equalizationMap[devGrey[globalId]];
    output[globalId * 3] = newGrey;
    output[globalId * 3 + 1] = newGrey;
    output[globalId * 3 + 2] = newGrey;
}

void Labwork::labwork9_GPU() {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);

    unsigned char *devInput, *devGrey, *devEqualizationMap;
    int *devHistogram;
    char *devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devGrey, pixelCount);
    cudaMalloc(&devEqualizationMap, 256);
    cudaMalloc(&devHistogram, sizeof(int) * 256);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemset(devHistogram, 0, sizeof(int) * 256);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    getGreyscale<<<gridSize, blockSize>>>(devInput, devGrey, inputImage->width, inputImage->height);
    histogram<<<gridSize, blockSize>>>(devGrey, devHistogram, inputImage->width, inputImage->height);
    histogramEqualizationMap<<<1, 256, sizeof(float) * 256>>>(devHistogram, devEqualizationMap, pixelCount);
    histogramEqualization<<<gridSize, blockSize>>>(devGrey, devEqualizationMap, devOutput, inputImage->width, inputImage->height);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devGrey);
    cudaFree(devEqualizationMap);
    cudaFree(devHistogram);
    cudaFree(devOutput);
}

__global__ void RGB2Value(unsigned char *input, float *value, int width, int height) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    float floatR = input[globalId * 3] / 255.0f;
    float floatG = input[globalId * 3 + 1] / 255.0f;
    float floatB = input[globalId * 3 + 2] / 255.0f;

    value[globalId] = max(max(floatR, floatG), floatB);
}

__global__ void kuwahara(unsigned char *input, float *value, char *output, int width, int height, int windowSize) {
    int globalIdX = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalIdX >= width) return;
    int globalIdY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalIdY >= height) return;
    int globalId = globalIdY * width + globalIdX;

    int windowPivots[4][2];
    // windowPivots[[w1x, w1y], [w2x, w2y], [w3x, w3y], [w4x, w4y]]
    // whereas w1x = w3x, w1y = w2y, w2x = w4x, w3y = w4y
    windowPivots[0][0] = windowPivots[2][0] = max(globalIdX - windowSize + 1, 0);
    windowPivots[0][1] = windowPivots[1][1] = max(globalIdY - windowSize + 1, 0);
    windowPivots[1][0] = windowPivots[3][0] = min(globalIdX + windowSize - 1, width - 1);
    windowPivots[2][1] = windowPivots[3][1] = min(globalIdY + windowSize - 1, height - 1);

    float standardDeviations[4];
    int noOfCells[4] = {0, 0, 0, 0};

    // window #1
    float tempTotal = 0.0f;
    for (int i = windowPivots[0][1]; i <= globalIdY; i++) {
        for (int j = windowPivots[0][0]; j <= globalIdX; j++) {
            noOfCells[0]++;
            tempTotal += value[i * width + j];
        }
    }

    float mean = tempTotal / noOfCells[0];
    tempTotal = 0.0f;
    for (int i = windowPivots[0][1]; i <= globalIdY; i++) {
        for (int j = windowPivots[0][0]; j <= globalIdX; j++) {
            tempTotal += pow(value[i * width + j] - mean, 2);
        }
    }

    standardDeviations[0] = sqrt(tempTotal / noOfCells[0]);

    // window #2
    tempTotal = 0.0f;
    for (int i = windowPivots[1][1]; i <= globalIdY; i++) {
        for (int j = windowPivots[1][0]; j >= globalIdX; j--) {
            noOfCells[1]++;
            tempTotal += value[i * width + j];
        }
    }

    mean = tempTotal / noOfCells[1];
    tempTotal = 0.0f;
    for (int i = windowPivots[1][1]; i <= globalIdY; i++) {
        for (int j = windowPivots[1][0]; j >= globalIdX; j--) {
            tempTotal += pow(value[i * width + j] - mean, 2);
        }
    }

    standardDeviations[1] = sqrt(tempTotal / noOfCells[1]);

    // window #3
    tempTotal = 0.0f;
    for (int i = windowPivots[2][1]; i >= globalIdY; i--) {
        for (int j = windowPivots[2][0]; j <= globalIdX; j++) {
            noOfCells[2]++;
            tempTotal += value[i * width + j];
        }
    }

    mean = tempTotal / noOfCells[2];
    tempTotal = 0.0f;
    for (int i = windowPivots[2][1]; i >= globalIdY; i--) {
        for (int j = windowPivots[2][0]; j <= globalIdX; j++) {
            tempTotal += pow(value[i * width + j] - mean, 2);
        }
    }

    standardDeviations[2] = sqrt(tempTotal / noOfCells[2]);

    // window #4
    tempTotal = 0.0f;
    for (int i = windowPivots[3][1]; i >= globalIdY; i--) {
        for (int j = windowPivots[3][0]; j >= globalIdX; j--) {
            noOfCells[3]++;
            tempTotal += value[i * width + j];
        }
    }

    mean = tempTotal / noOfCells[3];
    tempTotal = 0.0f;
    for (int i = windowPivots[3][1]; i >= globalIdY; i--) {
        for (int j = windowPivots[3][0]; j >= globalIdX; j--) {
            tempTotal += pow(value[i * width + j] - mean, 2);
        }
    }

    standardDeviations[3] = sqrt(tempTotal / noOfCells[3]);

    mean = min(standardDeviations[0], min(standardDeviations[1], min(standardDeviations[2], standardDeviations[3])));
    if (mean == standardDeviations[0]) {
        int r = 0, g = 0, b = 0;
        for (int i = windowPivots[0][1]; i <= globalIdY; i++) {
            for (int j = windowPivots[0][0]; j <= globalIdX; j++) {
                r += input[(i * width + j) * 3];
                g += input[(i * width + j) * 3 + 1];
                b += input[(i * width + j) * 3 + 2];
            }
        }

        output[globalId * 3] = r / noOfCells[0];
        output[globalId * 3 + 1] = g / noOfCells[0];
        output[globalId * 3 + 2] = b / noOfCells[0];
    } else if (mean == standardDeviations[1]) {
        int r = 0, g = 0, b = 0;
        for (int i = windowPivots[1][1]; i <= globalIdY; i++) {
            for (int j = windowPivots[1][0]; j >= globalIdX; j--) {
                r += input[(i * width + j) * 3];
                g += input[(i * width + j) * 3 + 1];
                b += input[(i * width + j) * 3 + 2];
            }
        }

        output[globalId * 3] = r / noOfCells[1];
        output[globalId * 3 + 1] = g / noOfCells[1];
        output[globalId * 3 + 2] = b / noOfCells[1];
    } else if (mean == standardDeviations[2]) {
        int r = 0, g = 0, b = 0;
        for (int i = windowPivots[2][1]; i >= globalIdY; i--) {
            for (int j = windowPivots[2][0]; j <= globalIdX; j++) {
                r += input[(i * width + j) * 3];
                g += input[(i * width + j) * 3 + 1];
                b += input[(i * width + j) * 3 + 2];
            }
        }

        output[globalId * 3] = r / noOfCells[2];
        output[globalId * 3 + 1] = g / noOfCells[2];
        output[globalId * 3 + 2] = b / noOfCells[2];
    } else {
        int r = 0, g = 0, b = 0;
        for (int i = windowPivots[3][1]; i >= globalIdY; i--) {
            for (int j = windowPivots[3][0]; j >= globalIdX; j--) {
                r += input[(i * width + j) * 3];
                g += input[(i * width + j) * 3 + 1];
                b += input[(i * width + j) * 3 + 2];
            }
        }

        output[globalId * 3] = r / noOfCells[3];
        output[globalId * 3 + 1] = g / noOfCells[3];
        output[globalId * 3 + 2] = b / noOfCells[3];
    }
}

void Labwork::labwork10_GPU(int windowSize) {
    int pixelCount = inputImage->width * inputImage->height;

    outputImage = (char *) malloc(pixelCount * 3);
    unsigned char *devInput;
    float *devValue;
    char *devOutput;

    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devValue, pixelCount * sizeof(float));
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    int blockX = 32;
    int blockY = 32;
    dim3 blockSize = dim3(blockX, blockY);
    dim3 gridSize = dim3((inputImage->width + blockX - 1) / blockX, (inputImage->height + blockY - 1) / blockY);

    RGB2Value<<<gridSize, blockSize>>>(devInput, devValue, inputImage->width, inputImage->height);
    kuwahara<<<gridSize, blockSize>>>(devInput, devValue, devOutput, inputImage->width, inputImage->height, windowSize);

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    cudaFree(devInput);
    cudaFree(devOutput);
}
