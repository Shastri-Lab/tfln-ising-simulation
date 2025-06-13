// gpu_mps.mm
// matrix–vector multiply using metal performance shaders (mps) – float32 version
//
// this was a bit of a pain to get compiled. I finally got it to work with:
// SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
// /usr/bin/clang++ \
//        -std=c++17 -fobjc-arc -stdlib=libc++ \
//        -isysroot "$SDK" \
//        -I"$SDK/usr/include/c++/v1" \
//        -mmacos-version-min=11.0 \
//        -O3 -march=native -fPIC -shared \
//        gpu_mps.mm \
//        -framework Foundation \
//        -framework Metal \
//        -framework MetalPerformanceShaders \
//        -o libgpu_mps.dylib

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#else
#error compile with -ObjC++
#endif

#include <cstddef>
#include <cstring>

extern "C"
void sgemv_mps(const float *A,
               const float *x,
               float       *y,
               size_t        m,
               size_t        n)
{
    if (!A || !x || !y || m == 0 || n == 0) return;

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return;

    const size_t elemsA = m * n;
    const size_t elemsX = n;
    const size_t elemsY = m;

    id<MTLBuffer> bufA = [dev newBufferWithBytes:A
                                          length:elemsA * sizeof(float)
                                         options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufX = [dev newBufferWithBytes:x
                                          length:elemsX * sizeof(float)
                                         options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufY = [dev newBufferWithLength:elemsY * sizeof(float)
                                          options:MTLResourceStorageModeShared];

    MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
        matrixDescriptorWithRows:m
                         columns:n
                        rowBytes:n * sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *dX = [MPSMatrixDescriptor
        matrixDescriptorWithRows:n
                         columns:1
                        rowBytes:sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrixDescriptor *dY = [MPSMatrixDescriptor
        matrixDescriptorWithRows:m
                         columns:1
                        rowBytes:sizeof(float)
                        dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:dA];
    MPSMatrix *matX = [[MPSMatrix alloc] initWithBuffer:bufX descriptor:dX];
    MPSMatrix *matY = [[MPSMatrix alloc] initWithBuffer:bufY descriptor:dY];

    MPSMatrixMultiplication *op =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                         transposeLeft:false
                                        transposeRight:false
                                           resultRows:m
                                        resultColumns:1
                                       interiorColumns:n
                                                alpha:1.0f
                                                 beta:0.0f];

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];

    [op encodeToCommandBuffer:commandBuffer
                  leftMatrix:matA
                 rightMatrix:matX
                resultMatrix:matY];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    memcpy(y, bufY.contents, elemsY * sizeof(float));
}