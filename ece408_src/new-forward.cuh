#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
// #define STRIDE  10
#define M_MAX   24
#define M_MIN   12
#define W_MAX   72
#define W_MIN   33
#define H_MAX   72
#define H_MIN   33
#define C_MAX   12
#define C_MIN   1
#define K_MAX   7
// #define B   100

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float w_1[14112];

__global__ void matrix_multiplication_1(float *y, const float *x, const float *w, const int B, const int M, const int C, const int H, const int W, const int K, const int W_out, const int H_out){

#define TILE_WIDTH 29
#define STRIDE  5
#define y4d(i2, i1, i0) y[(i2) * (M * H_out * W_out) + (i1) * (H_out * W_out) + i0]
#define x4d(i2, i1, i0) x[(i2) * (H * W) + (i1) * (W) + i0]
// #define w4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define w4d(i1, i0) w[(i1) * (C * K * K) + i0]

    __shared__ float wds[12][49];
    __shared__ float xds[49][TILE_WIDTH];
    __shared__ float xds_1[49][TILE_WIDTH]; 
    __shared__ float xds_2[49][TILE_WIDTH];
    __shared__ float xds_3[49][TILE_WIDTH];  
    __shared__ float xds_4[49][TILE_WIDTH]; 
    // __shared__ float xds_5[49][TILE_WIDTH]; 
    // __shared__ float xds_6[49][TILE_WIDTH]; 
    // __shared__ float xds_7[49][TILE_WIDTH]; 

    int bx = blockIdx.x; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int Row = ty;
    int Col = bx * TILE_WIDTH + tx;
    int B_idx = blockIdx.z;

    int numAColumns = C*K*K;
    // int numARows = M;
    int numBColumns = W_out*H_out;
    int numBRows = C*K*K;
    int numCColumns = W_out*H_out;
    int numCRows = M;
    int KK = K*K;

    float Cvalue = 0;
    float Cvalue_1 = 0;
    float Cvalue_2 = 0;
    float Cvalue_3 = 0;
    float Cvalue_4 = 0;
    // float Cvalue_5 = 0;
    // float Cvalue_6 = 0;
    // float Cvalue_7 = 0;
    int B_STRIDE = B/STRIDE;
    int H_out_idx = Col/W_out;
    int W_out_idx = Col%W_out;
    int M_4 = M*4;

    wds[ty][tx] = w4d(ty, tx);
    if(tx+TILE_WIDTH<numAColumns) {
        wds[ty][tx+TILE_WIDTH] = w4d(ty, tx+TILE_WIDTH);
    }

    if(Row<numBRows && Col<numBColumns) {
        for(int i=0; i<4; i++){
            xds[ty+M*i][tx] = x4d(B_idx, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
            xds_1[ty+M*i][tx] = x4d(B_idx+B_STRIDE, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
            xds_2[ty+M*i][tx] = x4d(B_idx+B_STRIDE*2, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
            xds_3[ty+M*i][tx] = x4d(B_idx+B_STRIDE*3, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
            xds_4[ty+M*i][tx] = x4d(B_idx+B_STRIDE*4, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
            // xds_5[ty+M*i][tx] = x4d(B_idx+B_STRIDE*5, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
            // xds_6[ty+M*i][tx] = x4d(B_idx+B_STRIDE*6, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
            // xds_7[ty+M*i][tx] = x4d(B_idx+B_STRIDE*7, H_out_idx+(ty+M*i)/K, W_out_idx+(ty+M*i)%K);
        }
    }
    if((Row+M_4)<numBRows && Col<numBColumns){
        xds[ty+M_4][tx] = x4d(B_idx, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
        xds_1[ty+M_4][tx] = x4d(B_idx+B_STRIDE, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
        xds_2[ty+M_4][tx] = x4d(B_idx+B_STRIDE*2, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
        xds_3[ty+M_4][tx] = x4d(B_idx+B_STRIDE*3, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
        xds_4[ty+M_4][tx] = x4d(B_idx+B_STRIDE*4, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
        // xds_5[ty+M_4][tx] = x4d(B_idx+B_STRIDE*5, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
        // xds_6[ty+M_4][tx] = x4d(B_idx+B_STRIDE*6, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
        // xds_7[ty+M_4][tx] = x4d(B_idx+B_STRIDE*7, H_out_idx+(ty+M_4)/K, W_out_idx+(ty+M_4)%K);
    }
    __syncthreads();
    if(Row<numCRows && Col<numCColumns){
        for (int i = 0; i < C*KK; ++i) {
            Cvalue += wds[ty][i] * xds[i][tx];
            Cvalue_1 += wds[ty][i] * xds_1[i][tx];
            Cvalue_2 += wds[ty][i] * xds_2[i][tx];
            Cvalue_3 += wds[ty][i] * xds_3[i][tx];
            Cvalue_4 += wds[ty][i] * xds_4[i][tx];
            // Cvalue_5 += wds[ty][i] * xds_5[i][tx];
            // Cvalue_6 += wds[ty][i] * xds_6[i][tx];
            // Cvalue_7 += wds[ty][i] * xds_7[i][tx];
        }
        y4d(B_idx, Row, Col) = Cvalue;
        y4d(B_idx+B_STRIDE, Row, Col) = Cvalue_1;
        y4d(B_idx+B_STRIDE*2, Row, Col) = Cvalue_2;
        y4d(B_idx+B_STRIDE*3, Row, Col) = Cvalue_3;
        y4d(B_idx+B_STRIDE*4, Row, Col) = Cvalue_4;
        // y4d(B_idx+B_STRIDE*5, Row, Col) = Cvalue_5;
        // y4d(B_idx+B_STRIDE*6, Row, Col) = Cvalue_6;
        // y4d(B_idx+B_STRIDE*7, Row, Col) = Cvalue_7;
    }


#undef TILE_WIDTH
#undef STRIDE
#undef y4d
#undef x4d 
#undef w4d  
}


__global__ void matrix_multiplication_2(float *y, const float *x, const float *w, const int B, const int M, const int C, const int H, const int W, const int K, const int W_out, const int H_out){

#define TILE_WIDTH 32
#define STRIDE  10
#define y4d(i2, i1, i0) y[(i2) * (M * H_out * W_out) + (i1) * (H_out * W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define w4d(i1, i0) w_1[(i1) * (C * K * K) + i0]

    __shared__ float wds[M_MAX][TILE_WIDTH];
    __shared__ float xds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float xds_1[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ float xds_2[TILE_WIDTH][TILE_WIDTH];
    __shared__ float xds_3[TILE_WIDTH][TILE_WIDTH];  
    __shared__ float xds_4[TILE_WIDTH][TILE_WIDTH];
    __shared__ float xds_5[TILE_WIDTH][TILE_WIDTH];
    __shared__ float xds_6[TILE_WIDTH][TILE_WIDTH];
    __shared__ float xds_7[TILE_WIDTH][TILE_WIDTH];
    __shared__ float xds_8[TILE_WIDTH][TILE_WIDTH];
    __shared__ float xds_9[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    // int by = blockIdx.y; 
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    int Row = ty;
    int Col = bx * TILE_WIDTH + tx;
    int B_idx = blockIdx.z;

    int numAColumns = C*K*K;
    // int numARows = M;
    int numBColumns = W_out*H_out;
    int numBRows = C*K*K;
    int numCColumns = W_out*H_out;
    // int numCRows = M;
    int KK = K*K;

    float Cvalue = 0;
    float Cvalue_1 = 0;
    float Cvalue_2 = 0;
    float Cvalue_3 = 0;
    float Cvalue_4 = 0;
    float Cvalue_5 = 0;
    float Cvalue_6 = 0;
    float Cvalue_7 = 0;
    float Cvalue_8 = 0;
    float Cvalue_9 = 0;
    int B_STRIDE = B/STRIDE;
    int H_out_idx = Col/W_out;
    int W_out_idx = Col%W_out;

    for (int ch = 0; ch < ceil(numAColumns/(float)TILE_WIDTH); ++ch) {
        wds[ty][tx] = 0;
        if ((ch*TILE_WIDTH+tx)<numAColumns){
            wds[ty][tx] = w4d(Row, ch*TILE_WIDTH + tx);
        }
        if ((ch*TILE_WIDTH+ty)<numBRows && Col<numBColumns){ 
            xds[ty][tx] =  x4d(B_idx, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_1[ty][tx] = x4d(B_idx+B_STRIDE, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_2[ty][tx] = x4d(B_idx+B_STRIDE*2, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_3[ty][tx] = x4d(B_idx+B_STRIDE*3, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_4[ty][tx] = x4d(B_idx+B_STRIDE*4, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_5[ty][tx] = x4d(B_idx+B_STRIDE*5, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_6[ty][tx] = x4d(B_idx+B_STRIDE*6, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_7[ty][tx] = x4d(B_idx+B_STRIDE*7, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_8[ty][tx] = x4d(B_idx+B_STRIDE*8, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
            xds_9[ty][tx] = x4d(B_idx+B_STRIDE*9, (ch*TILE_WIDTH + ty)/(KK), H_out_idx+(ch*TILE_WIDTH + ty)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty)%K);
        }
        if ((ch*TILE_WIDTH+ty+M)<numBRows && Col<numBColumns && ty+M < TILE_WIDTH){ 
            xds[ty+M][tx] =  x4d(B_idx, (ch*TILE_WIDTH+ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH+ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH+ty+M)%K);
            xds_1[ty+M][tx] = x4d(B_idx+B_STRIDE, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_2[ty+M][tx] = x4d(B_idx+B_STRIDE*2, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_3[ty+M][tx] = x4d(B_idx+B_STRIDE*3, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_4[ty+M][tx] = x4d(B_idx+B_STRIDE*4, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_5[ty+M][tx] = x4d(B_idx+B_STRIDE*5, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_6[ty+M][tx] = x4d(B_idx+B_STRIDE*6, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_7[ty+M][tx] = x4d(B_idx+B_STRIDE*7, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_8[ty+M][tx] = x4d(B_idx+B_STRIDE*8, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
            xds_9[ty+M][tx] = x4d(B_idx+B_STRIDE*9, (ch*TILE_WIDTH + ty+M)/(KK), H_out_idx+(ch*TILE_WIDTH + ty+M)%(KK)/K, W_out_idx+(ch*TILE_WIDTH + ty+M)%K);
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k+=2) {
            Cvalue += wds[ty][k] * xds[k][tx];
            Cvalue_1 += wds[ty][k] * xds_1[k][tx];
            Cvalue_2 += wds[ty][k] * xds_2[k][tx];
            Cvalue_3 += wds[ty][k] * xds_3[k][tx];
            Cvalue_4 += wds[ty][k] * xds_4[k][tx];
            Cvalue_5 += wds[ty][k] * xds_5[k][tx];
            Cvalue_6 += wds[ty][k] * xds_6[k][tx];
            Cvalue_7 += wds[ty][k] * xds_7[k][tx];
            Cvalue_8 += wds[ty][k] * xds_8[k][tx];
            Cvalue_9 += wds[ty][k] * xds_9[k][tx];
            Cvalue += wds[ty][k+1] * xds[k+1][tx];
            Cvalue_1 += wds[ty][k+1] * xds_1[k+1][tx];
            Cvalue_2 += wds[ty][k+1] * xds_2[k+1][tx];
            Cvalue_3 += wds[ty][k+1] * xds_3[k+1][tx];
            Cvalue_4 += wds[ty][k+1] * xds_4[k+1][tx];
            Cvalue_5 += wds[ty][k+1] * xds_5[k+1][tx];
            Cvalue_6 += wds[ty][k+1] * xds_6[k+1][tx];
            Cvalue_7 += wds[ty][k+1] * xds_7[k+1][tx];
            Cvalue_8 += wds[ty][k+1] * xds_8[k+1][tx];
            Cvalue_9 += wds[ty][k+1] * xds_9[k+1][tx];
        }
        __syncthreads();

    }

    if(Col<numCColumns){
        y4d(B_idx, Row, Col) =  Cvalue;
        y4d(B_idx+B_STRIDE, Row, Col) = Cvalue_1;
        y4d(B_idx+B_STRIDE*2, Row, Col) = Cvalue_2;
        y4d(B_idx+B_STRIDE*3, Row, Col) = Cvalue_3;
        y4d(B_idx+B_STRIDE*4, Row, Col) = Cvalue_4;
        y4d(B_idx+B_STRIDE*5, Row, Col) = Cvalue_5;
        y4d(B_idx+B_STRIDE*6, Row, Col) = Cvalue_6;
        y4d(B_idx+B_STRIDE*7, Row, Col) = Cvalue_7;
        y4d(B_idx+B_STRIDE*8, Row, Col) = Cvalue_8;
        y4d(B_idx+B_STRIDE*9, Row, Col) = Cvalue_9;
    }

#undef TILE_WIDTH
#undef STRIDE
#undef y4d
#undef x4d 
#undef w4d  
}


/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";


    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // printf("B:%d\n", B);
    // printf("M:%d\n", M);
    // printf("C:%d\n", C);
    // printf("H:%d\n", H);
    // printf("W:%d\n", W);
    // printf("K:%d\n", K);

    if(12==M){
#define TILE_WIDTH 29
#define STRIDE  5
        dim3 gridDim_mul(ceil((W_out*H_out)/(TILE_WIDTH*1.0)), 1, B/STRIDE);
        dim3 blockDim(TILE_WIDTH, M, 1);
        matrix_multiplication_1<<<gridDim_mul, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K, W_out, H_out);
#undef TILE_WIDTH
#undef STRIDE
    }
    else{
#define TILE_WIDTH 32
#define STRIDE  10
        cudaMemcpyToSymbol(w_1, w.dptr_, 14112*sizeof(float));
        dim3 gridDim_mul(ceil((W_out*H_out)/(TILE_WIDTH*1.0)), 1, B/STRIDE);
        dim3 blockDim(TILE_WIDTH, M, 1);
        matrix_multiplication_2<<<gridDim_mul, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K, W_out, H_out);
#undef TILE_WIDTH
#undef STRIDE
    }
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
