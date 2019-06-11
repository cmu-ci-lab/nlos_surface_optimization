#include "convolution_mkl.h"
#include "mkl_vsl.h"
int convolve(double* h, double* x, double* y, int nh, int nx, int ny, int iy0) {
    int status;
    VSLConvTaskPtr task;
    vsldConvNewTask1D(&task,VSL_CONV_MODE_AUTO,nh,nx,ny);
    vslConvSetStart(task, &iy0);
    status = vsldConvExec1D(task, h, 1, x, 1, y, 1);
    vslConvDeleteTask(&task);
    return status;
}
