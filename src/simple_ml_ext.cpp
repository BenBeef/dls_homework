#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t loops = (m + batch - 1) / batch;
    const float *X_b;
    const unsigned char *y_b;
    float *Z = (float *) malloc(sizeof(float) * batch * k);  // batch * k
    unsigned char *I_y = (unsigned char *) malloc(sizeof(char) * batch * k);  // batch * k
    float *X_t = (float *) malloc(sizeof(float) * batch * n);  // batch * k
    float *G = (float *) malloc(sizeof(float) * n * k);  // batch * k
    if (Z == NULL || I_y == NULL || X_t == NULL || G == NULL) {
        goto free_all;
    }
    for (size_t b = 0; b < loops; b++) {
        X_b = X + b * batch * n;         //    X_b, y_b = X[i*batch:(i+1)*batch], y[i*batch:(i+1)*batch]
        y_b = y + b * batch;
        for (size_t i = 0; i < batch; i++) {         // Z = np.matmul(X_b, theta)   # bt x n   n x k = bt x k
            for (size_t j = 0; j < k; j++) {
                Z[i * k + j] = 0;
                for (size_t l = 0; l < n; l++) {
                    Z[i * k + j] += X_b[i * n + l] * theta[l * k + j];
                }
            }
        }
        for (size_t i = 0; i < batch; i++) {         //  Z = np.exp(Z)   bt x k
            for (size_t j = 0; j < k; j++) {
                Z[i * k + j] = expf(Z[i * k + j]);
            }
        }
        for (size_t i = 0; i < batch; i++) {         //  Z = np.divide(Z, np.sum(Z, axis=1, keepdims=True))  # bt x k
            float j_sum = 0;
            for (size_t j = 0; j < k; j++) {
                j_sum += Z[i * k + j];
            }
            for (size_t j = 0; j < k; j++) {
                Z[i * k + j] /= j_sum;
            }
        }

        for (size_t i = 0; i < batch; i++) {       // I_y = np.zeros(Z.shape)  # bt x k
            for (size_t j = 0; j < k; j++) {
                I_y[i * k + j] = 0;
            }
            I_y[i * k + size_t(y_b[i])] = 1;                 // I_y[np.arange(len(y_b)), y_b] = 1  # bt x k
        }

        for (size_t i = 0; i < batch; i++) {       // Z = np.subtract(Z, I_y)  # bt x k
            for (size_t j = 0; j < k; j++) {
                Z[i * k + j] -= float(I_y[i * k + j]);
            }
        }
        for (size_t i = 0; i < n; i++) {           // X_t = np.transpose(X_b)  # bt x n  ->  n x bt
            for (size_t j = 0; j < batch; j++) {
                X_t[i * batch + j] = X_b[j * n + i];
            }
        }
        for (size_t i = 0; i < n; i++) {          //  G = np.matmul(X_t, Z)  # n x bt   bt x k  ->  n x k
            for (size_t j = 0; j < k; j++) {
                G[i * k + j] = 0;
                for (size_t l = 0; l < batch; l++) {
                    G[i * k + j] += X_t[i * batch + l] * Z[l * k + j];
                }
                G[i * k + j] *= lr / float(batch);  // G = np.multiply(lr / batch, G)  #  n x k
            }
        }

        for (size_t i = 0; i < n; i++) {           // theta -= delta  # n x k
            for (size_t j = 0; j < k; j++) {
                theta[i * k + j] -= G[i * k + j];
            }
        }
    }

    free_all:
    free(Z);
    free(I_y);
    free(X_t);
    free(G);
    if (Z == NULL || I_y == NULL || X_t == NULL || G == NULL) {
        throw;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
