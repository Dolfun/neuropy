#include <pybind11/pybind11.h>

int mul(int i, int j) {
    return i * j;
}

namespace py = pybind11;

PYBIND11_MODULE(hello2, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("mul", &mul, "A function that multiples two numbers");
}