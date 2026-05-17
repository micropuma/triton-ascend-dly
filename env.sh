# python 基础环境
pip install ninja cmake wheel pybind11
python -m pip install -U pip setuptools wheel

# 安装torch和torch_npu环境
python -m pip install -U numpy PyYAML attrs decorator psutil scipy pybind11
dnf install -y python3-devel

pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu==2.7.1

# 系统环境  
dny install -y clang-15 lld-15 ccache

export LLVM_INSTALL_PREFIX=/data/compiler-workspace/llvm-project/llvm-install

LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_PLUGIN_DIRS=./ascend \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install