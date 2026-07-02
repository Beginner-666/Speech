export KALDI_ROOT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichun/speech/speech/downloads/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/OpenBLAS/install/lib:${LD_LIBRARY_PATH:-}
export CONDA_CUDA_ROOT=/inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/conda
export LD_LIBRARY_PATH=$KALDI_ROOT/../cuda-lib-shims:$CONDA_CUDA_ROOT/lib:$CONDA_CUDA_ROOT/targets/x86_64-linux/lib:$CONDA_CUDA_ROOT/lib/python3.11/site-packages/nvidia/cublas/lib:$CONDA_CUDA_ROOT/lib/python3.11/site-packages/nvidia/cusolver/lib:$CONDA_CUDA_ROOT/lib/python3.11/site-packages/nvidia/cusparse/lib:$CONDA_CUDA_ROOT/lib/python3.11/site-packages/nvidia/curand/lib:$CONDA_CUDA_ROOT/lib/python3.11/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
export FST_PLUGIN_PATH=$KALDI_ROOT/tools/openfst/lib/fst:${FST_PLUGIN_PATH:-}
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export PYTHONUNBUFFERED=1
