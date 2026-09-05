# AttAcc 模拟器

本仓库包含一个基于 Python 的模拟器，用于分析由 xPU 和注意力加速器（Attention Accelerator，AttAcc）组成的异构系统在运行基于 Transformer 的生成模型（Transformer-based Generative Model，TbGM）推理时的表现。

AttAcc 是面向 TbGM 注意力层的加速器，采用基于 HBM 的存内计算（Processing-in-Memory，PIM）结构。

在模拟 xPU 与 AttAcc 系统时，该模拟器会输出 xPU 的性能和能耗；AttAcc 的行为则通过经过适当修改的 [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2) 进行模拟。

我们在 Ramulator2 中将 AttAcc 的存储器件设置为 HBM3，并实现了 AttAcc\_bank、AttAcc\_BG 和 AttAcc\_buffer，分别表示将处理单元部署在每个 bank、每个 bank group 或每个伪通道（位于 buffer die 上）的 AttAcc 架构。

有关 AttAcc 的更多细节，请参阅发表于 [ASPLOS 2024](https://www.asplos-conference.org/asplos2024) 的论文 [**AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference**](https://dl.acm.org/doi/10.1145/3620665.3640422)。

## 前置条件

- Python
- cmake、g++ 和 clang++（用于构建 Ramulator2）

AttAcc 模拟器已在以下系统环境中完成测试：

- 操作系统：Ubuntu 22.04.3 LTS（内核 6.1.45）
- 编译器：g++ 12.3.0
- Python 3.8.8

我们采用了与原始 Ramulator 2.0 类似的构建系统（CMake），它会自动下载以下外部库：

- [argparse](https://github.com/p-ranav/argparse)
- [spdlog](https://github.com/gabime/spdlog)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)

## 安装方法

1. 克隆 GitHub 仓库

```bash
$ git clone https://github.com/scale-snu/attacc_simulator.git
$ cd attacc_simulator
$ git submodule update --init --recursive
```

2. 构建 Ramulator2

```bash
$ bash set_pim_ramulator.sh
$ cd ramulator2
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cp ramulator2 ../ramulator2
$ cd ../../
```

## 运行方法

### 运行 GPU 模拟器

```bash
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ python main.py --system {} --gpu {} --ngpu {} --model {} --lin {} --lout {} --batch {} --pim {} --powerlimit --ffopt --pipeopt

$ python main.py --help

    ## set system configuration
    parser.add_argument("--system",  type=str, default="dgx",
            help="dgx(each GPU has 80GB HBM), \
                  dgx-cpu (In dgx-base, offloading the attention layer to cpu), \
                  dgx-attacc (dgx-base + attacc")
    parser.add_argument("--gpu", type=str, default='A100a',
            help="GPU type (A100a, A100, and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu", type=int, default=8,
            help="number of GPUs")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB).  default=80")



    ## set attacc configuration
    parser.add_argument("--pim", type=str, default='bank',
            help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--powerlimit",  action='store_true',
            help="power constraint for PIM ")
    parser.add_argument("--ffopt",  action='store_true',
            help="apply feedforward parallel optimization ")
    parser.add_argument("--pipeopt",  action='store_true',
            help="apply pipeline optimization ")


    ## set model and service environment
    parser.add_argument("--model", type=str, default='GPT-175B',
            help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word", type=int, default='2',
            help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",  type=int, default=2048,
            help="input sequence length")
    parser.add_argument("--lout",  type=int, default=128,
            help="number of generated tokens")
    parser.add_argument("--batch", type=int, default=1,
            help="batch size, default = 1")
```

### 示例

```bash
# dgx（配备 HBM3 的 A100）示例
$ python main.py --system dgx --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1

# 2xdgx（配备 HBM3 的 A100）示例
$ python main.py --system dgx --gpu A100a --ngpu 16 --model GPT-175B --lin 2048 --lout 128 --batch 1

# dgx-attacc（基于 HBM3）示例
 ## bank 级 PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim bank --powerlimit --ffopt --pipeopt

 ## bank group 级 PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim bg --powerlimit --ffopt --pipeopt

 ## buffer 级 PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim buffer --powerlimit --ffopt --pipeopt
```

## 面向 AttAcc 的 Ramulator 详情

### 运行方法

1. 为基于 Transformer 的生成模型生成 PIM 命令踪迹

```bash
$ cd ramulator2
$ cd trace_gen
$ python gen_trace_attacc_bank.py
$ python gen_trace_attacc_bg.py
$ python gen_trace_attacc_buffer.py
```

上述命令会生成 `attacc_bank.trace`、`attacc_bg.trace` 和 `attacc_buffer.trace`。它们分别是 AttAcc\_bank、AttAcc\_BG 和 AttAcc\_buffer 在单个解码器中运行 GPT-175B 注意力层时的踪迹。

可以通过设置以下参数来修改模型、批大小和请求配置：

```python
  parser.add_argument("-dh", "--dhead", type=int, default=128,
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=1,
                      help="Number of heads, default=1")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096,
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2,
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace",
                      help="output path")
```

2. 运行 Ramulator-AttAcc

```bash
$ ./ramulator2 -f attacc_bank.yaml
$ ./ramulator2 -f attacc_bg.yaml
$ ./ramulator2 -f attacc_buffer.yaml
```

程序会输出 DRAM/PIM 请求总数以及经过的内存周期总数（`memory_system_cycles`）。

命令日志将生成在 `log` 目录中。

### 带功耗约束的 AttAcc 建模

我们通过增大连续 MAC 命令之间的延迟（`nCCDAB`、`nCCDSB`），将 DRAM 的功耗约束体现到 AttAcc 模型中。

这些延迟根据激活操作和读取操作的能耗计算得出。

如需在无功耗约束（No Power Constraint，NPC）的情况下评估 AttAcc，请在 YAML 配置文件中取消注释 `preset: HBM3_5.2Gbps_NPC`，并注释掉 `preset: HBM3_5.2Gbps`。

## 联系方式

Jaehyun Park：jhpark@scale.snu.ac.kr

Jaewan Choi：jwchoi@scale.snu.ac.kr
