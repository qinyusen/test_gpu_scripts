# GPU Training Server Test Suite

面向 NVIDIA 数据中心 GPU（H100 / H200 / B200 / B300）训练服务器的自动化测试与基准测试工具集。
涵盖硬件检测、健康诊断、带宽基准、计算吞吐、多卡通信、压力稳定性、RDMA 网络和训练模拟。

> **支持 GPU 架构：** Hopper (H100/H200) · Blackwell (B200/B300)
> 系统自动检测 GPU 型号并使用对应的规格参数进行基准对比。

---

## 目录

- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [依赖安装](#依赖安装)
- [使用方法](#使用方法)
- [测试模块详解](#测试模块详解)
- [配置说明](#配置说明)
- [测试 SOP（标准操作流程）](#测试-sop标准操作流程)
- [报告输出](#报告输出)
- [故障排除](#故障排除)

---

## 项目结构

```
servertest/
├── h200_tester.py              # 主入口：CLI + 交互式菜单
├── install_deps.sh             # 一键安装三方工具
├── configs/
│   └── default.yaml            # 默认配置
├── modules/
│   ├── gpu_specs.py            # GPU 规格数据库 (H100/H200/B200/B300)
│   ├── gpu_info.py             # GPU 检测 & 信息
│   ├── health_check.py         # 健康诊断
│   ├── benchmark.py            # 内存带宽 + 计算吞吐
│   ├── nccl_test.py            # NCCL 多卡通信
│   ├── stress_test.py          # GPU 压力/稳定性
│   ├── rdma_test.py            # RDMA/InfiniBand
│   ├── training_sim.py         # 训练模拟
│   └── report.py               # 报告生成
├── requirements.txt
└── 调研.md                     # 行业框架调研
```

---

## 环境要求

### 最低要求（基础诊断）

| 项目 | 要求 |
|---|---|
| OS | Ubuntu 22.04 / RHEL 8+ / Rocky 8+ |
| Python | 3.10+ |
| NVIDIA Driver | ≥ 535（H200 支持） |
| CUDA | ≥ 12.1 |
| nvidia-smi | 必须可用 |
| pip 包 | rich, pyyaml |

### 完整测试（推荐）

| 项目 | 要求 |
|---|---|
| GPU | ≥ 1 张 NVIDIA 数据中心 GPU（H100/H200/B200/B300 SXM） |
| MPI | OpenMPI ≥ 4.1 |
| RDMA | Mellanox ConnectX-7 / BlueField |
| nvbandwidth | 源码编译安装 |
| nccl-tests | 源码编译安装 |
| gpu-burn | 源码编译安装 |
| PyTorch | ≥ 2.1（含 CUDA 支持） |
| transformers | ≥ 4.30（训练模拟可选） |

---

## 快速开始

```bash
# 1. 克隆项目到服务器
git clone git@github.com:qinyusen/test_gpu_scripts.git
cd test_gpu_scripts

# 2. 安装依赖（需要 root）
sudo bash install_deps.sh

# 3. 运行交互式测试（自动检测 GPU 型号）
python3 h200_tester.py

# 4. 或一键全量测试
python3 h200_tester.py --test all

# 5. 手动指定 GPU 型号（跳过自动检测）
python3 h200_tester.py --gpu-type b200 --test all
```

---

## 依赖安装

### 一键安装（推荐）

```bash
sudo bash install_deps.sh
```

该脚本自动完成：
1. 安装系统包（build-essential, openmpi, infiniband-diags, perftest）
2. 源码编译 nvbandwidth → `$INSTALL_DIR/nvbandwidth/`
3. 源码编译 nccl-tests → `$INSTALL_DIR/nccl-tests/build/`
4. 源码编译 gpu-burn → `$INSTALL_DIR/gpu-burn/`
5. 安装 Python 包（rich, pyyaml）
6. 检查 DCGM 和 RDMA 工具状态

默认安装目录 `/opt/h200-test-tools`，可通过环境变量自定义。

### 自定义安装目录

```bash
sudo H200_TOOLS_DIR=/data/tools bash install_deps.sh
```

### 手动安装单项

```bash
TOOLS=/opt/h200-test-tools

# nvbandwidth
git clone https://github.com/NVIDIA/nvbandwidth.git $TOOLS/nvbandwidth
cd $TOOLS/nvbandwidth && mkdir build && cd build
cmake .. && make -j$(nproc)

# nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git $TOOLS/nccl-tests
cd $TOOLS/nccl-tests
make MPI=1 MPI_HOME=/usr CUDA_HOME=/usr/local/cuda -j$(nproc)

# gpu-burn
git clone https://github.com/wilicc/gpu-burn.git $TOOLS/gpu-burn
cd $TOOLS/gpu-burn && make
```

---

## 使用方法

### 交互式菜单（默认模式）

```bash
python3 h200_tester.py
```

显示带编号的测试菜单，输入数字选择测试：

```
 [1]  GPU Information
 [2]  Health Check
 [3]  Memory Benchmark (nvbandwidth)
 [4]  Compute Benchmark
 [5]  NCCL Multi-GPU Test
 [6]  GPU Stress Test (gpu-burn)
 [7]  RDMA/IB Test
 [8]  Training Simulation
 [9]  Full Test Suite (All Tests)
 [0]  Generate Report
 [q]  Quit
```

### 命令行模式（脚本化/批量）

```bash
# 单项测试
python3 h200_tester.py --test gpu-info
python3 h200_tester.py --test health
python3 h200_tester.py --test benchmark --type memory
python3 h200_tester.py --test benchmark --type compute --dtype bf16
python3 h200_tester.py --test nccl
python3 h200_tester.py --test stress
python3 h200_tester.py --test rdma
python3 h200_tester.py --test training

# 全量测试
python3 h200_tester.py --test all

# GPU 型号控制
python3 h200_tester.py --gpu-type auto --test all       # 自动检测（默认）
python3 h200_tester.py --gpu-type h200 --test all        # 强制指定 H200
python3 h200_tester.py --gpu-type b300 --test benchmark  # 强制指定 B300

# 指定自定义配置
python3 h200_tester.py --config /path/to/config.yaml --test all
```

### GPU 自动检测

系统启动时自动运行 `nvidia-smi --query-gpu=name` 检测 GPU 型号，匹配规则：

| GPU 名称关键词 | 识别为 | 使用规格 |
|---|---|---|
| `H100` | H100 SXM5 | Hopper, 80GB HBM3, 3.4 TB/s |
| `H200` | H200 SXM | Hopper, 141GB HBM3e, 4.8 TB/s |
| `B200` | B200 SXM | Blackwell, 180GB HBM3e, 8 TB/s |
| `B300` | B300 SXM | Blackwell Ultra, 288GB HBM3e, 8 TB/s |

检测后自动选择对应的：
- **峰值 TFLOPS**（用于计算吞吐效率百分比）
- **内存带宽峰值**（用于带宽效率百分比）
- **TDP 功耗**（用于健康检查功耗阈值）
- **NVLink 带宽**（用于 NCCL 测试最低带宽阈值）

如果检测失败或不匹配，所有峰值显示为 `N/A`，测试仍可正常运行。

---

## 测试模块详解

### 1. GPU Information（GPU 信息）

检测所有 GPU 的硬件规格和运行状态。

| 指标 | 说明 |
|---|---|
| 型号 | 自动检测并确认 GPU 型号（H100/H200/B200/B300） |
| VRAM | 总量 / 已用 / 空闲 |
| 温度 | 实时温度 |
| 功耗 | 实时功耗 / 功耗上限 |
| 时钟频率 | SM 时钟 / 内存时钟 |
| PCIe | 链路代数和宽度（期望 Gen5 x16） |
| Persistence Mode | 应开启 |
| ECC 错误 | 单比特 / 双比特计数 |
| NVLink 拓扑 | 显示 `nvidia-smi topo -m` 输出 |

### 2. Health Check（健康诊断）

全面检查 GPU 和系统健康状态，输出 PASS/WARN/FAIL 评级。功耗上限根据 GPU 型号自动设定。

| 检查项 | 判定标准 |
|---|---|
| 温度 | < 80°C PASS, < 90°C WARN, ≥ 90°C FAIL |
| 功耗 | ≤ 功耗上限 ×1.05 PASS（上限自动匹配 GPU TDP） |
| ECC 单比特 | ≤ 100 WARN, > 100 WARN |
| ECC 双比特 | = 0 PASS, > 0 FAIL |
| PCIe 链路 | ≥ Gen4 x8 PASS |
| 时钟频率 | > 0 PASS |
| 节流 | 无活跃节流原因 PASS |
| Persistence Mode | Enabled PASS |
| Hugepages | 已配置 WARN |
| Swap | 已禁用 PASS |
| 文件描述符 | soft ≥ 65536 WARN |
| InfiniBand | 设备存在 WARN |
| NCCL 环境变量 | 列出已设置的变量 |

### 3. Memory Benchmark（内存带宽）

优先使用 NVIDIA 官方 nvbandwidth，不可用时 PyTorch fallback。

**nvbandwidth 模式测试项：**
- `host_to_device_memcpy_read_ce` — H2D 带宽（PCIe）
- `device_to_host_memcpy_write_ce` — D2H 带宽（PCIe）
- `device_to_device_memcpy_write_ce` — D2D 带宽（NVLink）
- `device_to_device_memcpy_read_ce` — D2D 读带宽
- `device_to_device_bidirectional_sm` — D2D 双向带宽

**GPU 参考值（D2D 峰值带宽）：** H100: 3,400 GB/s | H200: 4,800 GB/s | B200/B300: 8,000 GB/s

**效率评级：** ≥ 80% 绿色, 50-80% 黄色, < 50% 红色

### 4. Compute Benchmark（计算吞吐）

使用 PyTorch matmul 测试各精度 GEMM 吞吐量。峰值 TFLOPS 根据 GPU 型号自动匹配。

| 精度 | H100/H200 峰值 | B200 峰值 | B300 峰值 |
|---|---|---|---|
| FP32 | 67 TFLOPS | 90 TFLOPS | 125 TFLOPS |
| TF32 | 495 TFLOPS | 1,125 TFLOPS | 1,750 TFLOPS |
| FP16 | 990 TFLOPS | 2,250 TFLOPS | 3,500 TFLOPS |
| BF16 | 990 TFLOPS | 2,250 TFLOPS | 3,500 TFLOPS |
| FP8 | 1,979 TFLOPS | 4,500 TFLOPS | 7,000 TFLOPS |

默认配置：4096×4096 矩阵，10 次 warmup，100 次迭代。

### 5. NCCL Multi-GPU Test（多卡通信）

优先使用官方 nccl-tests（通过 mpirun 调用），不可用时 torchrun fallback。

| 操作 | 说明 |
|---|---|
| AllReduce | 最常用的集合通信 |
| AllToAll | 模型并行关键操作 |
| Broadcast | 参数同步 |
| ReduceScatter | 可选 |
| AllGather | 可选 |
| SendRecv | 可选 |

默认测试数据量范围 8B ~ 256MB，5 次 warmup，20 次迭代。

**NVLink 参考带宽：** H100/H200 ≥ 360 GB/s | B200/B300 ≥ 720 GB/s（40% NVLink 峰值）

### 6. GPU Stress Test（压力测试）

使用 gpu-burn 进行长时满载测试，验证热稳定性和内存正确性。

| 参数 | 默认值 | 说明 |
|---|---|---|
| duration_sec | 60 | 测试时长（秒） |
| use_tensor_cores | true | 使用 Tensor Core |
| memory_pct | 90 | 内存占用比例 |

### 7. RDMA/IB Test（网络测试）

检测 InfiniBand 设备并测量带宽和延迟。

| 测试 | 工具 |
|---|---|
| 写带宽 | ib_write_bw |
| 读带宽 | ib_read_bw |
| 写延迟 | ib_write_lat |
| 读延迟 | ib_read_lat |

**参考阈值：** 带宽 ≥ 50 GB/s, 延迟 ≤ 10 μs

### 8. Training Simulation（训练模拟）

使用真实或合成模型模拟训练负载。

| 模式 | 说明 |
|---|---|
| 真实模型 | 加载 HuggingFace GPT-2（需安装 transformers） |
| 合成模型 | 6 层 Transformer（无需额外依赖） |

输出：tokens/sec、步时、峰值显存、最终 loss。

---

## 配置说明

配置文件路径：`configs/default.yaml`

```yaml
# GPU type: auto-detect or override to h100/h200/b200/b300
gpu_type: auto

tools:
  install_dir: /opt/h200-test-tools    # 三方工具安装目录

benchmark:
  memory:
    nvbandwidth_buffer_mb: 512          # nvbandwidth 缓冲区大小
    nvbandwidth_samples: 3              # nvbandwidth 采样次数
  compute:
    dtypes: [fp32, tf32, fp16, bf16, fp8]
    matrix_size: 4096                   # GEMM 矩阵维度
    warmup: 10
    iterations: 100

health:
  temp_warning: 80                      # 温度警告阈值 °C
  temp_critical: 90                     # 温度严重阈值 °C
  power_limit: null                     # null = 自动匹配 GPU TDP

nccl:
  min_bandwidth_gbps: null              # null = 40% GPU NVLink 峰值
  test_allreduce: true
  test_alltoall: true
  test_broadcast: true

stress:
  duration_sec: 60                     # 压力测试时长
  use_tensor_cores: true

rdma:
  min_bandwidth_gbps: 50              # RDMA 最低可接受带宽
  max_latency_us: 10                  # RDMA 最大可接受延迟
  msg_size: 65536                     # 测试消息大小

training:
  model: gpt2                          # HuggingFace 模型名
  batch_size: 8
  seq_length: 2048
  num_steps: 50
  dtype: bf16

report:
  output_dir: ./reports
  format: json                         # json 或 html
```

---

## 测试 SOP（标准操作流程）

### SOP-1: 新服务器到货验收

**适用场景：** GPU 服务器首次上架，需要确认硬件完整可用。支持 H100/H200/B200/B300。

```
步骤 1: 环境准备
├── 确认 OS 已安装（Ubuntu 22.04 推荐）
├── 确认 NVIDIA 驱动已安装（nvidia-smi 可用）
├── 执行: sudo bash install_deps.sh
└── 确认所有工具安装成功

步骤 2: GPU 信息核对
├── python3 h200_tester.py --test gpu-info
├── 确认: 系统自动检测到 GPU 型号
├── 核对: GPU 数量是否与采购规格一致
├── 核对: 型号与预期一致（H100/H200/B200/B300）
├── 核对: VRAM 总量符合规格（H100: 80GB, H200: 141GB, B200: 180GB, B300: 288GB）
├── 核对: PCIe Gen5 x16
└── 核对: NVLink 拓扑显示正确

步骤 3: 健康诊断
├── python3 h200_tester.py --test health
├── 确认: 所有检查项 PASS
├── 重点关注: ECC 双比特错误 = 0
├── 重点关注: 温度 < 80°C（空载）
├── 重点关注: 无节流原因
└── 如有 WARN/FAIL: 记录问题，联系供应商

步骤 4: 内存带宽基准
├── python3 h200_tester.py --test benchmark --type memory
├── 确认: D2D 带宽效率 ≥ 90%（自动与 GPU 峰值对比）
└── 低于 80%: 检查散热/ECC/固件版本

步骤 5: 计算吞吐基准
├── python3 h200_tester.py --test benchmark --type compute
├── 确认: 各精度 TFLOPS ≥ 峰值的 80%（自动与 GPU 规格对比）
└── 异常低: 检查功耗限制、时钟频率、驱动版本

步骤 6: NCCL 多卡通信
├── python3 h200_tester.py --test nccl
├── 确认: AllReduce/AllToAll bus bandwidth ≥ 最低阈值（自动根据 NVLink 带宽计算）
└── 异常低: 检查 NVLink 连接、NVSwitch 状态

步骤 7: 压力稳定性
├── 修改 configs/default.yaml: stress.duration_sec = 600（10分钟）
├── python3 h200_tester.py --test stress
├── 确认: 所有 GPU PASS
├── 测试期间观察: 温度不超 90°C
└── 测试期间观察: 无 ECC 错误增长

步骤 8: 生成验收报告
├── python3 h200_tester.py --test all
├── 检查 reports/ 目录下的报告文件
└── 保存报告作为验收依据
```

**验收通过标准：**
- 8 项测试全部无 FAIL
- 内存带宽效率 ≥ 90%（自动与检测到的 GPU 峰值对比）
- 计算吞吐效率 ≥ 80%
- NCCL 带宽 ≥ 最低阈值（自动计算）
- 压力测试 10 分钟无错误

---

### SOP-2: 日常巡检

**适用场景：** 已投产服务器的周期性健康检查。

```
频率: 每周一次 或 维护窗口

步骤:
1. python3 h200_tester.py --test health
2. 重点关注:
   - ECC 错误是否增长
   - 温度是否异常升高
   - PCIe 链路是否降级
   - 节流是否出现
3. 异常处理:
   - ECC 双比特错误 > 0: 立即隔离 GPU，联系 NVIDIA
   - 持续节流: 检查散热（风扇/液冷）
   - PCIe 降级: 重新插拔或更换 riser cable
```

---

### SOP-3: 多节点集群验收

**适用场景：** 多台 GPU 服务器组成训练集群，验证节点间通信。

```
前置条件: 每台单节点已通过 SOP-1

步骤 1: 单节点验收
├── 在每台节点上执行 SOP-1
└── 确保所有单节点测试通过

步骤 2: RDMA 网络测试
├── python3 h200_tester.py --test rdma
├── 确认: IB 设备被识别
├── 确认: 端口状态 Active
├── 确认: 写带宽 ≥ 50 GB/s
├── 确认: 延迟 ≤ 10 μs
└── 异常: 检查 IB 线缆、交换机配置、子网管理器

步骤 3: 多节点 NCCL 测试
├── 在每个节点上配置:
│   export MASTER_ADDR=<主节点IP>
│   export MASTER_PORT=29500
│   export NCCL_SOCKET_IFNAME=ib0    # IB 网卡名
│   export NCCL_DEBUG=INFO
├── 运行 nccl-tests 手动测试:
│   mpirun -np <总GPU数> -hostfile hosts \
│     /opt/h200-test-tools/nccl-tests/build/all_reduce_perf \
│     -b 8 -e 256M -f 2 -g 1 -w 5 -n 20
└── 确认: 多节点 AllReduce 带宽正常

步骤 4: 训练验证
├── python3 h200_tester.py --test training
├── 可选: 加载更大模型（如 llama 模型）
└── 确认: 训练 loss 正常下降
```

---

### SOP-4: 故障诊断

**适用场景：** 训练过程中出现异常（loss spike、GPU 掉线、OOM）。

```
步骤 1: 快速诊断
├── python3 h200_tester.py --test health
├── python3 h200_tester.py --test gpu-info
└── 记录所有 WARN/FAIL 项

步骤 2: 定位故障 GPU
├── 检查 nvidia-smi 输出
├── 关注: 温度、ECC、功耗异常的 GPU
└── 对故障 GPU 执行:
    python3 h200_tester.py --test stress
    (stress.gpus 设为故障 GPU 编号)

步骤 3: 通信排查
├── python3 h200_tester.py --test nccl
├── 如果 AllReduce 带宽异常低:
│   - 检查 NVLink 连接: nvidia-smi nvlink -s
│   - 检查 NVSwitch: nvidia-smi nvswitch -a
│   - 重置 GPU: nvidia-smi -i <id> -r
└── 如果多节点异常:
    python3 h200_tester.py --test rdma

步骤 4: 固件/驱动排查
├── nvidia-smi -q | head -20  (查看驱动/CUDA 版本)
├── 确认驱动版本满足要求（Hopper ≥ 535, Blackwell ≥ 550）
├── 确认固件版本与集群一致
└── 必要时更新: apt upgrade nvidia-driver-*
```

---

### SOP-5: 定期基准回归

**适用场景：** 固件/驱动/驱动升级后，确认性能未退化。

```
频率: 每次变更后 或 每月一次

步骤:
1. 变更前运行全量测试，保存基线报告:
   python3 h200_tester.py --test all

2. 执行变更（驱动升级/固件更新等）

3. 变更后再次运行:
   python3 h200_tester.py --test all

4. 对比两份报告:
   - 内存带宽偏差 < 5%
   - 计算吞吐偏差 < 5%
   - NCCL 带宽偏差 < 10%

5. 如性能退化:
   - 检查功耗限制是否变更
   - 检查时钟频率是否降低
   - 回滚驱动验证
```

---

## 报告输出

测试结果自动保存到 `reports/` 目录。

### JSON 格式

```bash
python3 h200_tester.py --test all
# 报告位置: ./reports/gpu_report_<timestamp>.json
```

包含所有测试的完整数据，可用于自动化分析。

### HTML 格式

```bash
python3 h200_tester.py --test all --format html --output report.html
```

生成深色主题的可视化报告，包含：
- GPU 规格概览
- 健康检查 PASS/FAIL 状态
- 内存带宽效率图表
- 计算吞吐对比（各精度 vs 峰值）
- 训练模拟指标

---

## 故障排除

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `nvidia-smi not found` | 驱动未安装 | 安装 NVIDIA 驱动（Hopper ≥ 535, Blackwell ≥ 550） |
| `nvbandwidth not found` | 未编译安装 | 运行 `install_deps.sh` 或手动编译 |
| `nccl-tests not found` | 未编译安装 | 运行 `install_deps.sh`，确认 CUDA_HOME 正确 |
| `mpirun not found` | MPI 未安装 | `apt install openmpi-bin libopenmpi-dev` |
| `gpu_burn not found` | 未编译安装 | 运行 `install_deps.sh` 或手动 `make` |
| NCCL 带宽异常低 | NVLink/NVSwitch 问题 | 检查 `nvidia-smi nvlink -s`，重新插拔 |
| 内存带宽低于预期 | ECC/散热问题 | 检查温度、确认 ECC 启用、更新固件 |
| 训练模拟 OOM | VRAM 不足 | 减小 batch_size 或 seq_length |
| RDMA 测试超时 | IB 未配置 | 检查 `ibstat`，确认 SM/子网管理器运行 |
| PyTorch 导入失败 | 未安装 torch | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| DCGM 未检测到 | 未安装 | `apt install datacenter-gpu-manager` |
| CUDA_HOME 错误 | 环境变量未设 | `export CUDA_HOME=/usr/local/cuda` |

---

## GPU 关键规格参考

系统自动检测 GPU 型号，以下为各型号参考规格（dense TFLOPS）：

| 参数 | H100 SXM5 | H200 SXM | B200 SXM | B300 SXM |
|---|---|---|---|---|
| 架构 | Hopper | Hopper | Blackwell | Blackwell Ultra |
| 计算能力 | 9.0 | 9.0 | 10.0 | 10.0 |
| HBM 容量 | 80 GB (HBM3) | 141 GB (HBM3e) | 180 GB (HBM3e) | 288 GB (HBM3e) |
| 内存带宽 | 3,400 GB/s | 4,800 GB/s | 8,000 GB/s | 8,000 GB/s |
| TDP | 700W | 700W | 1,000W | 1,200W |
| FP32 | 67 TFLOPS | 67 TFLOPS | 90 TFLOPS | 125 TFLOPS |
| TF32 (dense) | 495 TFLOPS | 495 TFLOPS | 1,125 TFLOPS | 1,750 TFLOPS |
| FP16/BF16 (dense) | 990 TFLOPS | 990 TFLOPS | 2,250 TFLOPS | 3,500 TFLOPS |
| FP8 (dense) | 1,979 TFLOPS | 1,979 TFLOPS | 4,500 TFLOPS | 7,000 TFLOPS |
| NVLink | 第 4 代, 900 GB/s | 第 4 代, 900 GB/s | 第 5 代, 1,800 GB/s | 第 5 代, 1,800 GB/s |
| PCIe | Gen5 x16 | Gen5 x16 | Gen5 x16 | Gen5 x16 |
| 最低驱动 | 535 | 535 | 550 | 550 |
| 最低 CUDA | 12.1 | 12.1 | 12.4 | 12.4 |
