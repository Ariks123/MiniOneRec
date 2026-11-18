# Video Games：单机 4 卡 H20 后续训练

本配置仅用于后续 SFT / RL，采用已选定的第 2,900 轮 SID。原商品向量、RQ-VAE 权重、清洗结果、SID 文件及报告均不改写。配置和测试本身不代表已完成 H20 实机训练。

## 本次 SFT、gradNorm 与自动验证报告

本次仅训练 Qwen2.5-1.5B Base 的 SFT，不运行 RL。使用独立镜像 `minionerec:h20-sft-monitor`、独立持久容器 `minionerec-h20-sft`，保留之前准备好的环境。

~~~bash
source /mnt/ossdata/ziweizhao/env.sh
MINIONEREC_IMAGE=minionerec:h20-sft-monitor bash docker/run_h20.sh build
bash docker/run_h20.sh sft-run
bash docker/run_h20.sh sft-status
~~~

`sft-run` 启动脱离终端的宿主监督进程，依次完成准备、预检、SFT、全量测试验证和离线报告。对话关闭不终止该进程；`sft-status` 只读状态，不停止容器。默认最多10轮、学习率3e-4、每卡batch16、累积16、全局batch1024、长度512，验证loss连续3次不改善早停。实际拼接训练样本703,463条，含SID推荐、历史到标题及双向SID/标题对齐；验证41,923条，测试41,924条。

每个优化器更新步在 `weights/checkpoints/MiniOneRec/Video_Games_sid2900/sft/monitor/metrics.jsonl` 写入真实 loss、学习率及 gradNorm，输出最近100个有效更新步与全程均值/最大值。gradNorm是DeepSpeed提供的裁剪前、去除loss scale的全模型全局L2范数；不平均四张卡的范数。裁剪阈值保留1.0，有限范数大于1不直接判错。NaN/Inf同步停止所有rank并保留失败记录；跳过更新及缺失值不混入统计，前10步全部缺少范数会停止。

监督进程每30秒采集GPU利用率/显存，每60秒同步日志和刷新报告；状态、进程号和阶段记录在上述SFT目录的 `run_status.json`，详细日志为 `supervisor.log`、`training.log` 和 `validating.log`。自动输出：

- `output/reports/Video_Games/06_sft_training/index.html`：实时训练报告及PDF、曲线、CSV/JSON。
- `output/reports/Video_Games/07_sft_validation/index.html`：全量结果验证报告及PDF、CSV/JSON。

训练结束后重载最佳checkpoint，使用训练相同的提示与SID约束，以四卡beam50评估完整测试集，报告HR/NDCG@1、3、5、10、20、50以及候选有效/唯一率和逐行覆盖。训练前后完整验证集的loss/PPL只统计答案token，不能解释为通用语言困惑度。候选合法性由约束解码保证，推荐效果由HR/NDCG衡量。报告在尚未完成时明确显示进行中、待验证或失败；只有模型检查、全量覆盖、报告JSON和PDF均通过才标记任务完成。

续训可传 `sft-run --resume-from-checkpoint /mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Video_Games_sid2900/sft/checkpoint-N`；严格核对源模型权重/tokenizer、SID、元数据、训练/验证文件和超参身份，不自动覆盖已有运行。新容器会上传完整checkpoint（含优化器分片）。已有运行要重新从头训练时用 `--run-dir` 指定新目录。

## 配置

| 项目 | SFT | RL |
|---|---|---|
| 训练设备 | 4 张 H20，各进程绑定一张卡 | 4 张 H20，同卡训练和 HF 约束 beam 生成 |
| 精度与显存 | BF16、ZeRO-2、梯度检查点、SDPA | BF16、ZeRO-2、梯度检查点、SDPA |
| 每卡 batch | 16 条样本 | 16 条 completion |
| 梯度累积 | 16 | 16 |
| 有效批量/更新步 | 1,024 条 SFT 样本 | 1,024 条 completion，即 64 个采样 prompt 组 |
| 每个 prompt 的生成数 | - | 16 |

RL 沿用 ranking 奖励与原训练流程，不启用 vLLM。每卡 train / eval batch 均须是 num_generations 的整数倍，防止生成组被拆到不同 rank。ZeRO-2 分片优化器状态和梯度；每卡仍保留模型参数，RL 还涉及参考模型，能否容纳所选底模需实机测量。

## Docker：Qwen2.5-1.5B Base 的 SFT → RL

底模固定使用官方 [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)，对应 Base。先下载到工作区，再以本地路径训练。默认全参数 SFT，随后 RL 读取扩展了 SID 词表的 SFT checkpoint。

原仓库 `sft.sh` 使用 8 个进程、每卡 batch 16、全局 batch 1024，即梯度累积 8。现在使用 Qwen2.5-1.5B，四卡默认保留每卡 batch 16，梯度累积增为 16，仍为 `4 × 16 × 16 = 1024`。可通过 `--sft-micro-batch-size 4` 或 `8` 降低单次显存用量，启动器会自动增加累积并拒绝不能整除的组合。训练轮数和学习率继续沿用原设置（SFT 10 / 3e-4；RL 2 / 1e-5），可用 `--num-epochs` / `--learning-rate` 覆盖。

`requirements-h20.txt` 从原 `requirements.txt` 提取后续训练所需依赖，保留 Torch 2.6.0、Transformers 4.57.1、Accelerate 1.10.1、DeepSpeed 0.18.0、TRL 0.24.0 等版本。ninja 使用补丁版本 1.11.1.3，修复原 1.11.1.1 wheel 元数据不能通过 pip 平台检查的问题。原依赖中的 cu118 torchrec/fbgemm_gpu、cu11/cu12 双套 NVIDIA wheel 不进入镜像；CUDA 库由 cu124 Torch 依赖统一提供。GPU 数量通过启动器配置，不写入依赖表。

镜像采用 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`（Python 3.11）。保留 nvcc、编译器、ninja 和 libaio，DeepSpeed 扩展在带 GPU 的运行环境按需编译；构建阶段不要求 GPU。镜像构建最后会实际导入项目的 SFT / RL，检查固定版本、CUDA build 和 NCCL，任一失败即中止构建。默认使用 SDPA，不额外编译 FlashAttention。

宿主机需要 NVIDIA 驱动、Docker Engine 和 NVIDIA Container Toolkit。先按 [NVIDIA 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) 验证 GPU 透传：

~~~bash
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
~~~

在本工作区执行以下命令。镜像构建不要求 GPU；当前代理的 copy 模式复用固定四卡容器，下载、转换和 dry-run 不执行 GPU 训练。本地 bind 模式仅对 preflight / gpu-smoke / sft / rl 分配 GPU：

~~~bash
cd /mnt/ossdata/ziweizhao/MiniOneRec
source ../env.sh
bash docker/run_h20.sh build
bash docker/run_h20.sh check-env
bash docker/run_h20.sh smoke-test
bash docker/run_h20.sh gpu-smoke
bash docker/run_h20.sh download-model
bash docker/run_h20.sh prepare-data
bash docker/run_h20.sh dry-run sft
bash docker/run_h20.sh preflight sft
bash docker/run_h20.sh sft

# SFT 完成后再执行 RL
bash docker/run_h20.sh preflight rl
bash docker/run_h20.sh rl
~~~

`smoke-test` 在 CPU 上用随机微型 Qwen2 和本地 tokenizer 执行 SFT、约束 SID 生成及 ReReTrainer 的 RL 更新各一步，不下载权重。`gpu-smoke` 使用真实四张 H20 验证 NCCL BF16 all-reduce，并在同一套 ZeRO-2 配置下执行 SFT / RL 更新，RL 也覆盖正式配置的 bitsandbytes 优化器。它们用于发现依赖/API 和通信问题，不能代替 Qwen2.5-1.5B 的完整训练。

本地 Docker 使用工作区绑定挂载；当前托管 Docker 代理限制 OSS 路径和卷挂载，启动器会自动选择持久命名容器加 `docker cp` 同步，缓存及 checkpoint 在容器停止后保留：

| 内容 | 默认宿主机路径 |
|---|---|
| Base 权重 | `/mnt/ossdata/ziweizhao/weights/Qwen2.5-1.5B` |
| HF / Torch / ModelScope 缓存 | `/mnt/ossdata/ziweizhao/weights/` 下对应目录 |
| CUDA 扩展与 Triton 缓存 | `/mnt/ossdata/ziweizhao/.cache/` |
| 转换后的 CSV | `MiniOneRec/data/Amazon18/posttraining/Video_Games_sid2900/` |
| SFT 最终模型 | `weights/checkpoints/MiniOneRec/Video_Games_sid2900/sft/final_checkpoint/` |
| RL 最终模型 | `weights/checkpoints/MiniOneRec/Video_Games_sid2900/rl/final_checkpoint/` |

`.dockerignore` 只允许训练代码、配置、测试和入口文件进入构建上下文，不把数据集、历史输出或权重打进镜像。容器使用独立的 16 GiB 共享内存（可通过 H20_SHM_SIZE 调整），兼容当前平台对 host IPC 的限制。默认物理卡为 0,1,2,3；例如 `H20_GPUS=4,5,6,7 bash docker/run_h20.sh sft` 可换卡，容器内部统一使用 0,1,2,3。

H20_DOCKER_MODE 可设为 auto（默认）、bind 或 copy。auto 在当前 docker-proxy.sock 环境选择 copy；copy 使用持久容器 minionerec-h20-workspace（可通过 MINIONEREC_CONTAINER 改名），只同步后续训练需要的原始文件。prepare-data 在容器中生成 CSV，download-model 把权重同步回工作区，训练退出后把对应阶段 checkpoint 同步回工作区。命令结束会停止容器，不自动删除容器或其数据。正在运行的容器拒绝第二项操作，避免中断现有训练；同一容器依次执行 SFT 和 RL。

copy 模式代码来自构建镜像。修改代码后重新 build，并指定新的 MINIONEREC_CONTAINER 名称；启动器会拒绝把旧环境当作新镜像使用，旧容器中的文件继续保留。本地已有的 Base 权重或 SFT checkpoint 会在新容器缺少模型时上传；需要重新执行 prepare-data 生成该容器的训练输入。复制失败时可以根据日志中的 docker cp 命令恢复输出。

copy 容器的 GPU 设备及共享内存大小在创建时固定；修改 H20_GPUS 或 H20_SHM_SIZE 时也请使用新的 MINIONEREC_CONTAINER 名称。

本轮实际验证记录见 [docker_h20_validation.json](../output/posttraining_preflight/docker_h20_validation.json)：65 项隔离测试通过，CPU 与四卡 H20 的真实小模型 SFT/RL 更新、NCCL BF16 all-reduce、ZeRO-2 梯度累积、代理数据复制均通过；1.5B Base 权重和数据已准备，SFT 预检通过。正式 1.5B 训练尚未启动，RL 正式预检需要先产生 SFT final_checkpoint。DeepSpeed 0.18 的参考模型 dummy optimizer 在退出清理时出现上游 ignored IndexError；实际训练步断言和退出码均通过，本轮未用它证明完整训练或分布式 checkpoint 保存成功。

已有权重时可设置 `BASE_MODEL` 为挂载工作区内的本地绝对路径，跳过 download-model。`MODEL_REVISION` 可固定 Hub commit；`HF_ENDPOINT`、`HF_TOKEN` 按环境变量传入，凭据不写入镜像。`H20_OUTPUT_ROOT` 可改 checkpoint 根目录；RL 默认读取该目录的 `sft/final_checkpoint`，也可传 `--model-path`。例如：

~~~bash
# 保持全局 batch 1024，每卡 batch 改为 8，梯度累积自动变为 32
bash docker/run_h20.sh sft --sft-micro-batch-size 8

# 预览 RL 计划或使用指定的 SFT 模型
bash docker/run_h20.sh dry-run rl
bash docker/run_h20.sh rl --model-path /mnt/ossdata/ziweizhao/weights/checkpoints/MiniOneRec/Video_Games_sid2900/sft/final_checkpoint
~~~

preflight 会核对输入哈希、checkpoint、tokenizer、固定库版本和四张 H20，失败时不启动训练。它不加载完整模型、不执行 NCCL all-reduce，也不测量显存峰值。RL 的 16 个候选必须留在同一 rank，不能直接将其每卡 batch 降为 SFT 的 4；需要更低显存时同时调整 `--num-generations`、`--rl-batch-size` 和 `--rl-eval-batch-size`，这会改变候选组大小。

## 环境（Linux CUDA 服务器）

推荐使用独立 Python 3.11 环境。采用原项目固定的 PyTorch 2.6.0、Transformers 4.57.1、Accelerate 1.10.1、TRL 0.24.0、DeepSpeed 0.18.0，避免自定义 trainer 与升级后的 API 不匹配。

先按服务器驱动安装 PyTorch CUDA 12.4 wheel，再安装精简后续训练依赖：

~~~bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install setuptools==80.9.0 wheel==0.45.1 packaging==24.1 ninja==1.11.1.3
DS_BUILD_OPS=0 DS_ACCELERATOR=cuda python -m pip install --no-build-isolation -r requirements-h20.txt
~~~

服务器还需可用的 NVIDIA 驱动和 NCCL；DeepSpeed 的 CUDA 扩展构建可能需要与 PyTorch 相匹配的 CUDA toolkit、C++ 编译器和 ninja。CPU 检查及镜像构建不能代替四卡 NCCL 实机验证。requirements-h20.txt 不混装原 requirements.txt 中 CUDA 11/12 两套 vendor wheel，也不安装 NPU 依赖。

## 只检查，不训练

以下命令只打印启动计划及缺少的输入，不导入 GPU 框架、不转换数据、不启动训练：

~~~bash
bash sft.sh --dry-run
bash rl.sh --dry-run
~~~

也可直接使用 Python：

~~~bash
python scripts/launch_h20.py sft --dry-run
python scripts/launch_h20.py rl --dry-run
~~~

缺少生成底模或转换 CSV 时，dry-run 会如实列出 pending_inputs。它只表示启动计划可解析，不表示服务器已具备全部运行条件。

## 准备后续样本

转换输入默认是 data/Amazon18/Video_Games。使用原 train / valid / test 划分，保留全部 1–20 条历史样本，不重新清洗、不抽样、不只保留最长历史。四个 SID token 全部保留。

先做只读验证：

~~~bash
python scripts/prepare_video_games_posttraining.py --check-only
~~~

需要准备训练输入时再运行转换：

~~~bash
python scripts/prepare_video_games_posttraining.py
~~~

输出到 data/Amazon18/posttraining/Video_Games_sid2900：
train.csv、valid.csv、test.csv、info.txt、preparation_manifest.json。

启动器固定核验批准的 2,900 轮完整 SID SHA-256：
ca9d730905a67cc36cd77c665a3165de823f2689660b19741df9ee884993085f

每次启动还核验转换清单、原始输入和 CSV/info 的哈希，防止误用 10,000 轮 SID 或过期样本。迁移到 Linux 后可覆盖 --input-dir / --data-dir；无需修改报告中原 Windows 绝对路径，也不依赖这些路径寻找数据。

## 模型与启动

SFT 默认读取工作区 `weights/Qwen2.5-1.5B` 下的本地生成式 CausalLM，也可用 --base-model 指定其他本地目录。启动器不直接接受 Hub 仓库 ID，请先下载。Qwen3-Embedding-4B 是编码模型，不能用于 SFT / RL。

先在 H20 服务器做预检（不加载训练模型权重、不训练）：

~~~bash
bash sft.sh --base-model /mnt/ossdata/ziweizhao/weights/Qwen2.5-1.5B --preflight-only
~~~

数据、模型路径和四卡预检通过后，以下命令才会实际启动训练：

~~~bash
bash sft.sh --base-model /mnt/ossdata/ziweizhao/weights/Qwen2.5-1.5B
bash rl.sh --model-path output/h20_4gpu/Video_Games_sid2900/sft/final_checkpoint
~~~

也支持 BASE_MODEL / SFT_MODEL 环境变量。直接裸机运行且未设置 H20_OUTPUT_ROOT 时，默认输出分别为 output/h20_4gpu/Video_Games_sid2900/sft 和 rl；Docker 默认写入上文的 weights/checkpoints 目录。RL 必须使用同一套 SID 训练出的 SFT checkpoint；SFT 在 final_checkpoint 写入 posttraining_provenance.json，RL 预检核对其中 SID / 元数据哈希，并检查 tokenizer 能将 SID 各 token 单独编码、模型词表也已扩展。词表相同不能代替同一套 SID 的训练来源证明。

输出模型与 tokenizer 均由分布式安全接口保存。主入口仍可直接调用 sft.py / rl.py，原其他数据集参数保留兼容；四卡包装脚本固定当前 Video Games 任务。GPR / ts_rec 变体与独立 evaluate.sh 未纳入此次修改。

## NCCL

不再强制设置 NCCL_IB_DISABLE=1；默认保留服务器管理员已有设置，其余交给 NCCL 自动检测。启动进程数固定为 4，保持 NCCL 后端。设备可用 --devices 或 CUDA_VISIBLE_DEVICES 指定四个不同 GPU。

如实际服务器需要禁用 IB/RoCE，可显式传 --nccl-ib-disable 1；有健康 RDMA 网络时可保持 auto 或按需传 0。它控制网络 transport，不会禁用 NVLink / PCIe P2P。启动器不硬编码网卡名或关闭 P2P。

迁移后先检查 nvidia-smi topo -m，并按服务器拓扑用 nccl-tests 检查通信；preflight-only 仅检查 NCCL 是否可用，不会执行 all-reduce 性能测试。NCCL_DEBUG=INFO 可用于首次诊断。

## 设计依据

- Accelerate 1.10.1 DeepSpeed 配置：https://huggingface.co/docs/accelerate/v1.10.1/en/usage_guides/deepspeed
- TRL 0.24 GRPOConfig：https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_config.py
- NVIDIA NCCL 环境变量：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- NCCL tests：https://github.com/NVIDIA/nccl-tests

BF16 放在原生 DeepSpeed JSON 中作为唯一配置来源；Accelerate YAML 不重复 mixed_precision，避免外部 DeepSpeed 配置与 Accelerate 配置冲突。
