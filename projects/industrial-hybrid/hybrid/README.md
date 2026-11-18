# Industrial and Scientific 草拟与验证推理

在 Amazon Reviews 2018 Industrial and Scientific 的固定数据划分上，用 3 轮上限线性 SFT 后的 MiniOneRec GRPO 模型验证三种检索器提出的商品。验证器是 Qwen2.5-0.5B，RL 训练的 KL 系数为 0.001，训练用验证集 NDCG@10 选中第 828 步的权重。用同一模型的原始受约束 beam50 作为基线。草拟、验证及前缀引导来自 SpecGR，本实验另外控制检索与原生 beam 的入选配额。[论文方法](https://arxiv.org/html/2410.02939v2#S3.SS3)

## 输入与执行环境

下表的宿主机相对路径以 `/mnt/ossdata/ziweizhao` 为根。`rl_run`、`prepared`、`qwen_run` 等变量在下方复制命令中给出完整取值。容器 `industrial-hybrid-beta001-infer-v1` 的工作目录为 `/work`，本次产物根目录为 `/run/hybrid`。正式实验导出目录是 `weights/checkpoints/MiniOneRec/Industrial_hybrid_beta001_inference_20261118/`，可视化页面在其 `report/index.html`。

| 宿主机来源 | 容器副本 | 用途 |
|---|---|---|
| `$rl_run/final_checkpoint/` | `/assets/rl/` | 含已选 checkpoint-828 的验证器、词表及生成配置 |
| `$rl_run/evaluation/predictions.jsonl` | `/assets/legacy_test.jsonl`、`/run/hybrid/reference_test_beam.jsonl` | 4,533 个测试请求的历史 beam 排名 |
| `MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey/` | `/assets/data/` | 原项目准备的 train、valid、test CSV 及准备清单 |
| `MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey/Industrial_and_Scientific.index.json` | `/assets/sid.json` | 原始商品 ID 到完整三位 SID |
| `MiniOneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy` | `/assets/raw.npy` | ANN 与 UniSRec 的基础特征，3,686 × 2,560，FP16 |
| `$prepared/manifest.json`、`$prepared/Industrial_and_Scientific.emb-qwen-td.npy.ids.json` | `/assets/raw_manifest.json`、`/assets/raw_row_ids.json` | 原始向量与连续行号假设的出处 |
| `weights/checkpoints/UniSRec/industrial_hybrid_inference_ep20_export_v1/` | `/assets/unisrec/` | UniSRec 历史编码器、商品向量、原始实现的校验样例和映射 |
| `$qwen_run/final_checkpoint/`、`$qwen_run/item_index/` | `/assets/qwen/final_checkpoint/`、`/assets/qwen/item_index/` | 联合训练后的 Qwen、256 维检索头与索引 |
| `code/amazon2018industrialandscientific-qwen-retrieval/qwen_retrieval/` | `/assets/qwen_source/qwen_retrieval/` | 现有检索器的加载、编码与索引校验代码 |
| `code/industrial-hybrid-inference/` | `/work/` | 当前分支的推理代码、训练时 prompt 和原始 beam 的处理器 |

本次使用四块 NVIDIA H20，各 97,871 MiB，驱动为 580.95.05。推理镜像标签为 `minionerec:h20-original-torch28`，镜像 ID 为 `sha256:0da0b47810fbd4027b860446cabe30a5cb31b48dca96a76c2ac0002143957eac`。Python 3.11.11、PyTorch 2.8.0+cu128、Transformers 4.57.1、NumPy 1.26.3、Faiss CPU 1.9.0.post1。实际运行时文件和版本见 `manifest.json`，三个草拟器的向量来源、行序证据和 ANN 对精确检索的审计见 `ann_audit.json`。

## 建立一份可复现的环境

在宿主机运行以下命令。新容器使用单独的名称和写入层，便于与已有运行并存。通过 `docker cp` 从客户端读取真实的工作区文件，适用于 Docker 守护进程与客户端的挂载视图不同的环境。

```bash
cd /mnt/ossdata/ziweizhao
source env.sh
project=code/industrial-hybrid-inference
rl_run=weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1/rl
qwen_run=weights/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_qwen_retrieval_20261118_004220/sft
uni_export=weights/checkpoints/UniSRec/industrial_hybrid_inference_ep20_export_v1
uni_source=code/amazon2018industrialandscientific-cosine/unisrec
prepared="$uni_source/prepared/Industrial_and_Scientific"
dataset=MiniOneRec/data/Amazon18/posttraining/Industrial_and_Scientific_enddonkey
sid=MiniOneRec/data/Amazon18/Industrial_and_Scientific_enddonkey/Industrial_and_Scientific.index.json
raw=MiniOneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy
c=industrial-hybrid-beta001-replay

docker create --name "$c" --gpus all --ipc=host -w /work --entrypoint sleep minionerec:h20-original-torch28 infinity
docker start "$c"
docker exec "$c" mkdir -p /assets/data /assets/rl /assets/qwen /assets/qwen_source /assets/unisrec /run/hybrid /work/tests
docker cp "$rl_run/final_checkpoint/." "$c":/assets/rl/
docker cp "$rl_run/evaluation/predictions.jsonl" "$c":/assets/legacy_test.jsonl
docker exec "$c" cp /assets/legacy_test.jsonl /run/hybrid/reference_test_beam.jsonl
docker cp "$dataset/." "$c":/assets/data/
docker cp "$sid" "$c":/assets/sid.json
docker cp "$raw" "$c":/assets/raw.npy
docker cp "$prepared/manifest.json" "$c":/assets/raw_manifest.json
docker cp "$prepared/Industrial_and_Scientific.emb-qwen-td.npy.ids.json" "$c":/assets/raw_row_ids.json
docker cp "$uni_export/." "$c":/assets/unisrec/
docker cp "$qwen_run/final_checkpoint" "$c":/assets/qwen/
docker cp "$qwen_run/item_index" "$c":/assets/qwen/
docker cp code/amazon2018industrialandscientific-qwen-retrieval/qwen_retrieval "$c":/assets/qwen_source/
for f in data.py npu_support.py LogitProcessor.py scripts hybrid; do
    docker cp "$project/$f" "$c":/work/
done
for f in "$project"/tests/test_hybrid*.py; do
    docker cp "$f" "$c":/work/tests/
done
wheel_dir=/tmp/industrial-hybrid-drafter-wheels
mkdir -p "$wheel_dir"
# 已有同版本 cp311 wheel 时直接执行下一条 docker cp。
python -m pip download --only-binary=:all: --no-deps --platform manylinux2014_x86_64 --python-version 311 --implementation cp --abi cp311 --dest "$wheel_dir" faiss-cpu==1.9.0.post1
docker cp "$wheel_dir" "$c":/tmp/faiss-wheels
docker exec "$c" python -m pip install --no-index --no-deps --find-links /tmp/faiss-wheels faiss-cpu==1.9.0.post1
docker exec -it -w /work "$c" bash
```

UniSRec 的现成导出来自 `weights/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242/checkpoints/QwenUniSRec-Nov-17-2026_10-13-24.pth`。需要重导出时，在 `unisrec-qwen:recbole121-torch26-20261117` 镜像中把 `$uni_source`（含 `prepared`）置于 `/src/unisrec`，将上述训练目录的 `config.yaml` 和该权重置于 `/run/uni/config.yaml`、`/run/uni/model.pth`，把项目的 `hybrid/drafter_export.py` 置于 `/tmp/drafter_export.py`，再执行以下命令。脚本用 RecBole 的原训练映射恢复历史与商品编码器，并将不同 batch 大小的输出与原模型对照。

```bash
python /tmp/drafter_export.py --source /src/unisrec --config /run/uni/config.yaml \
  --checkpoint /run/uni/model.pth --output /run/uni/export
```

## 草拟、验证与排序

| 草拟器 | 历史表示 | 商品索引 |
|---|---|---|
| `ann` | 每个历史商品的向量先 L2 归一化，取平均后再次归一化 | CPU Faiss HNSW 内积，M=16、efConstruction=80、efSearch=128。全库及一位、两位前缀各有索引。当请求覆盖整个小桶时直接计算精确内积 |
| `unisrec` | 20 轮上限训练组的第 20 轮模型，使用原 RecBole 序列编码，右侧 padding 的内部 ID 为 0 | 从原模型导出的 3,686 × 300 单位向量，对历史表示求内积，逐商品排名 |
| `qwen` | 现有联合训练的 Qwen 主干和检索头，使用当时的历史 SID prompt、词表和池化 | 完整 SID 对应 3,670 × 256 单位向量，按已有 `row_to_items.json` 展开到 3,686 个商品 ID。作为独立草拟模型与 RL 验证器同时驻留 |

原始 2,560 维向量库的生成清单和原始行 ID 文件缺失。准备代码根据连续商品 ID 0 至 3,685 推定行序，并核对文件哈希和配套目录。这一限制同时影响 ANN 与 UniSRec 的基础特征。UniSRec 导出另外核验 RecBole 内部 ID 与商品 ID 的映射；目录中的商品 0 有效。Qwen 索引由现有模型针对排序后的每条 SID 计算，能逐行核验 SID 和全部别名。

| 参数/规则 | 本次取值与作用 |
|---|---|
| 最终数量 | K ∈ {1, 5, 10, 20, 50} |
| 检索配额 | α 取 0、0.1、…、1，并包含每个 K 的首个非零配额断点 1/K。按精确分数计算 q=⌊αK⌋，相同 q 的 α 共享结果，见 `alpha_aliases` |
| 草拟数量与轮数 | ρ=2，每轮至多 δ=2K。最多三轮，依次从全库、一位允许前缀、两位允许前缀中挑选，按商品 ID 跨轮去重。保留历史中已交互的可重复购买商品 |
| 前缀宽度 | β ∈ {1, 5, 10, 20, 50}，每层对合法前缀累计 token 对数概率后保留前 β 个，分数相同按 tokenizer ID 字典序。完整生成始终使用原始宽度 50 |
| 阈值 | 固定 seed 42 抽 512 个验证请求，汇总三种草拟器和各 K 首轮候选的验证分数，取 P10、P25、P50、P75、P90，同时评估全接受。接受条件为分数严格大于阈值 |
| 验证分数 | 固定 SID 为 `<a_*><b_*><c_*>`。逐位提取原词表条件对数概率，取算术平均。目录无已确认的额外末尾消歧位，冷暖商品均保留三位。历史、换行与 EOS 排除在平均值外 |
| 批量计算 | 同一用户 prompt prefill 一次。第一位使用历史最后位置的 logits，相同历史与前缀共享已计算的概率；后两位用已知 SID 前缀做 causal forward。每块 64 个分支，独立拷贝相应历史 KV 行。物理 batch 末块填充至同样形状 |
| 混合与补位 | q=0 直接返回原始 beam 顺序。累计接受数达 q、全库已处理或轮数用尽后，取分数最高的 r≤q 个检索商品，再按完整 beam 顺序选 K−r 个可用商品。α=1 且 r=K 时跳过完整 beam。α_eff=r/K。实际 r>0 时仅对已入选商品按同一验证分数排序，r=0 保留 beam 顺序 |
| 同分与碰撞 | 草拟器按相似度降序、商品 ID 升序；检索验证同分保留轮次与原草拟顺序。beam 遇到多商品共用 SID，按训练目标频次降序、商品 ID 升序选首个尚未入选的商品。混合重排稳定，同分时保留原入选顺序，检索在前 |
| 划分与指标 | 训练 36,259、验证 4,532、测试 4,533 个请求。训练历史和目标出现过的商品为暖，其余为冷。3,686 个商品中 39 个为冷，测试含 147 条冷目标。主指标按严格商品 ID 算 HR、NDCG，每条一个目标，HR 等于 Recall；另报按 SID 命中，便于对比历史结果 |

同一检索器、K、配额、前缀宽度下先在完整验证集选阈值，按严格商品 ID 的 NDCG、HR、平均验证候选数（越少越好）依次决胜，仍相同则沿用枚举顺序。每个 K 的总冠军也从验证指标选出，测试集只评估锁定配置。质量网格可重复使用同一请求的原始分数、前缀及草拟结果，在线基准重新执行实际路由并同步计时。

## 单条历史的线上推理

在 `/work` 执行以下任一命令。`--history` 接受从早到晚排列的 1 至 10 个目录商品 ID，`--test-row` 可从零起始的测试行号恢复历史。显式配置支持 K 为 1 至 50 的整数，α 在 [0,1] 之间，ρ>0，实际每轮上限为 max(1,⌊ρK⌋)，前缀宽取正整数，最多 1 至 3 轮。阈值可以为有限数值或 `all`。结果 JSON 给出配额、实际检索比例和每个返回商品的 ID、SID、来源、验证分数、接受轮次及冷暖标记。

```bash
python -m hybrid.infer --history 5,24,100 --k 10 --drafter ann --alpha 0.4 \
  --rho 2 --prefix-beams 5 --threshold -2 --max-rounds 3 --assets /assets --device cuda:0
python -m hybrid.infer --test-row 0 --k 10 --drafter beam
python -m hybrid.infer --test-row 0 --k 10 --winner --run-dir /run/hybrid
```

`--winner` 完整采用 `report/selection.json` 中指定 K 的验证冠军。也可用 `--selection /path/to/selection.json` 指定已冻结的配置文件。冠军模式与显式路由参数互斥。

## 验证、测试、逐步跟踪与报告

以下命令在新推理容器的 `/work` 下顺序执行。若使用 `--limit` 做小批调试，另选输出目录，保证全量选择读到完整分片。

```bash
cd /work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
run_dir=/run/hybrid
python -m hybrid.manifest --run-dir "$run_dir"
python -m unittest discover -s tests -p 'test_hybrid*.py' -v
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m hybrid.drafter_audit --output "$run_dir/ann_audit.json"
python -m hybrid.audit --run-dir "$run_dir"
python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.experiment --run-dir "$run_dir" --splits valid test
python -m hybrid.sweep --run-dir "$run_dir" select
python -m hybrid.sweep --run-dir "$run_dir" quality
python -m hybrid.trace --run-dir "$run_dir" --device cuda:0
python -m hybrid.diagnostics --run-dir "$run_dir"
python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.benchmark --run-dir "$run_dir" --mode both
python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.batch_audit --run-dir "$run_dir" --assets /assets
python -m hybrid.sweep --run-dir "$run_dir" report
```

`hybrid.audit` 对照在线和重放的排名、来源、轮次，并检查历史原生 beam。质量阶段逐行比对测试集 4,533 条完整 beam50。独立 CPU 的 `drafter_audit` 在另外固定的 256 条验证历史上，报告五种 δ 各自相对全库精确余弦的候选集合召回。

`trace.json` 默认记录固定四条测试历史、K=10、α=0.4、前缀宽 5，在三种草拟器、全接受和 0 阈值下的 24 次真实在线执行。可传 `--thresholds p50 p90` 重放已校准的两个阈值，或调整 `--samples`。文件记录前缀、候选与过滤数量、各商品分数和接受轮次、实际补位及最终顺序。

性能默认抽取同一批 256 条测试请求（seed 42），每种 batch 形状预热 3 次，再计时 3 轮。单卡 batch=1 测 p50/p95/p99，单卡 batch=4 及四卡每卡 batch=4 测请求/秒、商品/秒和验证器处理的实际输入 token/秒。四卡是四个独立推理副本，每次重复按最慢副本耗时计；计时覆盖已常驻模型从历史输入到返回最终排名。分阶段计时使用独立请求批次，另报峰值显存、验证器前向次数和分支 token。完整测试网格用于质量，真实性能测量的配置名单见 `report/benchmark_configs.json`。

| 产物 | 读取方式 |
|---|---|
| `report/index.html`、`report/validation-report.html` | 验证选择、分组测试结果、图表及实测性能 |
| `report/validation.csv`、`report/selection.json`、`report/test.csv` | 验证全网格、阈值和获胜配置、各锁定组合的全量测试指标 |
| `report/outcomes.json`、`report/paired.json` | 各 K 与基线的差值和同一测试行的配对 bootstrap 区间 |
| `report/funnel.json` | 三种草拟器直接推荐的效果，以及验证冠军在草拟、筛选、重排和补位各阶段的目标商品命中数 |
| `report/predictions/validation_winners.jsonl.gz` | 每个冠军对每个测试请求的商品、SID、来源、分数、轮次和实际比例 |
| `report/predictions/batch-*.npz` | 全部锁定配置的批量逐行推荐。按 `row_ids` 定位，`*_item=-1` 表示该位置空缺，`*_round=0` 表示 beam |
| `report/benchmark_protocol.json`、`report/benchmarks.json` | 实际抽样、版本、warmup、计时协议和每个性能配置的原始汇总 |
| `report/batch_audit.json` | 在同一批 256 条测试历史上比较单请求推理、复原原始四条一批的在线推理与归档排名；含商品集合、完整顺序和 HR/NDCG 差异 |
| `manifest.json`、`audit.json`、`ann_audit.json`、`trace.json` | 输入与软件指纹、与原始实现的一致性、独立 ANN 审计、逐轮决策 |

## 比较 UniSRec 每轮取 K 和 2K

本轮继承上面的模型和数据。在容器中把前一次完整运行放在 `/run/hybrid`，新实验写入 `/run/hybrid-rho1`。UniSRec-K 以 K 为每轮上限重新构造候选和跨轮去重集合。原来的 ANN、Qwen、UniSRec-2K 候选及 beam 预测可复用。新配置的 `arm` 与 `rho` 标记草拟器和倍率，`id` 保留原配置的标识，并为新的 UniSRec-K 设置独立标识。

```bash
cd /work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
new_run=/run/hybrid-rho1
python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.rho_generate --run-dir "$new_run"
python -m hybrid.rho_quality --run-dir "$new_run" select
python -m hybrid.rho_quality --run-dir "$new_run" quality
python -m hybrid.rho_smoke --run-dir "$new_run"
python -m hybrid.rho_attribution --run-dir "$new_run"
python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.benchmark --run-dir "$new_run" --mode both
python -m torch.distributed.run --standalone --nproc-per-node=4 --module hybrid.batch_audit --run-dir "$new_run"
python -m hybrid.rho_report --run-dir "$new_run" --original-run-dir /run/hybrid
python -m hybrid.infer --run-dir "$new_run" --test-row 0 --k 50 --winner
```

固定阈值对照比较两组在原阈值、相同检索配额及前缀宽度下的精确商品指标。扩展选参把重新校准的五个分位点加入原网格，所有检索方案共享这组阈值。校准只用原先抽取的 512 条验证历史。最终冠军和每个方案的最优配置都按验证集选定。测试结果、逐项校验、模型与代码指纹保存在工作区 `weights/checkpoints/MiniOneRec/Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118/`。

将新运行目录从容器导出后，在工作区项目根目录执行以下命令打包。ZIP 的 `index.html` 展示本次实验，`previous/index.html` 保留上次发布的可视化报告。

```bash
workspace=/mnt/ossdata/ziweizhao
runs="$workspace/weights/checkpoints/MiniOneRec"
zip_name=Industrial-Hybrid-Validation-Report.zip
python -m hybrid.rho_bundle --run-dir "$runs/Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118" \
  --previous-zip "$runs/Industrial_hybrid_beta001_inference_20261118/$zip_name" \
  --destination "$workspace/MiniOneRec/report/$zip_name"
```

## Industrial 2018 全流程报告

从已完成的运行重建算法、训练、验证三份离线报告。最终方案统一每轮草拟 2K，单倍候选结果作为对照。构建器重新计算关键质量指标、核对数据和权重哈希，并检查离线目录和链接。运行需要已安装 Matplotlib 和 NumPy 的 CPU Python 环境。

```bash
workspace=/mnt/ossdata/ziweizhao
cd "$workspace/code/industrial-hybrid-inference"
source "$workspace/env.sh"
python -m project_report --workspace "$workspace" \
  --destination "$workspace/MiniOneRec/report/Industrial-2018-Algorithm-Training-Validation-2K.zip"
```

解压后从 `index.html` 进入各报告。曲线旁提供说明和 CSV，`data/provenance.json` 记录来源和校验值。

## RQ-VAE 三位 SID 重训后的 2K 对照

本轮保留原商品 ID、交互行序、切分和文本向量，在新的目录里从头训练 RQ-VAE、线性三轮 SFT、β=0.001 GRPO 和 UniSRec。RQ-VAE 使用训练可见商品的拟合子集均值做中心化，三层各有 256 个码。量化权重由暖商品重建 MSE 选定。导出保持前两位，在每个前缀桶内以最小残差距离分配第三码，产出三位且唯一的商品 SID。未中心化配方与末层 Sinkhorn 的结果留档。

训练和导出入口分别为 `scripts/train_industrial_rqvae.py`、`scripts/repair_industrial_rqvae.py` 和 `scripts/export_industrial_rqvae.py`。最终数据清单写入 `$run/data/preparation_manifest.json` 后，UniSRec 编排运行归档的四卡 20 轮配置，并校验新目录中非 SID 资产的哈希。SFT 与 GRPO 由 `docker/industrial_rqvae_posttraining.py` 接续，重用原训练主体并按新验证集选择检查点。

在独立运行目录保留新索引、同一数据切分下的 SFT 和 GRPO 权重，以及重训后导出的 UniSRec。推理主对照复用原方案按验证集锁定的五档配置。每轮草拟 2K，`α=1`，最多三轮，前缀宽度 1，补位 beam 宽度 50。K=1 继续使用 −4.032527，其余 K 全接受。附表的 K=1 阈值只用原来 512 条验证历史，按旧 UniSRec 首轮候选的拒绝比例重新校准。纯 beam 和 UniSRec 原排用于拆解生成和排序的变化。

在运行目录中保存 `catalog/Industrial_and_Scientific.index.json`、`data/`、`rl/` 和 `unisrec/`。推理启动脚本在 SFT、RL 和 UniSRec 完成并导出后继续执行。容器镜像沿用原四卡测量环境。

```bash
cd /mnt/ossdata/ziweizhao/code/industrial-hybrid-inference
bash scripts/launch_industrial_rqvae_inference.sh "$run"
```

输入校验比较旧 CSV 和新 CSV 的每一行，只允许目标和历史 SID 两列发生变化。新生成器在原四卡、每组四条的批次中核对测试 beam 与新 RL 的归档预测。各轮质量分片带资产和代码哈希。若执行中断，修复对应阶段后重新运行同一启动脚本，已校验的分片和计时结果可复用。

运行进度在 `$run/inference/stage.status`，完整比较在 `$run/inference/report/comparison.html`。质量、配对置信区间和性能对照分别见 `quality_comparison.json`、`paired_comparison.json`、`performance_comparison.json`。最终验收信号为 `comparison_status.json` 的 `status=passed` 和 `stage.status=completed`。

对比报告从全部完成的归档生成。使用带有 Matplotlib 的 CPU 环境执行 `python -m scripts.report_industrial_rqvae_rerun`，通过 `--workspace`、`--run-dir`、`--inference-dir` 和 `--destination` 指定报告来源与 ZIP 目的地。
