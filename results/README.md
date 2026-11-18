# 2018 训练与评估原始结果

每个 ZIP 对应一个正式实验目录，保留训练配置、状态、曲线数据、评估指标和逐行预测。
所有 ZIP 使用 DEFLATE level 6，单包小于 95 MB，并已校验 CRC、文件数和未压缩总量。
`sources_results.json` 记录每个原文件的工作区相对来源、SHA256、ZIP 路径与包内成员路径；
`bundles/manifest.json` 记录各包的 SHA256 与大小。解压时按下表目录恢复。
权重、优化器、中间 checkpoint、smoke、失败 partial、运行环境副本及已取消的 2023 实验不在结果包中。

| ZIP | 原结果目录 | 文件数 | 压缩后字节 |
| --- | --- | ---: | ---: |
| [Industrial_and_Scientific_05b_cosine_ep10_20261117.zip](bundles/Industrial_and_Scientific_05b_cosine_ep10_20261117.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_cosine_ep10_20261117 | 49 | 6237319 |
| [Industrial_and_Scientific_05b_cosine_ep5_title_history_rank50_20261117.zip](bundles/Industrial_and_Scientific_05b_cosine_ep5_title_history_rank50_20261117.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_cosine_ep5_title_history_rank50_20261117 | 90 | 12119855 |
| [Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1.zip](bundles/Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_linear_ep3_ep5_title_history_20261117_retry1 | 60 | 7117965 |
| [Industrial_and_Scientific_05b_qwen_retrieval_20261118_004220.zip](bundles/Industrial_and_Scientific_05b_qwen_retrieval_20261118_004220.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_qwen_retrieval_20261118_004220 | 63 | 63746680 |
| [Industrial_and_Scientific_05b_rl_beta0p01_20261118.zip](bundles/Industrial_and_Scientific_05b_rl_beta0p01_20261118.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_rl_beta0p01_20261118 | 85 | 14635638 |
| [Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1.zip](bundles/Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_05b_rl_from_linear_ep3_20261117_retry1 | 75 | 13637024 |
| [Industrial_and_Scientific_RQVAE_2K_20261118.zip](bundles/Industrial_and_Scientific_RQVAE_2K_20261118.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_RQVAE_2K_20261118 | 157 | 42237193 |
| [Industrial_and_Scientific_SASRec_Baseline_20261118.zip](bundles/Industrial_and_Scientific_SASRec_Baseline_20261118.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_SASRec_Baseline_20261118 | 23 | 962831 |
| [Industrial_and_Scientific_enddonkey.zip](bundles/Industrial_and_Scientific_enddonkey.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_enddonkey | 22 | 2609824 |
| [Industrial_and_Scientific_enddonkey_05b_linear.zip](bundles/Industrial_and_Scientific_enddonkey_05b_linear.zip) | results/checkpoints/MiniOneRec/Industrial_and_Scientific_enddonkey_05b_linear | 23 | 2612150 |
| [Industrial_hybrid_beta001_inference_20261118-metadata.zip](bundles/Industrial_hybrid_beta001_inference_20261118-metadata.zip) | results/checkpoints/MiniOneRec/Industrial_hybrid_beta001_inference_20261118 | 117 | 2285508 |
| [Industrial_hybrid_beta001_inference_20261118-test.zip](bundles/Industrial_hybrid_beta001_inference_20261118-test.zip) | results/checkpoints/MiniOneRec/Industrial_hybrid_beta001_inference_20261118 | 4 | 50261033 |
| [Industrial_hybrid_beta001_inference_20261118-valid.zip](bundles/Industrial_hybrid_beta001_inference_20261118-valid.zip) | results/checkpoints/MiniOneRec/Industrial_hybrid_beta001_inference_20261118 | 4 | 50933465 |
| [Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118.zip](bundles/Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118.zip) | results/checkpoints/MiniOneRec/Industrial_hybrid_beta001_unisrec_k_vs_2k_20261118 | 165 | 36157677 |
| [UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_20261117_173712.zip](bundles/UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_20261117_173712.zip) | results/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_20261117_173712 | 23 | 873771 |
| [UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep15_20261117_182158.zip](bundles/UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep15_20261117_182158.zip) | results/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep15_20261117_182158 | 22 | 840729 |
| [UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242.zip](bundles/UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242.zip) | results/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep20_20261117_181242 | 22 | 861865 |
| [UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep25_20261117_182943.zip](bundles/UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep25_20261117_182943.zip) | results/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep25_20261117_182943 | 25 | 890367 |
| [UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep50_20261117_180305.zip](bundles/UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep50_20261117_180305.zip) | results/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep50_20261117_180305 | 13 | 23320 |
| [UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep50_retry1_20261117_180455.zip](bundles/UniSRec-Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep50_retry1_20261117_180455.zip) | results/checkpoints/UniSRec/Industrial_and_Scientific_Qwen_4gpu_bs512_lr2e-3_ep50_retry1_20261117_180455 | 19 | 836620 |
| [UniSRec-industrial_hybrid_inference_ep20_export_v1.zip](bundles/UniSRec-industrial_hybrid_inference_ep20_export_v1.zip) | results/checkpoints/UniSRec/industrial_hybrid_inference_ep20_export_v1 | 1 | 9265 |
| [Video_Games-RQVAE-2026-11-14T15-04-02-357Z.zip](bundles/Video_Games-RQVAE-2026-11-14T15-04-02-357Z.zip) | results/checkpoints/rqvae/Video_Games/2026-11-14T15-04-02-357Z | 25 | 2065174 |
| [Video_Games-RQVAE-compare_10000_2026-11-14T15-37-03-631Z.zip](bundles/Video_Games-RQVAE-compare_10000_2026-11-14T15-37-03-631Z.zip) | results/checkpoints/rqvae/Video_Games/compare_10000_2026-11-14T15-37-03-631Z | 30 | 2790104 |
| [Video_Games_sid2900.zip](bundles/Video_Games_sid2900.zip) | results/checkpoints/MiniOneRec/Video_Games_sid2900 | 28 | 17787806 |
