Video Games 可视化报告合集

打开 index.html 可离线查看所有报告、页面和表格。
01_cleaning：清洗报告，3页，1-20条历史窗口。
02_embedding_similarity：原始商品向量相似性报告，6页。
03_rqvae_training：RQ-VAE训练报告，4页，4100轮早停，最佳2900轮。
04_sid_validation：采用2900轮SID的验证报告，13页；第9-12页含与10000轮的对比，附CSV/JSON与参考文献。

训练/模型/SID原始产物目录：C:\Users\Administrator\Desktop\MiniOneRec\output\rqvae\Video_Games\2026-11-14T15-04-02-357Z
最终SID文件：data/Amazon18/Video_Games/Video_Games.index.json
原向量与SID内容诊断均不等同于用户推荐准确率；本次没有执行SFT或RL。

05_rqvae_10000_comparison：6页，原最佳2900轮与真正10000轮终点比较，含CSV/JSON。
10000轮实验目录：C:/Users/Administrator/Desktop/MiniOneRec/output/rqvae/Video_Games/compare_10000_2026-11-14T15-37-03-631Z
现有默认SID仍使用2900轮；10000轮的SID保存在该实验的endpoint_10000子目录。

当前正式采用：第2900轮SID（2026-11-15用户选定）。
选用记录：C:/Users/Administrator/Desktop/MiniOneRec/output/rqvae/Video_Games/selected_sid.json
