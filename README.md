# M2M-TSE: Multichannel-to-Multichannel Target Sound Extraction

基于 spatial_tse 的多通道目标语音提取项目。

## 来源

从 [wesep_spatial](https://github.com/Fin0saur/wesep_spatial) 的 spatial_tse 部分 fork 并扩展。

## 结构

```
M2M_TSE/
├── wesep/                 # 核心代码
│   ├── bin/              # 训练/推理脚本
│   ├── models/           # 模型定义
│   ├── modules/          # 模块组件
│   │   ├── spatial/      # 空间特征处理
│   │   ├── separator/    # 分离器
│   │   └── ...
│   ├── dataset/          # 数据处理
│   └── utils/            # 工具函数
├── examples/spatial/     # 示例配置
├── spatial/              # 数据准备脚本
└── processor_spatial.py  # 数据处理器
```

## 论文参考

- [Multichannel-to-Multichannel Target Sound Extraction Using Direction and Timestamp Clues](https://arxiv.org/abs/2409.12415) (2024)