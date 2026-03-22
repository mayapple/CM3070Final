# MobileNet 图像特征提取原型

## 项目简介

这是一个基于 MobileNet V2 的产品图像特征提取原型，用于从产品图像中自动提取视觉特征并转换为营销卖点。

## 功能特性

- ✅ 使用 MobileNet V2 提取图像特征
- ✅ 颜色特征分析（色调、饱和度、亮度）
- ✅ 纹理特征分析（表面质感）
- ✅ 形状特征分析（设计风格）
- ✅ 特征到营销卖点的自动转换
- ✅ 批量处理支持
- ✅ 评估功能（需要标注数据）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 分析单张图像

```bash
python main.py path/to/image.jpg
```

### 2. 批量分析图像目录

```bash
python main.py path/to/image/directory --batch
```

### 3. 保存结果到指定目录

```bash
python main.py path/to/image.jpg -o results/
```

### 4. 使用标注数据进行评估

```bash
python main.py path/to/image/directory --batch -a annotations.json -o results/
```

## 项目结构

```
prototype/
├── main.py                    # 主程序入口
├── image_analyzer.py          # 图像分析主模块
├── feature_extractor.py       # 特征提取器（MobileNet）
├── selling_point_converter.py # 卖点转换器
├── evaluator.py               # 评估模块
├── utils.py                   # 工具函数
└── requirements.txt           # 依赖包
```

## 输出格式

分析结果以 JSON 格式保存，包含：

- `image_path`: 图像路径
- `extracted_features`: 提取的特征（颜色、纹理、形状）
- `selling_points`: 转换的营销卖点
- `processing_time`: 处理时间（秒）
- `detailed_features`: 详细特征信息

## 标注文件格式

用于评估的标注文件应为 JSON 格式：

```json
{
  "image_001.jpg": {
    "color_features": ["红色系", "高饱和度", "高亮度"],
    "texture_features": ["光滑表面"],
    "shape_features": ["流线型设计"]
  }
}
```

## 注意事项

1. 首次运行时会自动下载 MobileNet V2 预训练模型（约 14MB）
2. 图像会自动 resize 到 224×224 像素
3. 支持常见图像格式：JPG, PNG, BMP
4. 建议使用 GPU 加速（可选，CPU 也可运行）

## 测试

运行各模块的测试：

```bash
python utils.py
python feature_extractor.py
python selling_point_converter.py
python image_analyzer.py
python evaluator.py
```


