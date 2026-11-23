# pairwise-contact-shadow-reproject

## 领域

游戏、图形与多媒体

## 主题

屏幕空间接触阴影

## 说明

重投影拖尾、步进断裂与 compute 栅栏缺失。

初始状态下 `graphics/contact_shadow_reproject.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
