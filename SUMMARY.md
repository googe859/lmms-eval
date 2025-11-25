# Qwen3-VL Token Pruning Summary

## 目标
在 `lmms-eval` 中运行 Qwen3-VL 的 MMMU 评测时，按照 `token_keep_ratio` 裁剪视觉 token，比较不同保留比例的效果。

## Hook 方案问题
- 语言模型 `forward` 时未必会在 kwargs 中提供 `deepstack_visual_embeds`。
- 即使传递了，旧实现中 `deepstack_visual_embeds` 是一次性迭代器，被消耗后无法用于实际裁剪。
- 缓存机制可能导致不重新运行模型，指标固定不变。

## 计划策略
1. 在视觉模型最高层输出根据范数选出保留索引。
2. 将同一批索引应用到所有 DeepStack 层的视觉 embedding，让语言模型只接收被保留的视觉 token。
3. 参考 `run_infer2.2.py` 的流程（先捕获视觉特征、再裁剪），在 `lmms-eval` 中完全实现。
