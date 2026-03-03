# %% [markdown]
# # BERT 预训练数据集构建
# 本 Notebook 展示如何从 WikiText-2 构建 BERT 预训练所需的数据管道。
# 核心任务包含：
# - NSP（下一句预测）样本构造
# - MLM（遮蔽语言模型）样本构造
# - 定长 padding 与批处理输出

# %%
# 安装 d2l（《动手学深度学习》工具包）
# --no-deps: 不自动安装依赖，避免覆盖当前环境已有包
# --quiet: 安静模式，减少输出噪音
# %pip install d2l --no-deps --quiet

# %% [markdown]
# ## 1) 依赖导入与数据下载配置
# 先导入所需库，并注册 WikiText-2 下载地址。
# 这一步提供后续数据读取、随机采样、张量化处理的基础能力。

# %%
# 导入标准库：文件路径操作
import os
# 导入标准库：随机采样（用于 NSP/MLM 随机策略）
import random
# 导入 PyTorch 主库
import torch
# 导入目录删除等文件系统操作
import shutil
# 导入 zip 相关异常类型（用于坏压缩包容错）
import zipfile
# 导入网络下载工具（备用镜像下载）
import urllib.request
# 导入 d2l 的 torch 子模块（包含 tokenize、Vocab、download_extract 等工具）
from d2l import torch as d2l

# %% [markdown]
# ## 2) 读取与清洗 WikiText-2
# 这里把原始文本读入并转换成段落-句子结构。
# 后续 NSP 会基于“句子对”进行采样，因此这一步是样本生成的前提。

# %%
# 在 d2l 数据注册表中登记 WikiText-2 的下载地址与 sha1 校验值
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

# 读取并预处理 WikiText-2 训练语料
def _read_wiki(data_dir):
    # 拼接训练文件路径
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    # 读取全部行
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # 预处理流程：
    # 1) 去掉首尾空白
    # 2) 转小写
    # 3) 按 ' . ' 粗粒度切句（与 d2l 示例一致）
    # 4) 仅保留至少 2 句的段落（方便构造 NSP 正样本）
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]

    # 打乱段落顺序，减少训练时顺序偏差
    random.shuffle(paragraphs)
    return paragraphs

# %% [markdown]
# ## 3) NSP：下一句预测样本构造
# 接下来定义 NSP 相关函数：
# - 一部分样本使用真实下一句（正样本）
# - 一部分样本替换为随机句子（负样本）
# 这是 BERT 预训练中的句间关系学习目标。

# %%
# 生成一条 NSP（Next Sentence Prediction）训练样本
def _get_next_sentence(sentence, next_sentence, paragraphs):
    # 50% 概率保留真实下一句 -> 正样本（is_next=True）
    if random.random() < 0.5:
        is_next = True
    else:
        # 50% 概率替换为随机句子 -> 负样本（is_next=False）
        # paragraphs 是三层结构：段落列表 -> 句子列表 -> token 列表（后续会 token 化）
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False

    # 返回句子对与标签
    return sentence, next_sentence, is_next

# %%
# 从一个段落中构造 NSP 训练样本
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    # 保存本段落产生的所有样本
    nsp_data_from_paragraph = []

    # 遍历相邻句子对：paragraph[i] 与 paragraph[i+1]
    for i in range(len(paragraph) - 1):
        # 生成正/负样本句子对
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)

        # BERT 输入格式为：<cls> tokens_a <sep> tokens_b <sep>
        # 总长度超出 max_len 则跳过
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue

        # 构造带特殊符号的 token 序列与 segment 标记
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)

        # 保存：(tokens, segments, NSP 标签)
        nsp_data_from_paragraph.append((tokens, segments, is_next))

    return nsp_data_from_paragraph

# %% [markdown]
# ## 4) MLM：遮蔽语言模型样本构造
# 这一部分按 BERT 规则采样 MLM：
# - 15% token 作为预测目标
# - 80% 替换为 `<mask>`，10% 保持原词，10% 替换为随机词
# 用于学习词级上下文表示。

# %%
# 根据 BERT 规则替换 MLM 目标位置的 token
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # 复制一份输入 token，避免原地修改原序列
    mlm_input_tokens = [token for token in tokens]

    # 记录 (被预测位置, 该位置原始 token 标签)
    pred_positions_and_labels = []

    # 打乱候选位置，随机选择要预测的 token
    random.shuffle(candidate_pred_positions)

    for mlm_pred_position in candidate_pred_positions:
        # 达到目标预测数后停止
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break

        masked_token = None

        # 80% 概率替换为 <mask>
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 剩余 20% 中：10% 保持原 token，10% 随机 token
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.choice(vocab.idx_to_token)

        # 将输入序列该位置替换为采样得到的 token
        mlm_input_tokens[mlm_pred_position] = masked_token

        # 标签始终保存原始 token（监督目标）
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))

    # 返回替换后输入与监督标签
    return mlm_input_tokens, pred_positions_and_labels

# %%
# 从 token 序列构建 MLM 训练字段
def _get_mlm_data_from_tokens(tokens, vocab):
    # 记录可被预测的位置（排除特殊 token）
    candidate_pred_positions = []

    # tokens 是字符串列表
    for i, token in enumerate(tokens):
        # 特殊符号不参与 MLM 预测
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)

    # 按 BERT 规则：预测约 15% token，至少预测 1 个
    num_mlm_preds = max(1, round(len(tokens) * 0.15))

    # 执行随机替换并拿到监督标签
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)

    # 按位置排序，便于后续对齐
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])

    # 拆分出位置列表与标签 token 列表
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    # 返回：
    # 1) 替换后输入 token id
    # 2) 预测位置
    # 3) 预测标签 token id
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

# %% [markdown]
# ## 5) 定长对齐：Padding 与损失掩码
# BERT 训练要求 batch 内张量形状一致。
# 本部分会把 token 序列与 MLM 目标统一填充到固定长度，并构建 `mlm_weights` 过滤填充位。

# %%
# 将样本字段统一 padding 到固定长度，便于批训练
def _pad_bert_inputs(examples, max_len, vocab):
    # 每条样本 MLM 最多预测 token 数（约 15%）
    max_num_mlm_preds = round(max_len * 0.15)

    # 各字段容器（每项最终都是一个张量）
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []

    # examples 中每项结构：
    # (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next)
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:

        # token_ids padding 到 max_len
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))

        # segment ids 同步 padding（补 0）
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))

        # valid_len 记录真实 token 长度（不含 <pad>）
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))

        # MLM 预测位置 padding 到 max_num_mlm_preds
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))

        # MLM 权重：真实预测位置为 1，填充位置为 0（用于 loss 掩码）
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))

        # MLM 标签 id 同样 padding
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))

        # NSP 标签：True/False -> 1/0
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))

    # 返回所有字段列表（后续由 DataLoader 自动堆叠成 batch）
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

# %% [markdown]
# ## 6) 封装 Dataset
# 将 NSP + MLM 样本整合为 `torch.utils.data.Dataset`，
# 并在内部维护词表与所有训练字段，便于 DataLoader 直接迭代。

# %%
# 自定义 WikiText 数据集：同时提供 NSP + MLM 所需全部字段
class _WikiTextDataset(torch.utils.data.Dataset):
    # paragraphs: 原始段落（每个段落是句子字符串列表）
    # max_len: BERT 输入最大长度
    def __init__(self, paragraphs, max_len):
        # 先把每个句子做分词：
        # 输入 paragraphs[i] 是句子字符串列表；
        # 输出 paragraphs[i] 是 token 列表组成的句子列表
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]

        # 展平得到全部句子，用于构建词表
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]

        # 构建词表：低频词过滤（min_freq=5），并保留 BERT 特殊符号
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])

        # 1) 先构造 NSP 样本
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))

        # 2) 再为每个样本附加 MLM 字段
        # 结果结构：
        # (token_ids, pred_positions, mlm_label_ids, segments, is_next)
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]

        # 3) 对所有字段做定长 padding
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    # 按索引返回一条完整训练样本
    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    # 返回样本数量
    def __len__(self):
        return len(self.all_token_ids)

# %% [markdown]
# ## 7) 数据加载函数与容错下载
# 这一段封装 `load_data_wiki`，并处理网络或压缩包异常，
# 保证在常见环境下都能尽量成功拿到训练数据。

# %%
# 加载 WikiText-2 并返回 DataLoader + 词表
def load_data_wiki(batch_size, max_len):
    """加载 WikiText-2 数据集并构建 BERT 预训练输入"""

    # 获取建议的数据加载线程数
    num_workers = d2l.get_dataloader_workers()

    # 先尝试走 d2l 标准下载与解压流程
    try:
        data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    except zipfile.BadZipFile:
        # 若压缩包损坏：清理后重试
        zip_path = os.path.join('..', 'data', 'wikitext-2-v1.zip')
        extract_dir = os.path.join('..', 'data', 'wikitext-2')

        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir)

        try:
            data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
        except zipfile.BadZipFile:
            # 仍失败则走镜像兜底，仅下载训练文本
            data_dir = extract_dir
            os.makedirs(data_dir, exist_ok=True)
            train_file = os.path.join(data_dir, 'wiki.train.tokens')

            # 备用下载地址（按顺序尝试）
            mirror_urls = [
                'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt',
                'https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/wiki.train.raw'
            ]

            success = False
            for url in mirror_urls:
                try:
                    urllib.request.urlretrieve(url, train_file)
                    # 判断下载文件存在且非空
                    if os.path.exists(train_file) and os.path.getsize(train_file) > 0:
                        success = True
                        break
                except Exception:
                    continue

            # 所有兜底都失败时，抛出明确错误提示
            if not success:
                raise RuntimeError(
                    '无法获取 WikiText-2 数据。请检查网络，或手动将 wiki.train.tokens 放到 ../data/wikitext-2/ 下。'
                )

    # 读取段落文本
    paragraphs = _read_wiki(data_dir)

    # 构造自定义 Dataset
    train_set = _WikiTextDataset(paragraphs, max_len)

    # 构造 DataLoader（打乱顺序用于训练）
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=True, num_workers=num_workers)

    return train_iter, train_set.vocab

# %% [markdown]
# ## 8) 形状检查与词表规模查看
# 最后通过一次迭代打印所有训练张量形状，并查看词表大小，
# 确认数据管道输出符合 BERT 预训练输入预期。

# %%
# 设置批大小与最大序列长度
batch_size, max_len = 512, 64

# 加载训练迭代器与词表
train_iter, vocab = load_data_wiki(batch_size, max_len)

# 取一个 batch，检查每个字段的张量形状是否符合预期
for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    # 依次打印：
    # tokens_X:      (B, max_len)
    # segments_X:    (B, max_len)
    # valid_lens_x:  (B,)
    # pred_positions:(B, max_num_mlm_preds)
    # mlm_weights_X: (B, max_num_mlm_preds)
    # mlm_Y:         (B, max_num_mlm_preds)
    # nsp_y:         (B,)
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break  # 只看第一个 batch 即可

# %%
# 查看词表大小（token 总数）
len(vocab)


