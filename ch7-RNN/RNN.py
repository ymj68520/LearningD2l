# %%
# ==================== 导入必要的库 ====================
# collections: 提供专门的容器数据类型，如Counter用于计数
# re: 正则表达式模块，用于文本处理
# d2l: Dive into Deep Learning工具库，提供深度学习相关的辅助函数
import collections  # 导入collections模块，用于词频统计
import re  # 导入正则表达式模块，用于文本清理
import math  # 导入数学模块，用于计算等
import random   # random: Python标准库，用于随机数生成
import torch  # 导入PyTorch库，用于张量操作和深度学习
from torch import nn  # 从torch导入神经网络模块，提供构建神经网络的工具
from torch.nn import functional as F  # 导入torch.nn.functional模块，提供常用的函数，如激活函数、损失函数等
from d2l import torch as d2l  # 从d2l导入torch相关工具

# %%
# ==================== 下载和读取时间机器数据集 ====================
# 什么是时间机器数据集？
# - 《时间机器》小说全文
# - 用于自然语言处理任务的经典数据集
# - 包含大量英文文本，用于训练语言模型

# 将数据集添加到d2l的数据中心
# 'time_machine': 数据集名称
# d2l.DATA_URL + 'timemachine.txt': 下载URL
# '090b5e7e70c295757f55df93cb0a180b9691891a': 文件哈希值，用于验证完整性
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')  # 注册数据集到d2l数据中心

def read_time_machine():  # 定义读取时间机器数据集的函数
    """
    加载时间机器数据集到文本行列表
    
    处理步骤：
    1. 下载数据集文件
    2. 读取所有行
    3. 使用正则表达式清理文本：
       - [^A-Za-z]+: 匹配非字母字符
       - 替换为空格
       - 转换为小写
       - 去除首尾空格
    
    返回：
        清理后的文本行列表
    """
    with open(d2l.download('time_machine'), 'r') as f:  # 下载并打开文件
        lines = f.readlines()  # 读取所有行到列表
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]  # 清理每一行：替换非字母为空格，转小写，去除首尾空格

# 读取数据集
lines = read_time_machine()  # 调用函数读取数据集

# %%
# ==================== 文本分词函数 ====================
# 什么是分词（tokenization）？
# - 将连续的文本分割成有意义的单元（词元）
# - 词元可以是单词（word）或字符（char）
# - 这是NLP预处理的基础步骤

def tokenize(lines, token='word'):  # 定义分词函数，参数：文本行列表，分词类型（默认单词）
    """
    将文本行拆分为单词或字符词元
    
    参数：
        lines: 文本行列表
        token: 分词类型，'word'（单词）或'char'（字符）
    
    返回：
        分词后的词元列表，每个元素是一个词元列表
    
    示例：
        输入：["hello world", "how are you"]
        输出（word）：[["hello", "world"], ["how", "are", "you"]]
        输出（char）：[["h","e","l","l","o"," ","w","o","r","l","d"], ...]
    """
    if token == 'word':  # 如果分词类型是单词
        return [line.split() for line in lines]  # 按空格分割每一行，返回单词列表
    elif token == 'char':  # 如果分词类型是字符
        return [list(line) for line in lines]  # 将每一行转换为字符列表
    else:  # 否则
        print('错误：未知词元类型：' + token)  # 打印错误信息
        
# 对时间机器数据集进行分词（按单词）
tokens = tokenize(lines)  # 调用分词函数，默认按单词分词

# %%
# ==================== 词汇表（Vocab）类 ====================
# 什么是词汇表？
# - 将词元映射到数字索引的字典
# - 用于将文本转换为模型可以处理的数字序列
# - 通常按词频排序，最常见的词有较小的索引

class Vocab:  # 定义词汇表类
    """
    文本词表：词元到索引的映射
    
    主要功能：
    - 统计词频，按频率排序
    - 创建词元到索引的映射
    - 支持未知词元处理
    - 提供索引到词元的反向映射
    """
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):  # 初始化方法，参数：词元列表，最小频率，保留词元
        """
        初始化词汇表
        
        参数：
            tokens: 词元列表，可以是1D或2D列表
            min_freq: 最小词频阈值，低于此频率的词元将被过滤
            reserved_tokens: 保留词元列表，如特殊符号<pad>、<bos>等
        
        处理流程：
        1. 统计所有词元的频率
        2. 按频率降序排序
        3. 过滤低频词元
        4. 创建索引映射（未知词元索引为0）
        """
        if tokens is None:  # 如果tokens为None
            tokens = []  # 设置为空列表
        if reserved_tokens is None:  # 如果reserved_tokens为None
            reserved_tokens = []  # 设置为空列表
            
        # 按出现频率排序词元
        counter = count_corpus(tokens)  # 统计词频
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],  # 按频率降序排序
                                   reverse=True)
        
        # 未知词元的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens  # 设置未知词元索引，初始化唯一词元列表
        
        # 添加满足频率要求的词元
        uniq_tokens += [  # 扩展唯一词元列表
            token for token, freq in self._token_freqs  # 遍历词频对
            if freq >= min_freq and token not in uniq_tokens  # 如果频率足够且不在列表中
        ]
        
        # 创建双向映射
        self.index_to_token, self.token_to_idx = [], dict()  # 初始化索引到词元和词元到索引的映射
        for token in uniq_tokens:  # 遍历唯一词元
            self.index_to_token.append(token)  # 添加到索引到词元列表
            self.token_to_idx[token] = len(self.index_to_token) - 1  # 设置词元到索引映射

    def __len__(self):  # 定义长度方法
        """返回词汇表大小"""
        return len(self.index_to_token)  # 返回索引到词元列表的长度

    def __getitem__(self, tokens):  # 定义索引方法，支持词元到索引转换
        """
        将词元转换为索引
        
        参数：
            tokens: 单个词元或词元列表
        
        返回：
            对应的索引或索引列表
        """
        if not isinstance(tokens, (list, tuple)):  # 如果不是列表或元组（单个词元）
            return self.token_to_idx.get(tokens, self.unk)  # 返回索引，未知词元返回unk索引
        return [self.__getitem__(token) for token in tokens]  # 递归转换列表中的每个词元

    def to_tokens(self, indices):  # 定义反向转换方法
        """
        将索引转换为词元
        
        参数：
            indices: 单个索引或索引列表
        
        返回：
            对应的词元或词元列表
        """
        if not isinstance(indices, (list, tuple)):  # 如果不是列表或元组（单个索引）
            return self.index_to_token[indices]  # 返回对应的词元
        return [self.index_to_token[index] for index in indices]  # 返回索引列表对应的词元列表

def count_corpus(tokens):  #@save  # 定义词频统计函数
    """
    统计词元的频率
    
    参数：
        tokens: 词元列表，可以是1D或2D列表
    
    返回：
        词元频率的Counter对象
    
    处理逻辑：
    - 如果输入是2D列表（多行文本），先展平为一维
    - 使用collections.Counter统计每个词元的出现次数
    """
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):  # 如果tokens为空或第一个元素是列表（2D）
        tokens = [token for line in tokens for token in line]  # 展平为1D列表
    return collections.Counter(tokens)  # 返回词频统计结果

# %%
# ==================== 创建词汇表实例 ====================
# 使用分词后的词元创建词汇表
# 这将建立词元到数字索引的映射关系
vocab = Vocab(tokens)  # 使用分词结果创建词汇表实例

# %%
# ==================== 加载时光机器语料库 ====================
# 什么是语料库？
# - 经过预处理的文本数据集
# - 通常是词元索引的序列
# - 用于训练语言模型或进行其他NLP任务

def load_corpus_time_machine(max_tokens=-1):  # 定义加载语料库函数，参数：最大词元数
    """
    返回时光机器数据集的词元索引列表和词表
    
    参数：
        max_tokens: 最大词元数量，-1表示使用全部词元
    
    返回：
        corpus: 词元索引的列表（一维）
        vocab: 词汇表对象
    
    处理流程：
    1. 读取原始文本行
    2. 按字符分词（而不是单词）
    3. 创建词汇表
    4. 将所有词元转换为索引
    5. 可选：截取前max_tokens个词元
    """
    lines = read_time_machine()  # 读取文本行
    tokens = tokenize(lines, 'char')  # 按字符分词
    vocab = Vocab(tokens)  # 创建词汇表
    
    # 将文本转换为词元的索引表示
    corpus = [vocab[token] for line in tokens for token in line]  # 展平并转换为索引
    
    if max_tokens > 0:  # 如果指定了最大词元数
        corpus = corpus[:max_tokens]  # 截取前max_tokens个词元
    
    return corpus, vocab  # 返回语料库和词汇表

# 加载完整的语料库
corpus, vocab = load_corpus_time_machine()  # 调用函数加载语料库

# ==================== 分词和构建词汇表 ====================
# 分词：将文本拆分成词（token）
# d2l.tokenize()会将每行文本拆分成单词列表
# 返回：列表的列表，外层列表对应每一行，内层列表对应该行的单词
tokens = tokenize(read_time_machine())

# 将所有文本行的词拼接成一个长序列（语料库corpus）
# 为什么要拼接？
# - 文本行的划分是人为的，不一定对应句子或段落
# - 语言模型需要连续的文本序列
# - 列表推导式：[token for line in tokens for token in line]
#   遍历每一行，再遍历该行的每个词
corpus = [token for line in tokens for token in line]

# 构建词汇表（Vocabulary）
# 什么是词汇表？
# - 语料库中所有唯一词的集合
# - 为每个词分配一个唯一的索引（ID）
# - 统计每个词的出现频率
# 
# d2l.Vocab会：
# 1. 统计每个词的频率
# 2. 按频率降序排序
# 3. 为每个词分配索引
vocab = Vocab(corpus)

# 查看词频最高的前10个词
# token_freqs: 列表，每个元素是(词, 频率)的元组
# 预期：最常见的词通常是'the', 'a', 'and'等功能词
# vocab.token_freqs[:10]

# %%
# ==================== 可视化词频分布 ====================

# 提取所有词的频率（不需要词本身，只要频率）
# freqs = [freq for token, freq in vocab.token_freqs]

# %%
# ==================== 二元语法（Bigram） ====================
# 什么是n元语法（n-gram）？
# - 连续n个词的序列
# - Unigram（1-gram）：单个词，如'the'
# - Bigram（2-gram）：两个连续的词，如'the time'
# - Trigram（3-gram）：三个连续的词，如'the time machine'
# 
# 为什么需要n-gram？
# - 捕捉词与词之间的关系
# - 'New York'比'New'和'York'分开更有意义
# - 提高语言模型的表达能力

# 构造二元语法tokens
# zip(corpus[:-1], corpus[1:]): 将相邻的词配对
# 例如：corpus = ['the', 'time', 'machine', 'by']
#       pairs = [('the', 'time'), ('time', 'machine'), ('machine', 'by')]
# 
# [:-1]: 从第一个词到倒数第二个词
# [1:]: 从第二个词到最后一个词
# zip将它们配对
bigram_tokens = [' '.join(pair) for pair in zip(corpus[:-1], corpus[1:])]

# 为二元语法构建词汇表
bigram_vocab = Vocab(bigram_tokens)

# 查看最常见的10个二元语法及其频率
# 例如：'of the', 'in the', 'to the'等
# 返回格式：[(('of', 'the'), 频率), ...]
# [(tuple(token.split()), freq) for token, freq in bigram_vocab.token_freqs[:10]]

# %%
# ==================== 三元语法（Trigram） ====================

# 构造三元语法tokens
# 将三个连续的词组合在一起
# 
# zip的多参数用法：
# corpus[:-2]: 第1个词到倒数第3个词
# corpus[1:-1]: 第2个词到倒数第2个词
# corpus[2:]: 第3个词到最后一个词
# 
# 例如：corpus = ['the', 'time', 'machine', 'by', 'h']
#       triples = [('the', 'time', 'machine'), 
#                  ('time', 'machine', 'by'),
#                  ('machine', 'by', 'h')]
trigram_tokens = [' '.join(triple) for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]

# 为三元语法构建词汇表
trigram_vocab = Vocab(trigram_tokens)

# 查看最常见的10个三元语法
# 三元语法能捕捉更复杂的短语和表达
# 例如：'one of the', 'it was a'等
# trigram_vocab.token_freqs[:10]

# %%
# ==================== 对比不同n-gram的词频分布 ====================

# 提取bigram和trigram的频率
# bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
# trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]

# %%
# ==================== 随机采样的序列数据迭代器 ====================

def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样生成小批量子序列
    
    什么是随机采样？
    - 从语料库中随机选取多个起始位置
    - 从每个起始位置提取固定长度的序列
    - 不同批次间的序列不保证连续
    
    参数:
        corpus: 词索引列表（整个语料库）
        batch_size: 批次大小（每批包含多少个序列）
        num_steps: 每个序列的长度（时间步数）
    
    生成:
        X, Y: 输入和标签张量
        - X的形状: (batch_size, num_steps)
        - Y的形状: (batch_size, num_steps)
        - Y是X向后偏移1位的结果（预测下一个词）
    
    优点：
    - 简单直接
    - 每个epoch都能看到不同的序列组合
    
    缺点：
    - 批次间的序列不连续，丢失了跨批次的上下文信息
    """
    # 从随机偏移量开始对序列进行分区
    # 随机偏移0到num_steps-1位置，避免每次都从同一位置开始
    corpus = corpus[random.randint(0, num_steps - 1):]
    
    # 计算可以分成多少个长度为num_steps的子序列
    # 减1是因为需要为标签留出空间（Y=X向后偏移1）
    num_subseqs = (len(corpus) - 1) // num_steps
    
    # 生成所有子序列的起始索引
    # range(0, num_subseqs * num_steps, num_steps)
    # 例如：num_steps=5时，起始位置为[0, 5, 10, 15, ...]
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    
    # 随机打乱起始索引
    # 这样在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列
    # 不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        """
        返回从pos位置开始的长度为num_steps的序列
        
        参数:
            pos: 起始位置
        
        返回:
            长度为num_steps的子序列
        """
        return corpus[pos: pos + num_steps]

    # 计算可以产生多少个批次
    num_batches = num_subseqs // batch_size
    
    # 生成每个批次
    for i in range(0, batch_size * num_batches, batch_size):
        # 获取当前批次的起始索引
        # 每批包含batch_size个序列
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        
        # 构造输入X：从每个起始位置提取num_steps个词
        X = [data(j) for j in initial_indices_per_batch]
        
        # 构造标签Y：从每个起始位置+1提取num_steps个词
        # Y[i] = X[i]的下一个词，即X向右偏移1位
        Y = [data(j + 1) for j in initial_indices_per_batch]
        
        # 转换为张量并返回
        yield torch.tensor(X), torch.tensor(Y)

# %%
# ==================== 测试随机采样 ====================

# 创建一个简单的序列用于演示
# my_seq = [0, 1, 2, 3, ..., 34]
my_seq = list(range(35))

# %%
# ==================== 顺序分区的序列数据迭代器 ====================

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    使用顺序分区生成小批量子序列
    
    什么是顺序分区？
    - 将语料库分成batch_size个连续的部分
    - 每个部分按顺序生成序列
    - 同一批次内的序列在原文中的位置不同，但批次间保持连续
    
    参数:
        corpus: 词索引列表
        batch_size: 批次大小
        num_steps: 每个序列的长度
    
    生成:
        X, Y: 输入和标签张量
    
    优点：
    - 保持了序列的连续性
    - 可以维护跨批次的隐藏状态
    - 更适合训练RNN
    
    缺点：
    - 每个epoch看到的序列顺序相同
    
    示例（batch_size=2, num_steps=5）：
    原序列: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,...]
    分为2批:
    批次1: [0,1,2,3,4]    批次2: [5,6,7,8,9]    ...
           [8,9,10,11,12]        [13,14,15,16,17] ...
    """
    # 从随机偏移量开始划分序列
    # 添加一点随机性，避免每次epoch从完全相同的位置开始
    offset = random.randint(0, num_steps)
    
    # 计算可用的tokens数量
    # 需要能被batch_size整除，以便均匀分配
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    
    # 提取输入序列X和标签序列Y
    # Y相对于X向后偏移1位
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    
    # 重塑为(batch_size, -1)
    # 将长序列分成batch_size行
    # 每行是一个独立的序列流
    # 
    # 例如：num_tokens=24, batch_size=2
    # 原来: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    # 重塑后:
    # [[0,1,2,3,4,5,6,7,8,9,10,11],
    #  [12,13,14,15,16,17,18,19,20,21,22,23]]
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    
    # 计算可以生成多少个批次
    num_batches = Xs.shape[1] // num_steps
    
    # 按顺序生成每个批次
    for i in range(0, num_steps * num_batches, num_steps):
        # 从每一行中提取num_steps列
        # 这样保证了每个批次内的序列在时间上是连续的
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


# %%
# ==================== 序列数据加载器类 ====================

class SeqDataLoader:
    """
    加载序列数据的迭代器类
    
    这个类封装了序列数据加载的逻辑：
    1. 选择采样策略（随机或顺序）
    2. 加载和处理语料库
    3. 提供统一的迭代接口
    
    使用方法：
        loader = SeqDataLoader(batch_size=32, num_steps=35, 
                               use_random_iter=False, max_tokens=10000)
        for X, Y in loader:
            # 训练模型
            ...
    """
    
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """
        初始化数据加载器
        
        参数:
            batch_size: 批次大小
            num_steps: 序列长度（时间步数）
            use_random_iter: 是否使用随机采样
                - True: 随机采样（批次间不连续）
                - False: 顺序分区（批次间连续）
            max_tokens: 最大token数量（限制语料库大小）
        """
        # 根据use_random_iter选择迭代函数
        if use_random_iter:
            # 使用随机采样
            self.data_iter_fn = seq_data_iter_random
        else:
            # 使用顺序分区
            self.data_iter_fn = seq_data_iter_sequential
        
        # 加载时间机器语料库和词汇表
        # load_corpus_time_machine会：
        # 1. 读取并清理文本
        # 2. 分词
        # 3. 构建词汇表
        # 4. 将文本转换为词索引序列
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        
        # 保存参数
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        """
        使对象可迭代
        
        返回数据迭代器，用于for循环
        """
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

# %%
# ==================== 加载时间机器数据集的便捷函数 ====================

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """
    返回时光机器数据集的迭代器和词汇表
    
    这是一个便捷函数，简化了数据加载过程
    
    参数:
        batch_size: 批次大小
        num_steps: 序列长度
        use_random_iter: 是否使用随机采样，默认False（使用顺序分区）
        max_tokens: 最大token数量，默认10000
    
    返回:
        data_iter: 数据迭代器，可用于for循环
        vocab: 词汇表对象，包含：
            - token到索引的映射
            - 索引到token的映射
            - 词频统计
    
    使用示例:
        train_iter, vocab = load_data_time_machine(32, 35)
        for X, Y in train_iter:
            # X: (32, 35) - 批次大小32，序列长度35
            # Y: (32, 35) - 对应的标签
            ...
    """
    # 创建数据迭代器
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    
    # 返回迭代器和词汇表
    # data_iter.vocab: 访问SeqDataLoader内部的词汇表
    return data_iter, data_iter.vocab

# ==================================================
# 初始化RNN隐藏状态
# ==================================================
def init_rnn_state(batch_size, num_hiddens, device):
    """
    初始化RNN的隐藏状态
    
    在每个批次开始时，需要初始化隐藏状态为零向量
    
    参数:
        batch_size: 批量大小
        num_hiddens: 隐藏层单元数
        device: 计算设备
    
    返回:
        包含一个张量的元组，形状为(batch_size, num_hiddens)，初始值全为0
    """
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# ==================================================
# RNN前向传播函数
# ==================================================
def rnn(inputs, state, params):
    """
    RNN的前向传播计算
    
    核心公式: H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)
             Y_t = H_t @ W_hq + b_q
    
    参数:
        inputs: 输入序列，形状为(时间步数, 批量大小, 词表大小)，已经过one-hot编码
        state: 隐藏状态，包含一个张量H
        params: 模型参数 [W_xh, W_hh, b_h, W_hq, b_q]
    
    返回:
        outputs: 所有时间步的输出，形状为(时间步数*批量大小, 词表大小)
        (H,): 最后一个时间步的隐藏状态
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 解包隐藏状态
    outputs = []  # 存储每个时间步的输出
    
    # 之前转置过了，所以时序维度在第一个维度上，可以直接迭代读取每个时间步的输入
    # X的形状是（时间步数，批量大小，词表大小）
    for X in inputs:
        # 更新隐藏状态：H_t = tanh(X_t * W_xh + H_{t-1} * W_hh + b_h)
        # 激活函数使用tanh，将值压缩到(-1, 1)范围内
        H = torch.tanh(X @ W_xh + H @ W_hh + b_h)
        # 计算输出：Y_t = H_t * W_hq + b_q
        # 输出层不使用激活函数，因为后面会使用交叉熵损失函数，它会将softmax计算包含在内
        Y = H @ W_hq + b_q
        outputs.append(Y)
    
    # 将所有时间步的输出拼接成一个张量，形状为(时间步数*批量大小, 词表大小)
    return torch.cat(outputs, dim=0), (H, )

# ==================================================
# 从零开始实现的RNN模型类
# ==================================================
class RNNModelScratch:
    """
    从零开始实现的循环神经网络模型
    
    该类封装了RNN的参数初始化、状态初始化和前向传播功能
    """
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        """
        初始化RNN模型
        
        参数:
            vocab_size: 词汇表大小
            num_hiddens: 隐藏层单元数
            device: 计算设备
            get_params: 参数初始化函数
            init_state: 状态初始化函数
            forward_fn: 前向传播函数
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)  # 初始化所有参数
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """
        模型的调用接口
        
        参数:
            X: 输入数据，形状为(批量大小, 时间步数)，包含字符索引
            state: 隐藏状态
        
        返回:
            输出和新的隐藏状态
        """
        # 将输入索引转换为one-hot编码，并转置使时间步在第一维
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """初始化隐藏状态"""
        return self.init_state(batch_size, self.num_hiddens, device)
    
# ==================================================
# 预测函数：根据前缀生成后续文本
# ==================================================
def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    基于给定前缀预测后续字符
    
    工作流程：
    1. 使用前缀字符预热模型（更新隐藏状态）
    2. 自回归生成后续字符（每次用上一步的输出作为下一步的输入）
    
    参数:
        prefix: 前缀字符串，用于初始化模型状态
        num_preds: 要预测的字符数量
        net: RNN模型
        vocab: 词汇表
        device: 计算设备
    
    返回:
        生成的完整文本（前缀 + 预测的字符）
    """
    state = net.begin_state(batch_size=1, device=device)  # 初始化状态
    outputs = [vocab[prefix[0]]]  # 输出列表，先放入前缀的第一个字符的索引
    # 定义获取输入的lambda函数：取outputs的最后一个元素作为输入
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    # 使用前缀的剩余字符预热模型，更新隐藏状态
    for y in prefix[1:]:
        _, state = net(get_input(), state)  # 只关心状态更新，不保存输出
        outputs.append(vocab[y])  # 将前缀字符的索引加入输出列表
    
    # 开始自回归预测：生成num_preds个新字符
    for _ in range(num_preds):
        y, state = net(get_input(), state)  # 获取预测输出
        # argmax找到概率最大的字符索引
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    
    # 将索引列表转换回字符串
    return ''.join([vocab.index_to_token[i] for i in outputs])

# ==================================================
# 梯度裁剪：防止梯度爆炸
# ==================================================
def grad_clipping(net, theta):
    """
    裁剪梯度，防止梯度爆炸
    
    在RNN训练中，梯度可能会随时间步累积而爆炸性增长
    梯度裁剪通过限制梯度的L2范数来缓解这个问题
    
    参数:
        net: 神经网络模型
        theta: 梯度裁剪的阈值
    
    工作原理:
        1. 计算所有参数梯度的L2范数: norm = sqrt(sum(grad^2))
        2. 如果 norm > theta，则将所有梯度缩放: grad = grad * (theta / norm)
    """
    # 获取所有需要梯度的参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    
    # 计算所有参数梯度的L2范数
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    
    # 如果范数超过阈值，按比例缩放所有梯度
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            
# ==================================================
# 训练一个epoch：遍历整个数据集一次
# ==================================================
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """
    训练模型一个epoch
    
    参数:
        net: RNN模型
        train_iter: 训练数据迭代器
        loss: 损失函数
        updater: 优化器或自定义更新函数
        device: 计算设备
        use_random_iter: 是否使用随机采样（True）还是顺序分区（False）
    
    返回:
        perplexity: 困惑度 = exp(平均损失)，衡量模型预测的不确定性
        speed: 处理速度（词元/秒）
    """
    state, timer = None, d2l.Timer()  # 初始化状态和计时器
    metric = d2l.Accumulator(2)  # 累加器：[训练损失总和, 词元数量]
    
    for X, Y in train_iter:  # 遍历每个小批量
        if state is None or use_random_iter:
            # 在使用随机抽样时，每个小批量的序列是独立的，需要重新初始化状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 使用顺序分区时，相邻批次的序列是连续的
            # 需要分离（detach）隐藏状态，截断梯度的反向传播
            # 这样可以避免梯度在过长的序列上累积，同时保留状态信息用于预测
            if isinstance(state, tuple):
                # LSTM 返回元组 (h, c)
                state = tuple(s.detach() for s in state)
            else:
                # GRU 和 RNN 返回张量
                state = state.detach()
        
        # 准备标签：将Y转置并展平成一维向量
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        y_hat, state = net(X, state)
        # 计算交叉熵损失的平均值
        l = loss(y_hat, y.long()).mean()
        
        # 反向传播和参数更新
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch优化器
            updater.zero_grad()  # 梯度清零
            l.backward()  # 反向传播计算梯度
            grad_clipping(net, 1)  # 裁剪梯度，阈值为1
            updater.step()  # 更新参数
        else:
            # 使用自定义的SGD更新函数
            l.backward()
            grad_clipping(net, 1)
            updater(X.shape[0])  # 传入批量大小进行更新
        
        # 累积损失和词元数
        metric.add(l * y.numel(), y.numel())
    
    # 返回困惑度和处理速度
    # 困惑度 = exp(平均损失)，越低表示模型越好
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

# ==================================================
# 完整的RNN训练函数
# ==================================================
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """
    训练RNN模型的完整流程
    
    参数:
        net: RNN模型
        train_iter: 训练数据迭代器
        vocab: 词汇表
        lr: 学习率
        num_epochs: 训练轮数
        device: 计算设备
        use_random_iter: 是否使用随机采样（默认False，使用顺序分区）
    
    功能:
        1. 训练指定轮数
        2. 每10轮打印一次预测结果
        3. 实时绘制困惑度曲线
        4. 训练结束后展示最终预测结果
    """
    loss = nn.CrossEntropyLoss()  # 交叉熵损失函数（内置softmax）
    # 创建动画绘图器，用于实时显示训练过程中的困惑度变化
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    
    # ===== 初始化优化器 =====
    if isinstance(net, nn.Module):
        # 对于nn.Module模型，使用PyTorch的SGD优化器
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        # 对于自定义模型，使用d2l提供的SGD函数
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    
    # 定义预测函数：固定生成50个字符
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    
    # ===== 训练和预测 =====
    for epoch in range(num_epochs):
        # 训练一个epoch
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        
        # 每10轮打印一次当前的预测结果
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])  # 在图表中添加当前困惑度
    
    # ===== 训练完成，输出最终结果 =====
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))  # 预测 "time traveller" 后的文本
    print(predict('traveller'))  # 预测 "traveller" 后的文本
    
# ==================================================
# 完整的RNN模型类（封装RNN层和输出层）
# ==================================================
class RNNModel(nn.Module):
    """
    循环神经网络模型
    
    架构:
        输入 -> One-hot编码 -> RNN层 -> 全连接层 -> 输出
    
    该类将RNN层和输出层组合成一个完整的字符级语言模型
    """
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        """
        初始化RNN模型
        
        参数:
            rnn_layer: PyTorch的RNN层（nn.RNN/nn.LSTM/nn.GRU）
            vocab_size: 词汇表大小
        """
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size  # 隐藏层维度
        
        # 判断RNN是否为双向
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            # 单向RNN：隐藏层到输出层的线性变换
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            # 双向RNN：隐藏层维度翻倍（正向+反向）
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        """
        前向传播
        
        参数:
            inputs: 输入序列，形状为(batch_size, num_steps)，包含字符索引
            state: 隐藏状态
        
        返回:
            output: 输出logits，形状为(num_steps*batch_size, vocab_size)
            state: 更新后的隐藏状态
        """
        # 将输入索引转换为one-hot编码
        # inputs.T: (num_steps, batch_size)
        # X: (num_steps, batch_size, vocab_size)
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        
        # RNN前向传播
        # Y: (num_steps, batch_size, num_hiddens) - 每个时间步的隐藏状态输出
        Y, state = self.rnn(X, state)
        
        # 全连接层处理
        # 首先将Y的形状改为(时间步数*批量大小, 隐藏单元数)
        # 这样可以批量处理所有时间步的输出
        # 输出形状是(时间步数*批量大小, 词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """
        初始化隐藏状态
        
        参数:
            device: 计算设备
            batch_size: 批量大小
        
        返回:
            初始隐藏状态（形状和类型取决于RNN类型）
        """
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU和nn.RNN以张量作为隐藏状态
            # 形状: (num_directions * num_layers, batch_size, num_hiddens)
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM以元组作为隐藏状态（包括隐藏状态h和记忆细胞c）
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))