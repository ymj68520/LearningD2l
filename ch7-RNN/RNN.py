# %%
# ==================== 导入必要的库 ====================
# collections: 提供专门的容器数据类型，如Counter用于计数
# re: 正则表达式模块，用于文本处理
# d2l: Dive into Deep Learning工具库，提供深度学习相关的辅助函数
import collections  # 导入collections模块，用于词频统计
import re  # 导入正则表达式模块，用于文本清理
import random   # random: Python标准库，用于随机数生成
import torch  # 导入PyTorch库，用于张量操作和深度学习
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
