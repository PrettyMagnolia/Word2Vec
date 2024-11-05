from torch.utils.data import Dataset
import torch
import random


class CBOWDataset(Dataset):
    def __init__(self, file_path, window_size, vocab, negative_sampling_size):
        """
        初始化CBOW数据集。

        参数：
        file_path (str): 语料库文件路径，每行一句话。
        window_size (int): 上下文窗口大小，即目标词左右各考虑的词数。
        vocab (dict): 词汇表，键为词，值为索引。
        negative_sampling_size (int): 每个目标词的负采样数。
        """
        super(CBOWDataset, self).__init__()
        self.window_size = window_size
        self.vocab = vocab
        self.negative_sampling_size = negative_sampling_size
        self.samples = []
        self.vocab_indices = list(vocab.values())  # 用于负采样

        # 加载文件并构建上下文-目标词对
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()  # 分词，假设空格分隔
                indices = [vocab[token] for token in tokens if token in vocab]

                for i in range(window_size, len(indices) - window_size):
                    context = indices[i - window_size:i] + indices[i + 1:i + window_size + 1]
                    target = indices[i]
                    self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]

        # 负采样
        negatives = random.sample(self.vocab_indices, self.negative_sampling_size)

        # 返回包含上下文、目标词和负样本的字典
        return {
            'context': torch.tensor(context, dtype=torch.long),
            'target': torch.tensor([target], dtype=torch.long),
            'negatives': torch.tensor(negatives, dtype=torch.long)
        }


class SkipGramDataset(Dataset):
    def __init__(self, file_path, window_size, vocab, negative_sampling_size):
        """
        初始化 Skip-Gram 数据集。

        参数：
        file_path (str): 语料库文件路径，每行一句话。
        window_size (int): 上下文窗口大小，即目标词左右各考虑的词数。
        vocab (dict): 词汇表，键为词，值为索引。
        negative_sampling_size (int): 每个目标词的负采样数。
        """
        super(SkipGramDataset, self).__init__()
        self.window_size = window_size
        self.vocab = vocab
        self.negative_sampling_size = negative_sampling_size
        self.samples = []
        self.vocab_indices = list(vocab.values())  # 用于负采样

        # 加载文件并构建中心词-上下文词对
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()  # 假设空格分隔词
                indices = [vocab[token] for token in tokens if token in vocab]

                # 生成中心词-上下文词对
                for i, center in enumerate(indices):
                    start = max(0, i - window_size)
                    end = min(len(indices), i + window_size + 1)
                    for j in range(start, end):
                        if i != j:
                            context = indices[j]
                            self.samples.append((center, context))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center, context = self.samples[idx]

        # 负采样
        negatives = random.sample(self.vocab_indices, self.negative_sampling_size)

        return {
            'center': torch.tensor(center, dtype=torch.long),
            'context': torch.tensor([context], dtype=torch.long),
            'negatives': torch.tensor(negatives, dtype=torch.long)
        }
