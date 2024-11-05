import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from dataset import CBOWDataset, SkipGramDataset
from model import CBOW, SkipGram
from utils import read_vocab_from_json


def get_log_dir(base_dir, params):
    file_name = os.path.basename(params.file_path).split(".")[0]
    log_dir = f"{base_dir}/lr{params.lr}_ws{params.window_size}_bs{params.batch_size}_ns{params.negative_sampling_size}_{file_name}"

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


class Trainer:
    def __init__(self, model, dataloader, loss_function, optimizer, config, vocab):
        self.model = model
        self.dataloader = dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.config = config
        self.vocab = vocab

    def cbow_train(self):
        """
        CBOW 模型训练过程：从上下文词预测中心词。
        """
        self.model.to(self.device)
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch in tqdm(self.dataloader):
                context = batch['context'].to(self.device)  # 输入为上下文词
                target = batch['target'].to(self.device)  # 目标为中心词
                negatives = batch['negatives'].to(self.device)  # 负采样的词

                # 前向传播
                self.model.zero_grad()
                loss = self.model(context, target, negatives)
                total_loss += loss.item()

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

            # 记录平均损失
            avg_loss = total_loss / len(self.dataloader)
            self.writer.add_scalar('CBOW Training Loss', avg_loss, epoch + 1)
            print(f"CBOW - Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss}")

        # 关闭 TensorBoard 记录器
        self.writer.close()

        # 保存词向量
        self.save_embeddings(self.vocab)

    def skip_gram_train(self):
        """
        Skip-Gram 模型训练过程：从中心词预测上下文词。
        """
        self.model.to(self.device)
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch in tqdm(self.dataloader):
                center = batch['center'].to(self.device)  # 输入为中心词
                context = batch['context'].to(self.device)  # 目标为上下文词
                negatives = batch['negatives'].to(self.device)  # 负采样的词

                # 前向传播
                self.model.zero_grad()
                loss = self.model(center, context, negatives)
                total_loss += loss.item()

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

            # 记录平均损失
            avg_loss = total_loss / len(self.dataloader)
            self.writer.add_scalar('Skip-Gram Training Loss', avg_loss, epoch + 1)
            print(f"Skip-Gram - Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss}")

        # 关闭 TensorBoard 记录器
        self.writer.close()

        # 保存词向量
        self.save_embeddings(self.vocab)

    def save_embeddings(self, vocab):
        word_embeddings = self.model.input_embedding.weight.data.cpu().numpy()
        word_to_vector = {word: word_embeddings[idx] for word, idx in vocab.items()}
        # Save word embeddings and model
        save_path = os.path.join(self.config.log_dir, "word_vectors.npy")
        np.save(save_path, word_to_vector)
        print(f"Embeddings saved to {save_path}")


if __name__ == "__main__":
    # 读取配置
    config = get_config()
    log_dir = get_log_dir(config.log_dir, config)
    config.log_dir = log_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 加载词汇表
    vocab = read_vocab_from_json(config.vocab_path)

    # 根据类型选择数据集和模型
    if config.type == 'cbow':
        dataset = CBOWDataset(file_path=config.file_path, window_size=config.window_size, vocab=vocab, negative_sampling_size=config.negative_sampling_size)
        model = CBOW(len(vocab), config.embedding_dim)
        train_function = Trainer(model, DataLoader(dataset, batch_size=config.batch_size, shuffle=True),
                                 nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=config.lr), config, vocab).cbow_train
    elif config.type == 'skipgram':
        dataset = SkipGramDataset(file_path=config.file_path, window_size=config.window_size, vocab=vocab, negative_sampling_size=config.negative_sampling_size)
        model = SkipGram(len(vocab), config.embedding_dim)
        train_function = Trainer(model, DataLoader(dataset, batch_size=config.batch_size, shuffle=True),
                                 nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=config.lr), config, vocab).skip_gram_train
    else:
        raise ValueError(f"Invalid model type '{config.type}'. Please choose either 'cbow' or 'skipgram'.")

    # 将模型移动到计算设备
    model.to(device)

    # 开始训练
    train_function()