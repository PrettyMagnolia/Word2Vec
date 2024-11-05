import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(CBOW, self).__init__()
        # 定义输入和输出的嵌入层
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_embedding = nn.Embedding(vocab_size, embed_dim)

        # 二分类损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, context, target, negatives):
        # 获取上下文词的嵌入表示并计算均值
        context_embeddings = self.input_embedding(context)
        context_mean = context_embeddings.mean(dim=1, keepdim=True)

        # 获取目标词和负采样词的嵌入表示
        target_embeddings = self.output_embedding(target)
        negative_embeddings = self.output_embedding(negatives)

        # 计算上下文嵌入与目标词的点积
        positive_dot_product = torch.sum(context_mean * target_embeddings, dim=2)

        # 计算上下文嵌入与负采样词的点积
        negative_dot_product = torch.bmm(negative_embeddings, context_mean.transpose(1, 2)).squeeze(2)

        # 拼接正例和负例的得分
        logits = torch.cat([positive_dot_product, negative_dot_product], dim=1)

        # 标签：正样本为1，负样本为0
        labels = torch.cat([torch.ones_like(positive_dot_product), torch.zeros_like(negative_dot_product)], dim=1)

        # 计算二分类损失
        loss = self.loss_fn(logits, labels.float())

        return loss


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram, self).__init__()

        # 初始化输入和输出嵌入层
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_embedding = nn.Embedding(vocab_size, embed_dim)

        # 二分类损失函数
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, center, context, negatives):
        # 中心词嵌入，扩展维度以便与上下文和负例做点积
        center_embedding = self.input_embedding(center).unsqueeze(1)  # [batch_size, 1, embed_dim]

        # 获取上下文词和负采样词的嵌入表示
        context_embedding = self.output_embedding(context)
        negative_embeddings = self.output_embedding(negatives)

        # 计算中心词与上下文词的点积
        positive_dot_product = torch.sum(center_embedding * context_embedding, dim=2)

        # 计算中心词与负采样词的点积
        negative_dot_product = torch.bmm(negative_embeddings, center_embedding.transpose(1, 2)).squeeze(2)

        # 拼接正例和负例的得分
        logits = torch.cat([positive_dot_product, negative_dot_product], dim=1)

        # 标签：正样本为1，负样本为0
        labels = torch.cat([torch.ones_like(positive_dot_product), torch.zeros_like(negative_dot_product)], dim=1)

        # 计算二分类损失
        loss = self.loss_fn(logits, labels.float())

        return loss