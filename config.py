import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Word2Vec Training")

    # 数据相关参数
    parser.add_argument('--file_path', type=str, default='/home/yifei/code/word2vec/data/en.txt',
                        help="Path to input text file")
    parser.add_argument('--vocab_path', type=str, default='/home/yifei/code/word2vec/data/vocab_en.json',
                        help="Path to vocabulary JSON")

    # 模型和训练参数
    parser.add_argument('--window_size', type=int, default=2, help="Context window size")
    parser.add_argument('--embedding_dim', type=int, default=100, help="Dimension of word embeddings")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--num_epochs', type=int, default=8, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8192, help="Batch size for training")
    parser.add_argument('--type', type=str, choices=['cbow', 'skipgram'], default='cbow',
                        help="Type of model to train: cbow or skipgram")
    parser.add_argument('--negative_sampling_size', type=int, default=5,
                        help="Number of negative samples for negative sampling")

    # 日志和输出
    parser.add_argument('--log_dir', type=str, default='runs/experiment', help="TensorBoard log directory")

    return parser.parse_args()