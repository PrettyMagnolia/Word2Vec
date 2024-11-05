import json

import jieba
from tqdm.notebook import tqdm


def build_vocab(filepath):
    vocab = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            vocab.update(words)

    # 根据 key 的字典序排序，重新生成词典
    sorted_vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
    return sorted_vocab


def save_vocab_to_json(vocab, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


def read_vocab_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab


def preprocess_sougou():
    with open("/home/yifei/code/word2vec/data/stopwords.txt", 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)
    with open('/home/yifei/code/word2vec/data/news_sohusite_xml.dat', 'r', encoding='GB18030') as f:
        data = f.readlines()
    res = []
    for d in tqdm(data):
        if d.startswith('<content>'):
            # 去除 <content> 和 </content> 标签
            content = (
                d.replace("<content>", "")
                .replace("</content>", "")
                .replace("\n", "")
                .replace("\u3000", "")
                .replace("\ue40c", "")
                .replace("\n", "")
                .strip()
            )
            if len(content) <= 0:
                continue
            # 使用 jieba 分词
            words = jieba.cut(content, cut_all=False)
            words = list(words)

            # 过滤停用词
            filtered_words = [word for word in words if word not in stopwords and word.strip()]

            if len(content) > 0:
                # 将清理后的内容加入结果列表，重新组合成一句话
                res.append(" ".join(filtered_words))
    with open('./data/sougou.txt', 'w', encoding='utf-8') as f:
        for r in res:
            f.write(r + '\n')


if __name__ == '__main__':
    # 使用示例
    file_path = '/home/yifei/code/word2vec/data/sougou.txt'  # 输入文本文件路径
    output_filepath = 'vocab_sougou.json'  # 输出词表文件路径

    vocab = build_vocab(file_path)
    save_vocab_to_json(vocab, output_filepath)
    print(f"词表已保存至 {output_filepath}")
