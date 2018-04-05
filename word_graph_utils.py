from wordcloud import WordCloud
import cv2
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import Random

plt.switch_backend('agg')


def gen_text(p_dict, times):
    text = []
    n = np.array(list(p_dict.keys()))
    s = np.array([j["fsize"] for n, j in p_dict.items()])
    s = (s - 20) ** 2
    # ratio = np.min(s) / np.max(s)
    for t in range(times):
        text += list(zip(n, s if t == 0 else (s > 0) * np.min(s)))
    return text


def gen_color(p_dict):
    color = {}
    for n, j in p_dict.items():
        color[n] = j["color"]
    return color


def to_rgb(cc):
    r, g, b = int("0x%s" % cc[:2], 16), int("0x%s" % cc[2:4], 16), int("0x%s" % cc[4:], 16)
    return r, g, b


def maxmin_fig(mask):
    mi = np.min(mask, axis=(0, 1))
    mx = np.max(mask, axis=(0, 1))
    mm = (mask - mi) / (mx - mi) * 255
    return mm.astype("uint8")


class colormap_color_func(object):

    def __init__(self, word_color_map):
        self.word_color_map = word_color_map
        import matplotlib.pyplot as plt
        self.colormap = plt.cm.get_cmap("viridis")

    def __call__(self, word, font_size, position, orientation,
                 random_state=None, **kwargs):

        if word in self.word_color_map.keys():
            r, g, b = to_rgb(self.word_color_map[word])
            return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)
        else:
            if random_state is None:
                random_state = Random()
            r, g, b, _ = np.maximum(0, 255 * np.array(self.colormap(
                random_state.uniform(0, 1))))
            return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)


class gen_items:
    def __init__(self, text):
        self.text = text

    def items(self):
        for a, b in self.text:
            yield a, b


if __name__ == '__main__':
    font_path = "./fonts/Griffy-Regular.ttf"
    mask_path = "./backgrounds/cat.png"
    with open("./text/YiCAT词云参数.txt", "r", encoding="utf8") as f:
        params = f.readlines()
    p_dict = {}
    for line in params:
        line_data = line.strip().split()
        p_dict[line_data[0]] = {"color": line_data[1][1:], "fsize": int(line_data[2])}

    text = gen_text(p_dict, 3)  # text times
    print("text length:", len(text), text[-1])
    text_items = gen_items(text)

    mask_pic = np.array(Image.open(mask_path))[:, :, :3]
    mask_pic_2 = cv2.resize(mask_pic, (mask_pic.shape[1] * 8, mask_pic.shape[0] * 8))  # resize mask
    mask_pic_2 = maxmin_fig(mask_pic_2)

    wc = WordCloud(font_path=font_path,
                   prefer_horizontal=1,
                   background_color="white",
                   mask=mask_pic_2,
                   max_font_size=300,  # 800 max word size, should be larger when mask is larger
                   random_state=80, margin=0,
                   max_words=500,  # more words
                   width=100, height=100  # useless when use mask
                   )
    # generate word cloud
    wc.generate_from_frequencies(text_items)

    word_color_map = gen_color(p_dict)
    wc.recolor(color_func=colormap_color_func(word_color_map))

    ndwc = wc.to_array()
    mmm = np.array([np.mean(ndwc, axis=2) == 255] * 3).transpose((1, 2, 0))
    back = mmm * (255 - mask_pic_2)
    frnt = (~mmm) * ndwc

    back[np.where(back > np.max(back) - 15)] = np.max(back) - 15
    plt.figure(figsize=(30, 30))  # larger, better quality
    plt.imshow(back + frnt + mask_pic_2)
    plt.axis("off")
    plt.savefig("word_graph_back.png")
    plt.close()
