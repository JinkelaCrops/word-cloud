from utils.wordcloudpos import WordCloudPos
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import Random

plt.switch_backend('agg')


def gen_text_color(p_dict, times, rep='\u0000'):
    sss = lambda s: (s - 20) ** 2

    s = np.array([j["fsize"] for n, j in p_dict.items()])
    sm = np.min(sss(s))
    text = {}
    color = {}
    for t in range(times):
        for n, j in p_dict.items():
            nt = n + rep * t
            st = sss(j["fsize"]) if t == 0 else sm
            text[nt] = st
            color[nt] = j["color"]
    return text, color


def to_rgb(cc):
    r, g, b = int("0x%s" % cc[:2], 16), int("0x%s" % cc[2:4], 16), int("0x%s" % cc[4:], 16)
    return r, g, b


def maxmin_fig(mask):
    mi = np.min(mask, axis=(0, 1))
    mx = np.max(mask, axis=(0, 1))
    mm = (mask - mi) / (mx - mi) * 255
    return mm.astype("uint8")


def square_fig(mask):
    x, y, _ = mask.shape
    if x < y:
        mar = int((y - x) / 2)
        return cv2.copyMakeBorder(mask, mar, mar, 0, 0, cv2.BORDER_CONSTANT)
    else:
        mar = int((x - y) / 2)
        return cv2.copyMakeBorder(mask, 0, 0, mar, mar, cv2.BORDER_CONSTANT)


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
    # p_dict: {word index: {font size: , font color: }}
    for line in params:
        line_data = line.strip().split()
        p_dict[line_data[0]] = {"color": line_data[1][1:], "fsize": int(line_data[2])}

    # gen text, color for times, 20
    text_items, word_color_map = gen_text_color(p_dict, 20, rep="\u0000")
    print("text length:", len(text_items), list(text_items.items())[-1])

    # mask input
    mask_pic = np.array(Image.open(mask_path))[:, :, :3]
    # mask resize
    mask_less = cv2.resize(mask_pic, (mask_pic.shape[1] * 8, mask_pic.shape[0] * 8))  # resize mask
    mask_less = maxmin_fig(mask_less)
    mask_less = 255 - square_fig(255 - mask_less)
    # maskmore and maskless,
    # maskmore(mask_more) is the background, maskless(mask_less) is the wordcloud
    pad_size = 100
    mar = int(pad_size / 2)
    mask_more = cv2.resize(mask_less, (mask_less.shape[1] + pad_size, mask_less.shape[0] + pad_size))
    mask_less = cv2.dilate(mask_more, np.ones((mar, mar), np.uint8), iterations=1)

    wc = WordCloudPos(font_path=font_path,
                      prefer_horizontal=1,
                      background_color="white",
                      mask=mask_less,
                      max_font_size=300,  # 800 max word size, should be larger when mask is larger
                      random_state=80, margin=2,
                      max_words=500,  # more words
                      width=100, height=100  # useless when use mask
                      )

    # user defined position
    userdefpos = {"English": (2211, 3059), "Chinese": (1916, 3878)}

    # replacement for p_dict.index
    wc.set_word_replace("\u0000")

    # generate word cloud
    wc.generate_from_frequencies_positions(text_items, userdefpos)

    # recolor
    wc.recolor(color_func=colormap_color_func(word_color_map))

    # generate back(background) from maskmore(mask_more)
    # generate front(pure wordclouds)
    ndwc = wc.to_array()
    mmm = np.array([np.mean(ndwc, axis=2) == 255] * 3).transpose((1, 2, 0))
    back = mmm * (255 - mask_more)
    frnt = (~mmm) * ndwc
    # background grey
    back[np.where(back > np.max(back) - 10)] = np.max(back) - 10

    # plot
    plt.figure(figsize=(30, 30))  # larger, better quality
    plt.imshow(back + frnt + mask_more)
    plt.axis("off")
    plt.savefig("word_graph1.png")
    plt.close()
