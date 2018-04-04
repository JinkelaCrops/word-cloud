from wordcloud import WordCloud
import cv2
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
font_path = "./fonts/Griffy-Regular.ttf"
mask_path = "./backgrounds/cat.png"

from random import Random


class colormap_color_func(object):

    def __init__(self):
        import matplotlib.pyplot as plt
        self.colormap = plt.cm.get_cmap("viridis")

    def __call__(self, word, font_size, position, orientation,
                 random_state=None, **kwargs):
        if random_state is None:
            random_state = Random()
        if word == "English":
            cc = "87CEEB"
            r, g, b = int("0x%s" % cc[:2], 16), int("0x%s" % cc[2:4], 16), int("0x%s" % cc[4:], 16)
        else:
            r, g, b, _ = np.maximum(0, 255 * np.array(self.colormap(
                random_state.uniform(0, 1))))
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)


def word_freq(k, ks):
    return (ks - k) ** 0


with open("./text/lans_en.txt", "r", encoding="utf8") as f:
    text = f.readlines()
text = text * 100
text = [[x.strip(), word_freq(k, len(text))] for k, x in enumerate(text)]


class aa:
    def __init__(self, text):
        self.text = text

    def items(self):
        for a, b in self.text:
            yield a, b


aaa = aa(text)

mask_pic = np.array(Image.open(mask_path))[:, :, :3]

mask_pic_2 = cv2.resize(mask_pic, (mask_pic.shape[1] * 8, mask_pic.shape[0] * 8))  # resize mask

wc = WordCloud(font_path=font_path,
               prefer_horizontal=1,
               background_color="white",
               mask=mask_pic_2,
               max_font_size=800,  # max word size, should be larger when mask is larger
               random_state=80, margin=0,
               max_words=500,  # more words
               width=100, height=100  # useless when use mask
               )
# generate word cloud
wc.generate_from_frequencies(aaa)

wc.recolor(color_func=colormap_color_func())

plt.figure(figsize=(30, 30))  # larger, better quality
plt.imshow(wc)
plt.axis("off")
plt.savefig("word_graph3.png")
plt.show()
