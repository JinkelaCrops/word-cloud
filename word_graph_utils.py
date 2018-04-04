from wordcloud import WordCloud
import cv2
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
font_path = "./fonts/Griffy-Regular.ttf"
mask_path = "./backgrounds/cat.png"


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

mask_pic = np.array(Image.open(mask_path))

mask_pic_2 = cv2.resize(mask_pic, (mask_pic.shape[1] * 10, mask_pic.shape[0] * 10))

wc = WordCloud(font_path=font_path,
               prefer_horizontal=1,
               background_color="white",
               mask=mask_pic_2,
               max_font_size=300, random_state=80, margin=0,
               max_words=2000,
               width=2000, height=2000
               )
# generate word cloud
wc.generate_from_frequencies(aaa)

plt.figure(figsize=(10, 10))
plt.imshow(wc)
plt.axis("off")
plt.savefig("word_graph4.png")
plt.show()
