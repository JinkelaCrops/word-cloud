from __future__ import division
from wordcloud import WordCloud
from wordcloud.wordcloud import IntegralOccupancyMap
import warnings
import re
from random import Random

import numpy as np
from operator import itemgetter

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class WordCloudPos(WordCloud):

    def set_word_replace(self, replacement):
        self.replacement = replacement
        return 0

    def generate_from_frequencies_positions(self, frequencies, userdefpos, max_font_size=None):
        """Create a word_cloud from words and frequencies.

        Parameters
        ----------
        frequencies : dict from string to float
            A contains words and associated frequency.

        max_font_size : int
            Use this font-size instead of self.max_font_size

        Returns
        -------
        self
        :param positions: dict

        """
        # make sure frequencies are sorted and normalized
        frequencies = sorted(frequencies.items(), key=itemgetter(1), reverse=True)
        if len(frequencies) <= 0:
            raise ValueError("We need at least 1 word to plot a word cloud, "
                             "got %d." % len(frequencies))
        frequencies = frequencies[:self.max_words]

        # largest entry will be 1
        max_frequency = float(frequencies[0][1])

        frequencies = [(word, freq / max_frequency)
                       for word, freq in frequencies]

        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = Random()

        if self.mask is not None:
            mask = self.mask
            width = mask.shape[1]
            height = mask.shape[0]
            if mask.dtype.kind == 'f':
                warnings.warn("mask image should be unsigned byte between 0"
                              " and 255. Got a float array")
            if mask.ndim == 2:
                boolean_mask = mask == 255
            elif mask.ndim == 3:
                # if all channels are white, mask out
                boolean_mask = np.all(mask[:, :, :3] == 255, axis=-1)
            else:
                raise ValueError("Got mask of invalid shape: %s"
                                 % str(mask.shape))
        else:
            boolean_mask = None
            height, width = self.height, self.width
        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        # create image
        img_grey = Image.new("L", (width, height))
        draw = ImageDraw.Draw(img_grey)
        img_array = np.asarray(img_grey)
        font_sizes, positions, orientations, colors = [], [], [], []

        last_freq = 1.

        if max_font_size is None:
            # if not provided use default font_size
            max_font_size = self.max_font_size

        if max_font_size is None:
            # figure out a good font size by trying to draw with
            # just the first two words
            if len(frequencies) == 1:
                # we only have one word. We make it big!
                font_size = self.height
            else:
                self.generate_from_frequencies_positions(dict(frequencies[:2]), userdefpos,
                                                         max_font_size=self.height)
                # find font sizes
                sizes = [x[1] for x in self.layout_]
                try:
                    font_size = int(2 * sizes[0] * sizes[1]
                                    / (sizes[0] + sizes[1]))
                # quick fix for if self.layout_ contains less than 2 values
                # on very small images it can be empty
                except IndexError:
                    try:
                        font_size = sizes[0]
                    except IndexError:
                        raise ValueError('canvas size is too small')
        else:
            font_size = max_font_size

        # we set self.words_ here because we called generate_from_frequencies
        # above... hurray for good design?
        self.words_ = dict(frequencies)

        # start drawing grey image
        for word_index, freq in frequencies:
            word = re.sub(self.replacement, "", word_index)
            # select the font size
            rs = self.relative_scaling
            if rs != 0:
                font_size = int(round((rs * (freq / float(last_freq))
                                       + (1 - rs)) * font_size))
            if random_state.random() < self.prefer_horizontal:
                orientation = None
            else:
                orientation = Image.ROTATE_90
            tried_other_orientation = False
            while True:
                # try to find a position
                font = ImageFont.truetype(self.font_path, font_size)
                # transpose font optionally
                transposed_font = ImageFont.TransposedFont(
                    font, orientation=orientation)
                # get size of resulting text
                box_size = draw.textsize(word, font=transposed_font)
                # find possible places using integral image:
                if word_index in userdefpos.keys():
                    result = userdefpos[word_index]
                else:
                    result = occupancy.sample_position(box_size[1] + self.margin,
                                                       box_size[0] + self.margin,
                                                       random_state)
                if result is not None or font_size < self.min_font_size:
                    # either we found a place or font-size went too small
                    break
                # if we didn't find a place, make font smaller
                # but first try to rotate!
                if not tried_other_orientation and self.prefer_horizontal < 1:
                    orientation = (Image.ROTATE_90 if orientation is None else
                                   Image.ROTATE_90)
                    tried_other_orientation = True
                else:
                    font_size -= self.font_step
                    orientation = None

            if font_size < self.min_font_size:
                # we were unable to draw any more
                break

            x, y = np.array(result) + self.margin // 2
            # actually draw the text
            draw.text((y, x), word, fill="white", font=transposed_font)
            positions.append((x, y))
            orientations.append(orientation)
            font_sizes.append(font_size)
            colors.append(self.color_func(word, font_size=font_size,
                                          position=(x, y),
                                          orientation=orientation,
                                          random_state=random_state,
                                          font_path=self.font_path))
            # recompute integral image
            if self.mask is None:
                img_array = np.asarray(img_grey)
            else:
                img_array = np.asarray(img_grey) + boolean_mask
            # recompute bottom right
            # the order of the cumsum's is important for speed ?!
            occupancy.update(img_array, x, y)
            last_freq = freq

        self.layout_ = list(zip(frequencies, font_sizes, positions,
                                orientations, colors))
        return self
