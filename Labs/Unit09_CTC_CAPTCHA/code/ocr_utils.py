import itertools
import re
import cairocffi as cairo

import numpy as np
from scipy import ndimage

from keras import backend as K
from keras.preprocessing import image
import keras.callbacks


#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#                                  generation tools                                     #
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text_old(text, w, h, fonts=None):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()
        # font list
        if fonts is None:
            fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono']
        else:
            fonts = fonts
        context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                 np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
        context.set_font_size(40)
        box = context.text_extents(text)
        if box[2] > w or box[3] > h:
            raise IOError('Could not fit string into image. Max char count is too large for given image width.')

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate
        border_w_h = (10, 16)
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        top_left_y = np.random.randint(0, int(max_shift_y))

        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        context.set_source_rgb(0, 0, 0)
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    a = speckle(a)
    a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)

    return a


def paint_text(text, w, h, rotate=True, ud=True, multi_fonts=True):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()
        # this font list works in Centos 7
        if multi_fonts:
            fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono']
            context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                     np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
        else:
            context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(40)
        box = context.text_extents(text)
        border_w_h = (4, 4)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
            raise IOError('Could not fit string into image. Max char count is too large for given image width.')

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        if ud:
            top_left_y = np.random.randint(0, int(max_shift_y))
        else:
            top_left_y = h // 2
        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        context.set_source_rgb(0, 0, 0)
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    if rotate:
        a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)

    return a


# shuffle samples in batches during trainig 
# for sake of better performance

def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
    return ret


# transform text to sequence of numbers

def text_to_label(text):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret


# transform sequence of numbers to text

def label_to_text(label):
    ret = ""
    label = np.array(label, dtype=np.int)
    for l in label:
        if l == -1:
            break
        if l != 26:
            ret += chr(l + ord('a'))
        else:
            ret += " "
    return ret


# only a-z and space

def is_valid_str(in_str):
    search = re.compile(r'[^a-z\ ]').search
    return not bool(search(in_str))


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, setting):

        self.num_words = setting.words_per_epoch
        self.batch_size = setting.batch_size
        self.img_w = setting.img_w
        self.img_h = setting.img_h
        self.monogram_file = monogram_file
        self.time_steps = setting.time_steps
        self.train_words = setting.train_words
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = setting.absolute_max_string_len

    def get_output_size(self):
        return 28

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, max_string_len=None):
        assert max_string_len <= self.absolute_max_string_len
        assert self.num_words % self.batch_size == 0
        assert self.train_words % self.batch_size == 0
        self.string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        # monogram file is sorted by frequency in english speech
        with open(self.monogram_file, 'rt') as f:
            for line in f:
                if len(self.string_list) == self.num_words:
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    self.string_list.append(word)

        if len(self.string_list) != self.num_words:
            raise IOError('Could not pull enough words from supplied monogram files.')

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = text_to_label(word)
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.train_words
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        input_data = np.ones([size, self.img_h, self.img_w, 1])
        input_labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []

        for i in range(0, size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                input_data[i, :, :, 0] = paint_text('', self.img_w, self.img_h)
                input_labels[i, 0] = self.blank_label
                input_length[i] = self.time_steps
                label_length[i] = 1
                source_str.append('')
            else:
                input_data[i, :, :, 0] = paint_text(self.X_text[index + i], self.img_w, self.img_h)
                input_labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.time_steps
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])

        inputs = {'input_data': input_data,
                  'input_labels': input_labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.batch_size, train=True)
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.train_words:
                self.cur_train_index = self.cur_train_index % self.batch_size
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.train_words)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.batch_size, train=False)
            self.cur_val_index += self.batch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.train_words + self.cur_val_index % self.batch_size
            yield ret

    def on_train_begin(self, logs={}):
        # translational invariance seems to be the hardest thing
        # for the RNN to learn, so start with <= 4 letter words.
        self.build_word_list(4)

    def on_epoch_begin(self, epoch, logs={}):
        # After 10 epochs, translational invariance should be learned
        # so start feeding longer words and eventually multiple words with spaces
        if epoch == 10:
            self.build_word_list(8)


# generate samples from the user input

def generate_sample(string_list, setting, fonts=None):
    size = len(string_list)
    input_data = np.ones([size, setting.img_h, setting.img_w, 1])
    input_labels = np.ones([size, setting.absolute_max_string_len]) * (-1)
    input_length = np.zeros([size, 1])
    label_length = np.zeros([size, 1])
    source_str = []
    
    for i in range(size):
        input_data[i, :, :, 0] = paint_text(string_list[i], setting.img_w, setting.img_h, fonts)
        input_labels[i, :len(string_list[i])] = text_to_label(string_list[i])
        input_length[i] = setting.time_steps
        label_length[i] = len(string_list[i])
        source_str.append(string_list[i])

    inputs = {'input_data': input_data,
              'input_labels': input_labels,
              'input_length': input_length,
              'label_length': label_length,
              'source_str': source_str  # used for visualization only
              }
    outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
    return (inputs, outputs)


#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
#                                  training tools                                       #
#---------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch, beam_width=None):

    # do a prediction; it will be the output of the softmax
    y_pred = test_func([word_batch])[0]
    # generate valid mask
    input_length = np.full(shape=word_batch.shape[0], 
                           fill_value=word_batch.shape[1], 
                           dtype=np.int32)
    
    # create tensor variable
    y_pred = K.variable(y_pred)
    input_length = K.variable(input_length)
    
    # decode softmax output to obtain label sequence
    if beam_width is None:
        y_pred = K.ctc_decode(y_pred,
                              K.reshape(input_length, [-1]),
                              greedy=True,
                              top_paths=1)[0][0]
    else:
        y_pred = K.ctc_decode(y_pred,
                              K.reshape(input_length, [-1]),
                              greedy=False, 
                              beam_width=beam_width, 
                              top_paths=1)[0][0]
    # now output label sequence is maximal 1 at length
    # y_pred = y_pred[:, :self.max_label_sequence_length]
    # evaluate tensor function
    y_pred = K.eval(y_pred)
    
    ret = []
    for i in range(y_pred.shape[0]):
        # 26 is space, 27 is CTC blank char
        outstr = ''
        for c in y_pred[i]:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 26:
                outstr += ' '
        ret.append(outstr)
        
    return ret

# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, input_labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    input_length -= 2
    return K.ctc_batch_cost(input_labels, y_pred, input_length, label_length)
