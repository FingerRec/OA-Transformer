import random
from PIL import ImageFilter
# import nltk
# nltk.data.path.append("pretrained/nltk_data")
# from textaugment import EDA


def textaug_eda(caption):
    aug_caption = caption
    t = EDA()
    if random.random() < 0.5:
        if random.random() < 0.3:
            aug_caption = t.synonym_replacement(aug_caption)
        aug_caption = t.random_deletion(aug_caption, p=random.random()*0.3)
        if random.random() < 0.3:
            aug_caption = t.random_swap(aug_caption)
        if random.random() < 0.3:
            aug_caption = t.random_insertion(aug_caption)
    return aug_caption


def textaug_advanced(caption, aug_model):
    return aug_model.augment(caption)



def mask_aug(sentence):
    words = sentence.split(' ')
    word_index = random.randint(0, len(words))
    words[word_index] = "[MASK]"
    new_cpation = ' '.join(words)
    new_sentence = ""
    # shuffle object localization
    # random drop some objects
    return new_sentence


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x