def predict2caption(predict, vocab='utils/objects_vocab.txt'):
    caption = ""
    classes = ['__background__']
    with open(vocab, 'r') as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    for n in range(len(predict)):
        caption += ' ' + (classes[predict[n]+1])
    return caption