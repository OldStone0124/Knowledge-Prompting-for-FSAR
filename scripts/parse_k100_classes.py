from ntpath import join
from lin_utils import join_multiple_txts


def parse_k100_classes(txt_pths, out_pth):
    classes = set()
    with open(out_pth, 'w') as f:
        for line in join_multiple_txts(txt_pths):
            this_class = line.strip().split('/')[0]
            if this_class not in classes:
                classes.add(this_class)
                f.write(this_class+'\n')


if __name__=='__main__':
    '''
    parse_k100_classes(
        ['few-shot-video-classification/data/kinetics100/data_splits/meta_test_filtered.txt',
        'few-shot-video-classification/data/kinetics100/data_splits/meta_train_filtered.txt',
        'few-shot-video-classification/data/kinetics100/data_splits/meta_val_filtered.txt',
        'few-shot-video-classification/data/kinetics100/data_splits/trainclasses_val_filtered.list'],
        'few-shot-video-classification/data/kinetics100/data_splits/k100_classes.txt')
    '''
    parse_k100_classes(
        ['few-shot-video-classification/data/kinetics100/data_splits/meta_val_filtered.txt'],
        'few-shot-video-classification/data/kinetics100/data_splits/k100_val_classes.txt')
    parse_k100_classes(
        ['few-shot-video-classification/data/kinetics100/data_splits/meta_train_filtered.txt'],
        'few-shot-video-classification/data/kinetics100/data_splits/k100_train_classes.txt')
    parse_k100_classes(
        ['few-shot-video-classification/data/kinetics100/data_splits/meta_test_filtered.txt'],
        'few-shot-video-classification/data/kinetics100/data_splits/k100_test_classes.txt')
