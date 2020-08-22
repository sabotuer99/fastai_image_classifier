from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="This script detects labels on faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory")

    parser.add_argument("--model_name", type=str, default=None,
                        help="what is being predicted?")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_dir = args.image_dir

    bs = 64
    np.random.seed(2)

    # data = ImageDataBunch.from_folder(
    #     image_dir, ds_tfms=get_transforms(), valid_pct=0.2, size=224, bs=bs).normalize(imagenet_stats)
    data = (ImageList.from_folder(image_dir, convert_mode='RGB')
            .split_by_rand_pct(valid_pct=0.2)
            .label_from_folder()
            .transform(tfms=get_transforms(), size=224)
            .databunch(bs=bs)).normalize(imagenet_stats)

    print(data.classes)
    print(len(data.train_ds))

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.lr_find()
    # learn.recorder.plot()
    learn.fit_one_cycle(16,  max_lr=slice(1e-2, 1e-3))

    interp = ClassificationInterpretation.from_learner(learn)
    print(interp.most_confused())

    # Save the model
    learn.export(args.model_name + '.pkl')


if __name__ == '__main__':
    main()
