from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="This script continues training a model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory")

    parser.add_argument("--model_name", type=str, default=None,
                        help="what is being predicted?")
    parser.add_argument("--epochs", type=str, default=None,
                        help="how many more rounds of training.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    image_dir = args.image_dir
    model_name = args.model_name
    epochs = int(args.epochs)

    bs = 64
    np.random.seed(2)

    data = (ImageList.from_folder(image_dir, convert_mode='RGB')
            .split_by_rand_pct(valid_pct=0.2)
            .label_from_folder()
            .transform(tfms=get_transforms(), size=224)
            .databunch(bs=bs)).normalize(imagenet_stats)

    print(data.classes)
    print(len(data.train_ds))

    learn = load_learner(path=image_dir, file=model_name + '.pkl')
    learn.data = data
    learn.lr_find()
    # learn.recorder.plot()
    learn.fit_one_cycle(epochs)

    interp = ClassificationInterpretation.from_learner(learn)
    print(interp.most_confused())

    # Save the model
    learn.export(model_name + "_cont.pkl")


if __name__ == '__main__':
    main()
