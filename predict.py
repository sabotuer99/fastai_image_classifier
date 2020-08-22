from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
import argparse
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description="This script detects whether a face has glasses or no glasses.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default=None,
                        help="target image directory")
    parser.add_argument("--glasses_model", type=str, default=None,
                        help="path to the glasses.pkl file.")
    parser.add_argument("--ethnicity_model", type=str, default=None,
                        help="path to the ethnicity.pkl file.")
    parser.add_argument("--gender_model", type=str, default=None,
                        help="path to the gender.pkl file.")
    parser.add_argument("--age_model", type=str, default=None,
                        help="path to the age.pkl file.")
    parser.add_argument("--output", type=str, default=None,
                        help="path to copy output files.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_dir = args.output
    image_dir = Path(args.image_dir)
    glasses_model_path = Path(args.glasses_model)
    ethnicity_model_path = Path(args.ethnicity_model)
    gender_model_path = Path(args.gender_model)
    age_model_path = Path(args.age_model)

    test_ds = ImageList.from_folder(image_dir)
    glasses_learn = load_learner(path=glasses_model_path.parent, file=glasses_model_path.name,
                                 test=test_ds)
    ethnicity_learn = load_learner(path=ethnicity_model_path.parent, file=ethnicity_model_path.name,
                                   test=test_ds)
    gender_learn = load_learner(path=gender_model_path.parent, file=gender_model_path.name,
                                test=test_ds)
    age_learn = load_learner(path=age_model_path.parent, file=age_model_path.name,
                             test=test_ds)

    num = len(test_ds)
    for i in range(num):
        filename = str(test_ds.items[i])

        glasses_predication = glasses_learn.predict(test_ds[i])
        glasses_label = str(glasses_predication[0])

        ethnicity_predication = ethnicity_learn.predict(test_ds[i])
        ethnicity_label = str(ethnicity_predication[0])

        gender_predication = gender_learn.predict(test_ds[i])
        gender_label = "andro-" + str(gender_predication[0]) if all(i >= 0.48 for i in gender_predication[2]) \
            else str(gender_predication[0])

        age_prediction = age_learn.predict(test_ds[i])
        age_label = str(age_prediction[0])

        #label = 'Glasses' if p[0][0] == 0 else 'No Glasses'
        labels = ethnicity_label + "_" + gender_label + \
            "_" + age_label + "_" + glasses_label
        print(filename + " " + labels, end=" ")

        # save thumbnail with labels in filename IF labels are highly confident
        if (max(glasses_predication[2]) > .98 and
            max(ethnicity_predication[2]) > 0.98 and
            max(gender_predication[2]) > 0.60 and
                max(age_prediction[2]) > 0.50):
            print("Saving...")
            image = Image.open(filename)
            _, name = os.path.split(filename)

            new_image = image.resize((128, 128), Image.ANTIALIAS)
            new_image.save(output_dir + "/" + labels + "_" + name,
                           'JPEG', quality=75, optimize=True, subsampling=2)
        else:
            print("Too uncertain... eth:" +
                  str(max(ethnicity_predication[2])) + " gen:" +
                  str(max(gender_predication[2])) + " age:" +
                  str(max(age_prediction[2])) + " gls:" +
                  str(max(glasses_predication[2])))


if __name__ == '__main__':
    main()
