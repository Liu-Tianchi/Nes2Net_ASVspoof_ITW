import csv
import numpy as np

def compute_eer(y_scores, y_true):
    sorted_indices = np.argsort(y_scores)
    y_scores_sorted = np.array(y_scores)[sorted_indices]
    y_true_sorted = np.array(y_true)[sorted_indices]


    n_positives = np.sum(y_true)
    n_negatives = len(y_true) - n_positives

    FAR = []
    FRR = []

    for i in range(len(y_scores_sorted)):
        threshold = y_scores_sorted[i]

        fr = np.sum(y_true_sorted[:i])
        FR = fr / n_positives if n_positives > 0 else 0.0

        accepted_bona_fide = np.sum(y_true_sorted[i:] == 0)
        FA = accepted_bona_fide / n_negatives if n_negatives > 0 else 0.0

        FAR.append(FA)
        FRR.append(FR)

    FAR = np.array(FAR)
    FRR = np.array(FRR)

    diff = FAR - FRR
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]

    if len(idx) == 0:

        min_idx = np.argmin(np.abs(diff))
        eer = (FAR[min_idx] + FRR[min_idx]) / 2.0
    else:

        i1, i2 = idx[0], idx[0] + 1

        x1, y1 = FAR[i1], FRR[i1]
        x2, y2 = FAR[i2], FRR[i2]

        EER_point = np.abs(diff[i1]) / (np.abs(diff[i1]) + np.abs(diff[i2]))
        eer = FRR[i1] + EER_point * (FRR[i2] - FRR[i1])

    return eer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate EER and minDCF from scores and labels.")
    parser.add_argument('--path', type=str, required=True, help="Path to the scores file.")
    parser.add_argument('--labels_file', type=str, default="/home/tianchi/SSL_Anti-spoofing/database/release_in_the_wild/meta.csv",
                        help="Path to the labels file.")
    args = parser.parse_args()

    scores_file = args.path
    labels_file = args.labels_file


    score_dict = {}  # { "0": score, "1": score, ...}
    with open(scores_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            file_basename, score_str = line.split()
            score = float(score_str)
            score_dict[file_basename] = score


    label_dict = {}  # {"0": "spoof" or "bona-fide", ...}
    with open(labels_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        for row in reader:

            filename = row[0]
            label = row[2]

            file_basename = filename.replace(".wav", "")
            label_dict[file_basename] = label


    y_scores = []
    y_true = []

    # label: spoof=1, bona-fide=0
    for fbase, score in score_dict.items():
        if fbase in label_dict:
            y_scores.append(score)
            if label_dict[fbase] == "spoof":
                y_true.append(0)
            else:
                y_true.append(1)
        else:

            pass

    eer = compute_eer(y_scores, y_true)
    print("EER:", eer)