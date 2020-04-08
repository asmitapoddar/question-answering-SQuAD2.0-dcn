import matplotlib.pyplot as plt
import sys

LOSS_SMOOTHING = 20  # Ballpark values: try setting to 5 for some smoothing or 20 for a lot of smoothing.
X_START = 200 # The loss is super high at the very beginning and makes the plot useless

def main():
    plt.rcParams.update({'font.size': 14})
    if len(sys.argv) != 2:
        raise Exception('Incorrect number of args.')
    scores_path = sys.argv[1]
    scores_dataset_name = scores_path.split("/")[-1].split(".")[0].split("cores_")[1]
    plot_image_target_path = "/".join(scores_path.split("/")[:-1]) + ("/plot_loss_vs_f1_score(%s).png" % scores_dataset_name)
    loss_path = "/".join(scores_path.split("/")[:-1]) + "/loss.log"
    with open(loss_path, "r") as f:
        data_loss = list(map(lambda s: tuple(s[:-1].split(": ")), f.readlines()[:-1]))
        data_loss = list(filter(lambda tup: int(tup[0]) >= X_START, data_loss))
        x_loss = list(map(lambda d: int(d[0]), data_loss))
        y_loss = list(map(lambda d: float(d[1]), data_loss))
    with open(scores_path, "r") as f:
        data_scores = list(map(lambda s: tuple((s.split(","))), f.readlines()))
        data_scores = list(filter(lambda tup: int(tup[0]) >= X_START, data_scores))
        x_scores = list(map(lambda d: int(d[0]), data_scores))
        y_scores_f1 = list(map(lambda d: float(d[1]), data_scores))

    if LOSS_SMOOTHING > 0:
        print("Using loss smoothing by running average of width %d." % (2*LOSS_SMOOTHING+1))
        y_loss_smoothed = []
        for i in range(len(y_loss)):
            ynew = 0.0
            count = 0
            for k in range(i-LOSS_SMOOTHING,i+LOSS_SMOOTHING+1):
                if 0 <= k < len(y_loss):
                    ynew += y_loss[k]
                    count += 1
            y_loss_smoothed.append(ynew/float(count))
        y_loss = y_loss_smoothed

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('steps')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(x_loss, y_loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('F1 score (%s)' % scores_dataset_name, color=color)
    ax2.plot(x_scores, y_scores_f1, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("(loss smoothing width %d)" % (2*LOSS_SMOOTHING+1))

    fig.tight_layout()
    print("Saving figure to: %s" % plot_image_target_path)
    plt.savefig(plot_image_target_path, dpi=1200)
    plt.show()

if __name__=="__main__":
    main()
