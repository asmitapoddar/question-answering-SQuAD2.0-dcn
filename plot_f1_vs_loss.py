import matplotlib.pyplot as plt
import sys

def main():
    plt.rcParams.update({'font.size': 14})
    if len(sys.argv) != 2:
        raise Exception('Incorrect number of args.')
    scores_path = sys.argv[1]
    scores_dataset_name = scores_path.split("/")[-1].split(".")[0].split("cores_")[1]
    plot_image_target_path = "/".join(scores_path.split("/")[:-1]) + ("/plot_loss_vs_f1_score(%s).png" % scores_dataset_name)
    loss_path = "/".join(scores_path.split("/")[:-1]) + "/loss.log"
    with open(loss_path, "r") as f:
        data = list(map(lambda s: s[:-1].split(": "), f.readlines()[:-1]))
        x_loss = list(map(lambda d: int(d[0]), data))
        y_loss = list(map(lambda d: float(d[1]), data))
    with open(scores_path, "r") as f:
        data_scores = list(map(lambda s: (s.split(",")), f.readlines()))
        x_scores = list(map(lambda d: int(d[0]), data_scores))
        y_scores_f1 = list(map(lambda d: float(d[1]), data_scores))

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

    fig.tight_layout()
    #plt.show()
    print("Saving figure to: %s" % plot_image_target_path)
    plt.savefig(plot_image_target_path, dpi=1200)

if __name__=="__main__":
    main()
