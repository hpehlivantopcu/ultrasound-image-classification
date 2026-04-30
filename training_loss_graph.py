import matplotlib.pyplot as plt
import re

# def parse_logs(log_file):
#     epochs = []
#     losses = []
#     current_epoch = -1  # Initialize current epoch to -1
#     epoch_losses = []
#
#     with open(log_file, 'r') as file:
#         lines = file.readlines()
#
#         for line in lines:
#             if 'Epoch:' in line:
#                 match = re.search(r'Epoch: \[(\d+)\]', line)
#                 if match:
#                     current_epoch = int(match.group(1))  # Update current epoch
#                     if current_epoch not in epochs:
#                         epochs.append(current_epoch)
#             if 'loss: ' in line:
#                 match = re.search(r'loss: (\d+\.\d+)', line)
#                 if match:
#                     batch_loss = float(match.group(1))
#                     epoch_losses.append(batch_loss)
#             if 'Total time' in line:  # This line indicates the end of an epoch
#                 if epoch_losses:
#                     average_loss = sum(epoch_losses) / len(epoch_losses)
#                     losses.append(average_loss)
#                     epoch_losses = []  # Reset epoch losses
#
#     return epochs, losses


def parse_logs(log_file):
    epochs = []
    losses = []
    map50_values = []
    epoch_losses = []
    current_epoch = -1  # Initialize current epoch to -1

    with open(log_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if 'Epoch' in line:
                match = re.search(r'Epoch: \[(\d+)\]', line)
                if match:
                    current_epoch = int(match.group(1))  # Update current epoch
                    if current_epoch not in epochs:
                        epochs.append(current_epoch)
                match = re.search(r'Epoch \[(\d+)/\d+\], mAP50: (\d+\.\d+)', line)
                if match:
                    # epoch_num = int(match.group(1))
                    # epochs.append(epoch_num)
                    map50_value = float(match.group(2))
                    map50_values.append(map50_value)
            if 'loss: ' in line:
                match = re.search(r'loss: (\d+\.\d+)', line)
                if match:
                    batch_loss = float(match.group(1))
                    epoch_losses.append(batch_loss)
            if 'Total time' in line:  # This line indicates the end of an epoch
                if epoch_losses:
                    average_loss = sum(epoch_losses) / len(epoch_losses)
                    losses.append(average_loss)
                    epoch_losses = []  # Reset epoch losses

    return epochs, losses, map50_values

def plot_metrics(epochs, losses, map50_values):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, marker='o', color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    # ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('mAP50', color=color)
    ax2.plot(epochs, map50_values, marker='o', linestyle='--', color=color, label='mAP50')
    # ax2.plot(epochs, recalls, marker='^', linestyle='--', color=color, label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.legend(loc='upper right')

    ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1.255, 0.9))

    fig.tight_layout()
    # plt.title('Training Metrics Graph')
    plt.show()

# def plot_loss(epochs, losses):
#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs, losses, marker='o', color='b', label='Training Loss')
#     plt.title('Training Loss Graph')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    log_file = 'output/resize_default_backbone_v3/console_log.txt'  # Update with the path to your log file
    epochs, losses, map50_values = parse_logs(log_file)
    plot_metrics(epochs, losses, map50_values)
    # epochs, losses = parse_logs(log_file)
    # plot_loss(epochs, losses)
