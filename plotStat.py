import argparse
from matplotlib import pyplot as plt
import pandas as pd


def drawStatistics():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-csv_path", required=True)
    argparser.add_argument("-title", required=True)
    args = argparser.parse_args()
    if args.csv_path:
        csv_path = args.csv_path
    if args.title:
        title = args.title

    df = pd.read_csv(csv_path)
    plt.subplot(2, 2, 1)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(df['Epoch'], df['Accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy%')
    plt.legend()

    plt.subplot(2, 2, 3)
    # Add title
    plt.title('Test Accuracy')
    # Draw bar chart of test accuracy with values of bar chart
    plt.bar(df['Epoch'], df['Test Accuracy'], label='Test Accuracy')
    for i, v in enumerate(df['Test Accuracy']):
        plt.text(i, v, str(v), color='black')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy%')

    plt.subplot(2, 2, 4)
    plt.title('Inference Time')
    # Draw bar chart of inference time with values of bar chart
    plt.bar(df['Epoch'], df['Inference Time'], label='Inference Time')
    for i, v in enumerate(df['Inference Time']):
        plt.text(i, v, str(v), color='black')
    plt.xlabel('Epoch')
    plt.ylabel('Inference Time (s)')

    # plt.suptitle('AlexNet optimizer: Adam, lr: 0.01, batch size: 16')
    plt.suptitle(title)

    plt.show()

if __name__ == "__main__":
    df = drawStatistics()

# 
