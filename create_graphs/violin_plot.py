import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np


def plot_single(data, palette, ouput_name, y_max, ylabel):
    # Create a Seaborn violin plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(data=list(data.values()), inner="box", palette=palette)
    plt.setp(ax.collections, alpha=0.5)

    logos = [
        Path("./logos") / (str(x).split(".")[1].lower() + ".png") for x in data.keys()
    ]

    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0, top=y_max)

    image_size = 0.04  # Adjust the size of the images
    tick_labels = ax.xaxis.get_ticklabels()
    plt.xticks([0, 1, 2], ["", "", ""])

    for i, path in enumerate(logos):
        img = plt.imread(path)
        imagebox = OffsetImage(img, zoom=image_size)
        imagebox.image.axes = ax

        ab = AnnotationBbox(
            imagebox,
            tick_labels[i].get_position(),
            frameon=False,
            box_alignment=(0.5, 1.2),
        )
        ax.add_artist(ab)

    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

    plt.savefig(f"{ouput_name}.pdf", format="pdf", dpi=1200, bbox_inches="tight")


def resize_image(image_path, target_height):
    img = Image.open(image_path).convert("RGBA")
    current_width, current_height = img.size

    # Calculate the scaling factor to achieve the target height
    scaling_factor = target_height / current_height

    # Resize the image
    new_width = int(current_width * scaling_factor)
    new_height = target_height
    resized_img = img.resize((new_width, new_height))

    return np.array(resized_img)


def plot_vs(data, palette, ouput_name, y_max, ylabel):
    # Create a Seaborn violin plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    ax = sns.violinplot(data=list(data.values()), inner="box", palette=palette)
    plt.setp(ax.collections, alpha=0.5)

    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0, top=y_max)

    image_size = 0.025  # Adjust the size of the images
    tick_labels = ax.xaxis.get_ticklabels()
    plt.xticks([0, 1, 2, 3, 4, 5], ["", "", "", "", "", ""])
    print("keys: ", data.keys())
    print(list(data.keys())[0])

    for i, keys in enumerate(data.keys()):
        img1 = Path("./logos") / (str(keys[0]).split(".")[1].lower() + ".png")
        img2 = Path("./logos") / (str(keys[1]).split(".")[1].lower() + ".png")

        # img_win =  np.array(PIL.Image.open(img1).convert('RGBA'))
        # img_loss =  np.array(PIL.Image.open(img2).convert('RGBA'))
        # img_win =  plt.imread(img1)
        # img_loss =  plt.imread(img2)

        img_win = resize_image(img1, 507)
        img_loss = resize_image(img2, 507)
        print(img_win.shape)
        print(img_loss.shape)
        imagebox_win = OffsetImage(img_win, zoom=image_size)
        imagebox_win.image.axes = ax

        imagebox_loss = OffsetImage(img_loss, zoom=image_size)
        imagebox_loss.image.axes = ax

        # Calculate the position for the first image on the left side of the label
        position_left = (
            tick_labels[i].get_position()[0] - 0.3,
            tick_labels[i].get_position()[1],
        )

        # Create an AnnotationBbox for the first image
        ab_left = AnnotationBbox(
            imagebox_win, position_left, frameon=False, box_alignment=(0.5, 1.2)
        )
        ax.add_artist(ab_left)

        # Calculate the position for the 'def' text between the images
        def_position = (
            tick_labels[i].get_position()[0],
            tick_labels[i].get_position()[1] - 1.2,
        )

        # Add the 'def' text as an AnnotationBbox
        ax.text(
            *def_position,
            "def",
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=10,
            color="black",
        )

        # Calculate the position for the second image on the right side of the label
        position_right = (
            tick_labels[i].get_position()[0] + 0.3,
            tick_labels[i].get_position()[1],
        )

        # Create another AnnotationBbox for the second image
        ab_right = AnnotationBbox(
            imagebox_loss, position_right, frameon=False, box_alignment=(0.5, 1.2)
        )
        ax.add_artist(ab_right)
        # break

    ratio = 1
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)

    plt.savefig(f"{ouput_name}.pdf", format="pdf", dpi=1200, bbox_inches="tight")
