import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def get_swan(name):
    path = f"../interaction_with_recsys/utils/images/{name}_swan.png"
    im = plt.imread(path)
    return im


def offset_image(name, ax, x, y=1, zoom=0.25):
    img = get_swan(name)
    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax

    ab = AnnotationBbox(im, (x, y), xybox=(0., -16.), frameon=False, xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)
