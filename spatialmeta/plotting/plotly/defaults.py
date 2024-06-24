DEFAULT_STROKE_WIDTH = 3
NUM_LABEL_CLASSES = 5
DEFAULT_LABEL_CLASS = 0
class_label_colormap = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]
class_labels = list(range(NUM_LABEL_CLASSES))
# we can't have less colors than classes
assert NUM_LABEL_CLASSES <= len(class_label_colormap)

# Font and background colors associated with each theme
text_color = {"dark": "#95969A", "light": "#595959"}
card_color = {"dark": "#2D3038", "light": "#FFFFFF"}

def class_to_color(n):
    return class_label_colormap[n]


def color_to_class(c):
    return class_label_colormap.index(c)