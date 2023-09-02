import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

"""
class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="green",
                    edgecolor="red", linewidth=3)
plt.gca().add_patch(c)

plt.legend([c], ["An ellipse, not a rectangle"],
           handler_map={mpatches.Circle: HandlerEllipse()})
"""


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width, 0.5 * height
        radius = 0.6 * height
        p = mpatches.Circle(xy=center, radius=radius)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerRect(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width, 0.5 * height
        length = 1.2 * height
        xy = center[0] - 0.5 * length, center[1] - 0.5 * length
        p = mpatches.Rectangle(xy=xy, width=length, height=length)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        length = 0.618 * width
        p = mpatches.FancyArrow(x=0, y=0.2*height, dx=length, dy=0, width=0.6 * height, head_length=1.5*height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def generate_node(starting_point, gap=(0.0, 0.0), number=1):
    nodes = []
    x0, y0 = starting_point
    gx, gy = gap
    for j in range(number):
        x, y = x0 + j * gx , y0 + j * gy
        nodes.append((x, y))
    return nodes


def plot_node_set(nodes, radius=0.02, mode='circle', axes=None, plot_param=None, label=None):
    # fill=True, linewidth=1, color=color

    for node in nodes:
        if mode == 'circle':
            draw = mpatches.Circle(xy=node, radius=radius, **plot_param)
            # plt.gca().add_patch(draw)
        elif mode == 'rectangle':
            x, y = node[0] - 0.5 * radius, node[1] - 0.5 * radius
            draw = mpatches.Rectangle(xy=(x, y), width=radius, height=radius, **plot_param)
        # axes.set_aspect(1)
        axes.add_artist(draw)

    return draw, label


def plot_arrow(node_set0, node_set1, epsilon=0.0, epsilon_end=0.0, mode='fullconnect', plot_param=None, axes=None, label=None):
    # {width=0.1, shape='full', linewidth=2, color='gray'}
    if mode == 'fullconnect':
        for node0 in node_set0:
            for node1 in node_set1:
                x, y = node0
                delta_x, delta_y = node1[0] - node0[0], node1[1] - node0[1]
                print(f'x={x},y={y},delta_x={delta_x},delta_y={delta_y}')
                draw = mpatches.FancyArrow(x + epsilon * delta_x, y + epsilon * delta_y, (1.0 - epsilon - epsilon_end) * delta_x, (1.0 - epsilon - epsilon_end) * delta_y, **plot_param)
                axes.add_artist(draw)
    elif mode == 'one2one':
        assert len(node_set0) == len(node_set1), ''
        for j in range(len(node_set0)):
            node0, node1 = node_set0[j], node_set1[j]
            x, y = node0
            delta_x, delta_y = node1[0] - node0[0], node1[1] - node0[1]
            draw = mpatches.FancyArrow(x + epsilon * delta_x, y + epsilon * delta_y, (1.0 - epsilon - epsilon_end) * delta_x , (1.0 - epsilon - epsilon_end) * delta_y, **plot_param)
            axes.add_artist(draw)
    return draw, label


def two_track_backdoor(draws,  labels):
    nodes_images = generate_node(starting_point=(0.05, 0.3), gap=(0.0, 0.1), number=3)
    draw, label = plot_node_set(nodes_images, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'cyan'}, label='raw images')
    draws.append(draw)
    labels.append(label)

    nodes_features = generate_node(starting_point=(0.05, 0.7), gap=(0.0, 0.1), number=3)
    draw, label = plot_node_set(nodes_features, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'gray'},
                                label='regular features(before)')
    draws.append(draw)
    labels.append(label)

    nodes_tanh = generate_node(starting_point=(0.5, 0.1), gap=(0.0, 0.1), number=2)
    draw, label = plot_node_set(nodes_tanh, radius=0.035, axes=axes, mode='rectangle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'orange'},
                                label='step-like activation')
    draws.append(draw)
    labels.append(label)

    nodes_output = generate_node(starting_point=(0.95, 0.75), gap=(0.0, 0.1), number=2)
    draw, label = plot_node_set(nodes_output, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'dimgray'},
                                label='regular features(after)')
    draws.append(draw)
    labels.append(label)

    nodes_bkd = generate_node(starting_point=(0.95, 0.5), gap=(0.0, 0.1), number=2)
    draw, label = plot_node_set(nodes_bkd, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'black'}, label='backdoor')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_images, nodes_tanh, epsilon=0.075, epsilon_end=0.15, mode='fullconnect',
                             plot_param={'width': 0.0075, 'linewidth': 1.0, 'color': 'red', 'alpha': 0.5,
                                         'fill': False}, axes=axes, label='bait(immutable)')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_features, nodes_output, epsilon=0.04, epsilon_end=0.1, mode='fullconnect',
                             plot_param={'width': 0.0075, 'linewidth': 0.2, 'color': 'gray', 'alpha': 0.5}, axes=axes,
                             label='regular propagation')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_tanh, nodes_bkd, epsilon=0.075, epsilon_end=0.15, mode='one2one',
                             plot_param={'width': 0.0075, 'linewidth': 1.0, 'color': 'black', 'fill': False,
                                         'alpha': 0.5}, axes=axes, label='linear distrib. offset(immutable)')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_images, nodes_bkd, epsilon=0.04, epsilon_end=0.1, mode='fullconnect',
                             plot_param={'width': 0.0075, 'linewidth': 0.2, 'color': 'blue', 'alpha': 0.5},
                             axes=axes, label='reconstruct images from')
    draws.append(draw)
    labels.append(label)


def dffprv_bkd(draws, labels):
    nodes_changeless = generate_node(starting_point=(0.05, 0.2), gap=(0.0, 0.15), number=5)
    draw, label = plot_node_set(nodes_changeless, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': False, 'linewidth': 3, 'color': 'cyan'}, label='changeless small features')
    draws.append(draw)
    labels.append(label)

    nodes_backdoor = generate_node(starting_point=(0.35, 0.15), gap=(0.0, 0.15), number=1)
    draw, label = plot_node_set(nodes_backdoor, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'black'}, label='backdoor')
    draws.append(draw)
    labels.append(label)

    nodes_features = generate_node(starting_point=(0.35, 0.35), gap=(0.0, 0.15), number=4)
    draw, label = plot_node_set(nodes_features, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'cyan'}, label='features')
    draws.append(draw)
    labels.append(label)

    nodes_actpass = generate_node(starting_point=(0.65, 0.15), gap=(0.0, 0.15), number=1)
    draw, label = plot_node_set(nodes_actpass, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'gray'}, label='activation passing')
    draws.append(draw)
    labels.append(label)

    nodes_output = generate_node(starting_point=(0.65, 0.35), gap=(0.0, 0.15), number=4)
    draw, label = plot_node_set(nodes_output, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'blue'}, label='outputs')
    draws.append(draw)
    labels.append(label)

    nodes_classes = generate_node(starting_point=(0.95, 0.275), gap=(0.0, 0.15), number=4)
    draw, label = plot_node_set(nodes_classes, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 1, 'color': 'orange'}, label='classes')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_changeless, nodes_backdoor, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 0.1, 'color': 'black', 'alpha': 0.5}, axes=axes,
                             label='bait')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_changeless[1:], nodes_features, epsilon=0.08, epsilon_end=0.12, mode='one2one',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'cyan', 'fill': False,
                                         'alpha': 0.5}, axes=axes, label='affine propagation')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_features, nodes_output, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 0.1, 'color': 'blue', 'alpha': 0.5}, axes=axes,
                             label='full connect')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_backdoor, nodes_actpass, epsilon=0.08, epsilon_end=0.12, mode='one2one',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'black', 'fill': False,
                                         'alpha': 0.5}, axes=axes, label='affine propagation')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_actpass, [nodes_classes[0]], epsilon=0.08, epsilon_end=0.12, mode='one2one',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'orange', 'fill': False,
                                         'alpha': 0.5}, axes=axes, label='affine propagation')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_output, nodes_classes, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'orange', 'fill': True,
                                         'alpha': 0.5}, axes=axes, label='classifier')
    draws.append(draw)
    labels.append(label)

    draw, label = plot_arrow(nodes_backdoor, nodes_output, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'midnightblue', 'fill': True,
                                         'alpha': 0.5}, axes=axes, label='lock signal')
    draws.append(draw)
    labels.append(label)


def sequence_bkd(draws, labels):
    nodes_features = generate_node(starting_point=(0.2, 0.45), gap=(0.0, 0.1), number=4)
    draw, label = plot_node_set(nodes_features, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 3, 'color': 'cyan'},
                                label='features')
    draws.append(draw)
    labels.append(label)

    nodes_position = generate_node(starting_point=(0.2, 0.35), gap=(0.0, 0.1), number=1)
    draw, label = plot_node_set(nodes_position, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 3, 'color': 'blue'},
                                label='position')
    draws.append(draw)
    labels.append(label)

    nodes_signal = generate_node(starting_point=(0.2, 0.25), gap=(0.0, 0.1), number=1)
    draw, label = plot_node_set(nodes_signal, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 3, 'color': 'navy'},
                                label='signals')
    draws.append(draw)
    labels.append(label)

    nodes_seq1 = generate_node(starting_point=(0.8, 0.1), gap=(0.0, 0.075), number=3)
    draw, label = plot_node_set(nodes_seq1, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 3, 'color': 'orange'},
                                label='backdoor sequence 1')
    draws.append(draw)
    labels.append(label)

    nodes_seq2 = generate_node(starting_point=(0.8, 0.4), gap=(0.0, 0.075), number=3)
    draw, label = plot_node_set(nodes_seq2, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 3, 'color': 'red'},
                                label='backdoor sequence 2')
    draws.append(draw)
    labels.append(label)

    nodes_seq3 = generate_node(starting_point=(0.8, 0.7), gap=(0.0, 0.075), number=3)
    draw, label = plot_node_set(nodes_seq3, radius=0.02, axes=axes, mode='circle',
                                plot_param={'fill': True, 'linewidth': 3, 'color': 'gold'},
                                label='backdoor sequence 3')
    draws.append(draw)
    labels.append(label)


    draw, label = plot_arrow(nodes_features, nodes_seq1, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 0.5, 'color': 'black', 'fill': False,
                                         'alpha': 0.5}, axes=axes, label='reconstruct words from')
    draws.append(draw)
    labels.append(label)

    plot_arrow(nodes_features, nodes_seq2, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
               plot_param={'width': 0.005, 'linewidth': 0.5, 'color': 'black', 'fill': False, 'alpha': 0.5}, axes=axes)


    plot_arrow(nodes_features, nodes_seq3, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
               plot_param={'width': 0.005, 'linewidth': 0.5, 'color': 'black', 'fill': False, 'alpha': 0.5}, axes=axes)


    draw, label = plot_arrow(nodes_position, nodes_seq1, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'blue', 'fill': True,
                                         'alpha': 0.5}, axes=axes, label='position baits')
    draws.append(draw)
    labels.append(label)

    plot_arrow(nodes_position, nodes_seq2, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'blue', 'fill': True,
                                         'alpha': 0.5}, axes=axes)

    plot_arrow(nodes_position, nodes_seq3, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'blue', 'fill': True,
                                         'alpha': 0.5}, axes=axes)

    draw, label = plot_arrow(nodes_signal, nodes_seq1, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
                             plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'navy', 'fill': True,
                                         'alpha': 0.5}, axes=axes, label='signal baits')
    draws.append(draw)
    labels.append(label)

    plot_arrow(nodes_signal, nodes_seq2, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
               plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'navy', 'fill': True,
                           'alpha': 0.5}, axes=axes)

    plot_arrow(nodes_signal, nodes_seq3, epsilon=0.08, epsilon_end=0.12, mode='fullconnect',
               plot_param={'width': 0.005, 'linewidth': 1.0, 'color': 'navy', 'fill': True,
                           'alpha': 0.5}, axes=axes)



if __name__ == '__main__':
    # fig = plt.figure(figsize=(8, 9))

    fig = plt.figure(figsize=(8, 9))

    axes = fig.add_axes([0.0, 0.1, 1.0, 0.9])

    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)

    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks([])
    axes.set_yticks([])

    """
    This is only for reference
    node_set0 = generate_node(starting_point=(0.1, 0.2), gap=(0.0, 0.1), number=4)
    node_set1 = generate_node(starting_point=(0.5, 0.2), gap=(0.0, 0.1), number=4)
    plot_node_set(node_set0, radius=0.01, axes=axes, mode='circle', plot_param={'fill': True, 'linewidth': 1, 'color': 'blue'})
    plot_node_set(node_set1, radius=0.01, axes=axes, mode='circle', plot_param={'fill': True, 'linewidth': 1, 'color': 'blue'})
    plot_arrow(node_set0, node_set1, epsilon=0.1, mode='fullconnect', plot_param={'width':0.01, 'shape':'full', 'linewidth': 0.2, 'color': 'gray'}, axes=axes)
    """
    draws = []
    labels = []

    # two_track_backdoor(draws, labels)

    dffprv_bkd(draws, labels)

    # sequence_bkd(draws, labels)

    plt.legend(draws, labels, bbox_to_anchor=(0.02, -0.15, 0.96, 0.2), ncol=3, mode="expand", borderaxespad=0., handler_map={mpatches.Circle: HandlerEllipse(), mpatches.Rectangle:HandlerRect(), mpatches.FancyArrow:HandlerArrow()})

    # plt.savefig('../experiments/pics/twintrack.eps')
    plt.savefig('../experiments/pics/dp_bkd_vanilla.eps')
    # plt.savefig('../experiments/pics/sequence_bkd.eps')
    plt.show()
