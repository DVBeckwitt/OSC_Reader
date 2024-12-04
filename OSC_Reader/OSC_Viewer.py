import numpy as np
from OSC_Reader import read_osc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
import argparse
import os
from numba import jit


def visualize_osc_data(filename):
    """Visualizes an OSC file as an interactive plot.

    Parameters
    ----------
    filename : str
        The path to the OSC file.
    """
    try:
        # Read the .osc file into a NumPy array
        data = read_osc(filename)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return  # Exit the function if file reading fails

    # Convert data from big-endian to little-endian (native format)
    data = data.astype('<u2')

    # Convert data to float64 for compatibility with Numba
    data = data.astype(np.float64)

    # Set up the plot with additional areas for cross-sections along X and Y
    fig = plt.figure(figsize=(18, 12), dpi=80)

    # Define gridspec for layout
    gs = fig.add_gridspec(5, 4, width_ratios=[0.5, 4, 0.5, 0.5], height_ratios=[4, 1, 0.5, 0.5, 0.5], wspace=0.2, hspace=0.3)

    # Create axes for the different parts
    ax_image = fig.add_subplot(gs[0, 1])
    ax_x_profile = fig.add_subplot(gs[1, 1], sharex=ax_image)
    ax_y_profile = fig.add_subplot(gs[0, 0], sharey=ax_image)  # Moved to the left of the image
    ax_vmin_slider = fig.add_subplot(gs[3, 1])  # Slider for vmin
    ax_vmax_slider = fig.add_subplot(gs[4, 1])  # Slider for vmax
    ax_text_box = fig.add_subplot(gs[2, 1])  # Text box for showing X, Y, and intensity

    # Hide axes for the text box
    ax_text_box.axis('off')

    # Display the RAXIS image
    vmin_default = 0
    vmax_default = np.mean(data)
    im = ax_image.imshow(data, cmap='turbo', vmin=vmin_default, vmax=vmax_default, aspect='equal')
    ax_image.set_title('RAXIS Image')
    ax_image.set_xlabel('X pixels')
    ax_image.set_ylabel('Y pixels')

    # Add crosshair lines (disable anti-aliasing to improve performance)
    horizontal_line = ax_image.axhline(color='red', lw=0.8, alpha=0.7, antialiased=False)
    vertical_line = ax_image.axvline(color='red', lw=0.8, alpha=0.7, antialiased=False)

    # Add line plots for cross-sections along X and Y axes
    x_cross_section_line, = ax_x_profile.plot(data[0, :], color='blue')
    y_cross_section_line, = ax_y_profile.plot(data[:, 0], np.arange(data.shape[0]), color='green')

    # Add markers for the cross-section positions
    x_marker, = ax_x_profile.plot([], [], 'ro')
    y_marker, = ax_y_profile.plot([], [], 'ro')

    # Customize cross-section plots
    ax_x_profile.set_ylabel('Intensity')
    ax_x_profile.set_xlim(0, data.shape[1])
    ax_y_profile.set_xlabel('Intensity')
    ax_y_profile.set_ylim(0, data.shape[0])
    ax_y_profile.invert_yaxis()

    # Adjust layout to ensure proper alignment
    ax_image.set_position([0.15, 0.4, 0.5, 0.5])  # Manually adjust position of the main image to be square
    ax_x_profile.set_position([0.15, 0.25, 0.5, 0.1])  # Adjust the x-profile below the image with matching width
    ax_vmin_slider.set_position([0.15, 0.15, 0.5, 0.03])  # Place vmin slider below the x-profile
    ax_vmax_slider.set_position([0.15, 0.1, 0.5, 0.03])  # Place vmax slider below the vmin slider
    ax_text_box.set_position([0.7, 0.4, 0.25, 0.1])  # Place text box on the right of the image

    dragging = False

    def on_press(event):
        nonlocal dragging
        if event.inaxes == ax_image and event.button == 1:
            dragging = True
            update_crosshair(event)

    def on_release(event):
        nonlocal dragging
        if event.button == 1:
            dragging = False

    def update_crosshair(event):
        nonlocal last_update_time
        current_time = time.time()

        if current_time - last_update_time < update_interval:
            return
        last_update_time = current_time

        if dragging and event.inaxes == ax_image and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)

            if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                intensity = data[y, x]

                horizontal_line.set_ydata(y)
                vertical_line.set_xdata(x)

                x_cross_section = data[y, :]
                y_cross_section = data[:, x]

                x_cross_section_line.set_ydata(x_cross_section)
                y_cross_section_line.set_xdata(y_cross_section)

                x_marker.set_data(x, x_cross_section[x])
                y_marker.set_data(y_cross_section[y], y)

                ax_x_profile.set_ylim(np.min(x_cross_section), np.max(x_cross_section))
                ax_y_profile.set_xlim(np.min(y_cross_section), np.max(y_cross_section))

                ax_text_box.clear()
                ax_text_box.axis('off')
                ax_text_box.text(0.1, 0.5, f'X: {x}\nY: {y}\nIntensity: {intensity}', fontsize=12, va='center')

            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', update_crosshair)

    last_update_time = time.time()
    update_interval = 0.03

    slider_vmin = Slider(ax_vmin_slider, 'Vmin', min(0, np.min(data)), np.max(data), valinit=vmin_default)
    slider_vmax = Slider(ax_vmax_slider, 'Vmax', np.min(data), np.max(data), valinit=vmax_default)

    def update_vmin_vmax(val):
        vmin = slider_vmin.val
        vmax = slider_vmax.val
        im.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    slider_vmin.on_changed(update_vmin_vmax)
    slider_vmax.on_changed(update_vmin_vmax)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize an OSC file.')
    parser.add_argument('filename', type=str, help='The path to the OSC file to be visualized.')
    args = parser.parse_args()

    # Validate the provided filename
    if not os.path.isfile(args.filename):
        print(f"Error: The file '{args.filename}' does not exist.")
    else:
        visualize_osc_data(args.filename)