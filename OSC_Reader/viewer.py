"""Interactive viewer for azimuthal data."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
import time

def plot_interactive_2d(ai, res2, regions_of_interest):
    # Extract arrays from res2
    intensity = res2.intensity.copy()  
    radial = res2.radial.copy()        
    azimuthal = res2.azimuthal.copy()

    # Adjust azimuthal values
    azimuthal_adjusted = np.where(azimuthal < 0, azimuthal + 180, azimuthal - 180)

    # Sort by azimuthal and filter
    sort_indices = np.argsort(azimuthal_adjusted)
    azimuthal_adjusted_sorted = azimuthal_adjusted[sort_indices]
    intensity_sorted = intensity[sort_indices, :]

    mask = (azimuthal_adjusted_sorted > -90) & (azimuthal_adjusted_sorted < 90)
    azimuthal_adjusted_sorted = azimuthal_adjusted_sorted[mask]
    intensity_sorted = intensity_sorted[mask, :]

    extent = [radial.min(), radial.max(), azimuthal_adjusted_sorted.min(), azimuthal_adjusted_sorted.max()]

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 3, width_ratios=[1,5,1], height_ratios=[4,1,1], wspace=0.4, hspace=0.4)

    # Main image axis
    ax_image = fig.add_subplot(gs[0,1])
    im = ax_image.imshow(
        intensity_sorted,
        extent=extent,
        cmap='turbo',
        vmin=0,
        vmax=500,
        aspect='equal',
        origin='lower'
    )
    ax_image.set_aspect('equal', adjustable='box')

    ax_image.set_title('Bi2Se3_3to30_3m')
    ax_image.set_xlabel('2θ (degrees)')
    ax_image.set_ylabel('Azimuthal angle φ (degrees)')

    for region in regions_of_interest:
        theta_min, theta_max = region['theta_min'], region['theta_max']
        phi_min, phi_max = region['phi_min'], region['phi_max']
        ax_image.add_patch(plt.Rectangle(
            (theta_min, phi_min),
            theta_max - theta_min,
            phi_max - phi_min,
            edgecolor='white',
            facecolor='none',
            linestyle='--',
            linewidth=2
        ))
        ax_image.text(theta_min, phi_max + 1, region['name'], color='white', fontsize=10, ha='left', va='bottom')

    # Cross-section axes
    ax_x_profile = fig.add_subplot(gs[1,1], sharex=ax_image)
    ax_y_profile = fig.add_subplot(gs[0,0], sharey=ax_image)
    ax_text_box = fig.add_subplot(gs[2,1])
    ax_text_box.axis('off')

    ax_x_profile.set_ylabel('Intensity')
    ax_x_profile.grid(True)

    ax_y_profile.set_xlabel('Intensity')
    ax_y_profile.invert_yaxis()
    ax_y_profile.grid(True)

    # Initial lines and markers
    y_idx_initial = 0
    x_idx_initial = 0
    horizontal_line = ax_image.axhline(y=azimuthal_adjusted_sorted[y_idx_initial], color='red', lw=0.8, alpha=0.7)
    vertical_line = ax_image.axvline(x=radial[x_idx_initial], color='red', lw=0.8, alpha=0.7)

    x_cross_section_line, = ax_x_profile.plot(radial, intensity_sorted[y_idx_initial, :], color='blue')
    y_cross_section_line, = ax_y_profile.plot(intensity_sorted[:, x_idx_initial], azimuthal_adjusted_sorted, color='green')

    x_marker, = ax_x_profile.plot(radial[x_idx_initial], intensity_sorted[y_idx_initial, x_idx_initial], 'ro')
    y_marker, = ax_y_profile.plot(intensity_sorted[y_idx_initial, x_idx_initial], azimuthal_adjusted_sorted[y_idx_initial], 'ro')

    ax_x_profile.set_xlim(radial.min(), radial.max())
    ax_x_profile.set_ylim(0, np.max(intensity_sorted))

    ax_y_profile.set_xlim(0, np.max(intensity_sorted))
    ax_y_profile.set_ylim(azimuthal_adjusted_sorted.min(), azimuthal_adjusted_sorted.max())

    ax_text_box.text(0.1, 0.5, f'X: {radial[x_idx_initial]:.2f}\nY: {azimuthal_adjusted_sorted[y_idx_initial]:.2f}\nIntensity: {intensity_sorted[y_idx_initial, x_idx_initial]:.2f}', fontsize=12, va='center')

    # Sliders for adjusting intensity color limits
    ax_vmin_slider = fig.add_subplot(gs[1,0])
    ax_vmax_slider = fig.add_subplot(gs[2,0])
    vmin_default = 0
    vmax_default = np.mean(intensity_sorted)
    slider_vmin = Slider(ax_vmin_slider, 'Vmin', min(0, np.min(intensity_sorted)), np.max(intensity_sorted), valinit=vmin_default)
    slider_vmax = Slider(ax_vmax_slider, 'Vmax', np.min(intensity_sorted), np.max(intensity_sorted), valinit=vmax_default)

    def update_vmin_vmax(val):
        im.set_clim(slider_vmin.val, slider_vmax.val)
        fig.canvas.draw_idle()

    slider_vmin.on_changed(update_vmin_vmax)
    slider_vmax.on_changed(update_vmin_vmax)

    # Function to update crosshair and profiles
    def update_crosshair(x, y):
        if x < radial.min() or x > radial.max() or y < azimuthal_adjusted_sorted.min() or y > azimuthal_adjusted_sorted.max():
            return
        x_idx = np.argmin(np.abs(radial - x))
        y_idx = np.argmin(np.abs(azimuthal_adjusted_sorted - y))

        intensity_val = intensity_sorted[y_idx, x_idx]

        horizontal_line.set_ydata(y)
        vertical_line.set_xdata(x)

        x_cross_section = intensity_sorted[y_idx, :]
        y_cross_section = intensity_sorted[:, x_idx]

        x_cross_section_line.set_ydata(x_cross_section)
        y_cross_section_line.set_xdata(y_cross_section)

        x_marker.set_data(radial[x_idx], x_cross_section[x_idx])
        y_marker.set_data(y_cross_section[y_idx], azimuthal_adjusted_sorted[y_idx])

        # Update text box
        ax_text_box.clear()
        ax_text_box.axis('off')
        ax_text_box.text(0.1, 0.5, f'X: {radial[x_idx]:.2f}\nY: {azimuthal_adjusted_sorted[y_idx]:.2f}\nIntensity: {intensity_val:.2f}', fontsize=12, va='center')

        # Adjust profiles to visible region after zoom/pan
        adjust_profiles_to_visible()

        fig.canvas.draw_idle()

    # Adjust the profile views to the currently visible region of the image
    def adjust_profiles_to_visible():
        # Current crosshair positions
        cx = vertical_line.get_xdata()[0]
        cy = horizontal_line.get_ydata()[0]

        x_idx = np.argmin(np.abs(radial - cx))
        y_idx = np.argmin(np.abs(azimuthal_adjusted_sorted - cy))

        # Current visible region in the main image
        xlim = ax_image.get_xlim()
        ylim = ax_image.get_ylim()

        # Mask out only the data within the visible region
        x_visible_mask = (radial >= xlim[0]) & (radial <= xlim[1])
        y_visible_mask = (azimuthal_adjusted_sorted >= ylim[0]) & (azimuthal_adjusted_sorted <= ylim[1])

        visible_intensity = intensity_sorted[y_visible_mask, :][:, x_visible_mask]
        if visible_intensity.size > 0:
            visible_max = np.max(visible_intensity)
        else:
            visible_max = np.max(intensity_sorted)  # fallback if no data visible

        # Update x-profile limits
        x_cross_section = intensity_sorted[y_idx, :]
        ax_x_profile.set_ylim(0, visible_max)

        # Update y-profile limits
        y_cross_section = intensity_sorted[:, x_idx]
        ax_y_profile.set_xlim(0, visible_max)

    # Interaction logic for dragging crosshair
    drag_state = {'dragging': False}
    update_interval = 0.03
    last_update_time = time.time()

    def on_press(event):
        if event.inaxes == ax_image and event.button == 1:
            drag_state['dragging'] = True
            on_motion(event)

    def on_release(event):
        if event.button == 1:
            drag_state['dragging'] = False

    def on_motion(event):
        nonlocal last_update_time
        if not drag_state['dragging']:
            return
        current_time = time.time()
        if current_time - last_update_time < update_interval:
            return
        last_update_time = current_time

        if event.inaxes == ax_image and event.xdata is not None and event.ydata is not None:
            update_crosshair(event.xdata, event.ydata)

    # Callbacks for zoom/pan events
    def on_xlim_changed(event_ax):
        adjust_profiles_to_visible()

    def on_ylim_changed(event_ax):
        adjust_profiles_to_visible()

    ax_image.callbacks.connect('xlim_changed', on_xlim_changed)
    ax_image.callbacks.connect('ylim_changed', on_ylim_changed)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show()

if __name__ == "__main__":
    print("Please define ai, res2, and regions_of_interest before calling plot_interactive_2d.")
