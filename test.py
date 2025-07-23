import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
from matplotlib.animation import FuncAnimation

# Initialize sliding window
window_size = 15
data_window = deque(maxlen=window_size)

# Initialize with some random values
current_value = 0.0
for _ in range(window_size):
    current_value += np.random.normal(0, 0.1)
    current_value = np.clip(current_value, -0.9, 0.9)
    data_window.append(current_value)

# Trend variables
trend_direction = np.random.choice([-1, 1])
trend_strength = 0.02
trend_duration = np.random.randint(5, 15)
trend_counter = 0


def generate_new_value():
    """Generate a new value that follows trends"""
    global current_value, trend_direction, trend_strength, trend_duration, trend_counter

    # Check if we need a new trend
    trend_counter += 1
    if trend_counter >= trend_duration:
        trend_direction = np.random.choice([-1, 1])
        trend_strength = np.random.uniform(0.01, 0.04)
        trend_duration = np.random.randint(5, 15)
        trend_counter = 0

    # Generate new value with trend and noise
    trend = trend_direction * trend_strength
    noise = np.random.normal(0, 0.05)
    current_value += trend + noise

    # Keep value in bounds with soft clamping
    if current_value > 0.8:
        current_value = 0.8 + (current_value - 0.8) * 0.1
    elif current_value < -0.8:
        current_value = -0.8 + (current_value + 0.8) * 0.1

    current_value = np.clip(current_value, -0.95, 0.95)
    return current_value


def create_plot():
    """Create the styled matplotlib plot"""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Set up the plot
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, window_size - 1)

    # Remove x-axis labels
    ax.set_xticks([])

    # Style the plot
    ax.set_facecolor("#f0f0f0")
    fig.patch.set_facecolor("white")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add zero line
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)

    return fig, ax


def update_plot(frame, ax, fig):
    """Update function for animation"""
    # Generate new value
    new_value = generate_new_value()
    data_window.append(new_value)

    # Clear previous plot
    ax.clear()

    # Reset plot settings
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, window_size - 1)
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)

    # Get data
    x = np.arange(len(data_window))
    y = np.array(data_window)

    # Plot the line
    ax.plot(x, y, color="black", linewidth=2, zorder=3)

    # Fill area under curve with proper color separation
    # Use where parameter to fill only positive or negative regions
    ax.fill_between(x, y, 0, where=(y >= 0), color="green", alpha=0.3, interpolate=True)
    ax.fill_between(x, y, 0, where=(y <= 0), color="red", alpha=0.3, interpolate=True)

    ax.set_ylabel("", fontsize=12)
    ax.set_title("Valence", fontsize=14, pad=20)


def plot_to_image(fig):
    """Convert matplotlib figure to numpy array for cv2"""
    # Draw the figure
    fig.canvas.draw()

    # Convert to numpy array
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Convert RGB to BGR for cv2
    buf = buf[:, :, ::-1]

    return buf


# For static testing
if __name__ == "__main__":
    # Create figure
    fig, ax = create_plot()

    # Option 1: Animated plot (for testing the look)
    ani = FuncAnimation(
        fig,
        lambda frame: update_plot(frame, ax, fig),  # type: ignore
        interval=1000,
        cache_frame_data=False,
    )
    plt.show()

    # Option 2: Single frame function (for cv2 integration)
    def generate_frame():
        """Generate a single frame for cv2 integration"""
        fig, ax = create_plot()

        # Generate new value
        new_value = generate_new_value()
        data_window.append(new_value)

        # Update plot
        update_plot(0, ax, fig)

        # Convert to image
        img = plot_to_image(fig)

        # Clean up
        plt.close(fig)

        return img

    # Example usage for cv2:
    # frame = generate_frame()
    # cv2.imshow('Plot', frame)
