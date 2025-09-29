import matplotlib.pyplot as plt

def plot_detection_result(image, prediction):
    """ plot image with detection result """
    plt.imshow(image)
    plt.title(f'Detection Score: {prediction:.2f}')
    plt.axis('off')
    plt.show