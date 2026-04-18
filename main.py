import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def Prewitt_kernel():

    Prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)

    Prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    return Prewitt_x, Prewitt_y


def apply_filter(img, kernel):

    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

    result = np.zeros_like(img, dtype=np.float32)
    for i in range(k_h):
        for j in range(k_w):
            result += (
                img_padded[i : i + img.shape[0], j : j + img.shape[1], :] * kernel[i, j]
            )

    return result


def Prewitt_outline(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    Prewitt_x, Prewitt_y = Prewitt_kernel()

    Gx = apply_filter(gray[..., np.newaxis], Prewitt_x)[..., 0]
    Gy = apply_filter(gray[..., np.newaxis], Prewitt_y)[..., 0]

    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = np.clip(magnitude, 0, 255)

    outline = np.stack([magnitude, magnitude, magnitude], axis=2).astype(np.uint8)
    return outline


def Canny_outline(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray, threshold1=40, threshold2=120, apertureSize=3, L2gradient=False
    )

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def output_images(
    image1, image2, image3, title1="Image 1", title2="Image 2", title3="Image 3"
):

    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)

    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for idx, (img_rgb, orig) in enumerate(
        [
            (img1_rgb, image1),
            (img2_rgb, image2),
            (img3_rgb, image3),
        ]
    ):
        prewitt = Prewitt_outline(orig)
        canny = Canny_outline(orig)

        prewitt_rgb = cv2.cvtColor(prewitt, cv2.COLOR_BGR2RGB)
        canny_rgb = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)

        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title("Original")

        axes[idx, 1].imshow(prewitt_rgb)
        axes[idx, 1].set_title("Prewitt")

        axes[idx, 2].imshow(canny_rgb)
        axes[idx, 2].set_title("Canny")

    for ax in axes.flat:
        ax.axis("off")

    plt.savefig("outline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():

    path1 = "image1.jpg"
    path2 = "image2.jpg"
    path3 = "image3.jpeg"

    img1 = load_image(path1)
    img2 = load_image(path2)
    img3 = load_image(path3)

    output_images(img1, img2, img3)


if __name__ == "__main__":
    main()
