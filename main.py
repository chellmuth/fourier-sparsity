import numpy
import numpy.fft

from PIL import Image

def run():
    compression_ratio = 0.95

    image = Image.open("mandrill.tiff").convert("L")
    rows = image.size[0]
    cols = image.size[1]

    data = numpy.array(image.getdata())
    data.resize(image.size)

    image.show("truth")

    transform = numpy.fft.fft2(data)
    coefficients = []
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            coefficients.append((
                transform[i][j].real,
                2 * (i * cols + j)
            ))
            coefficients.append((
                transform[i][j].imag,
                2 * (i * cols + j) + 1
            ))

    coefficients.sort(key=lambda coefficient: abs(coefficient[0]))

    for i in range(int(compression_ratio * (2 * rows * cols))):
        coefficients[i] = (0, coefficients[i][1])

    coefficients.sort(key=lambda coefficient: coefficient[1])

    for i in range(image.size[0]):
        for j in range(image.size[1]):
            real = coefficients[2 * (i * cols + j)][0]
            imag = coefficients[2 * (i * cols + j) + 1][0]
            transform[i][j] = real + imag * 1j

    sparse_data = numpy.fft.ifft2(transform).real
    sparse_image = Image.fromarray(sparse_data.astype("uint8"), "L")

    sparse_image.show("compressed")

if __name__ == "__main__":
    run()
