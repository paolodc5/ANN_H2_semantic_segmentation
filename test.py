import numpy as np
import cv2
import scipy.ndimage as nd

def calculate_gvf(image, mu=0.2, iterations=80):
    """
    Calcola il Gradient Vector Flow (GVF) di un'immagine in scala di grigi.
    
    :param image: Array numpy rappresentante l'immagine in scala di grigi (64x128).
    :param mu: Coefficiente di regolarizzazione del GVF.
    :param iterations: Numero di iterazioni per la diffusione.
    :return: Campo vettoriale GVF (u, v).
    """
    # Calcolo dei gradienti di intensità
    gradient_x = nd.sobel(image, axis=1)
    gradient_y = nd.sobel(image, axis=0)
    
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2) + 1e-10  # Evita divisioni per zero
    
    f_x = gradient_x / magnitude
    f_y = gradient_y / magnitude
    
    # Inizializzazione del campo vettoriale
    u = f_x
    v = f_y
    
    for _ in range(iterations):
        # Laplaciani di u e v
        u_lap = nd.laplace(u)
        v_lap = nd.laplace(v)
        
        # Aggiornamento di u e v
        u = u + mu * u_lap - (u - f_x) * (f_x**2 + f_y**2)
        v = v + mu * v_lap - (v - f_y) * (f_x**2 + f_y**2)
    
    return u, v

# Test su un'immagine fittizia
if __name__ == "__main__":
    # Crea un'immagine sintetica (esempio: rettangolo bianco su sfondo nero)
    image = np.zeros((64, 128), dtype=np.float32)
    cv2.rectangle(image, (32, 16), (96, 48), 255, -1)
    image = image / 255.0  # Normalizza a [0, 1]
    
    u, v = calculate_gvf(image)
    
    # Visualizza il risultato
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap="gray")
    plt.quiver(u, v)
    plt.title("Gradient Vector Flow (GVF)")
    plt.show()
