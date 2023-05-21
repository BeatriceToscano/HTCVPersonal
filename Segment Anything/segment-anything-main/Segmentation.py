import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from segment_anything import SamPredictor, sam_model_registry
if __name__=="__main__":
    # Sostituisci "<model_type>" con il tipo di modello che hai scaricato ("default", "vit_l", o "vit_b")
    # e "<path/to/checkpoint>" con il percorso dove hai salvato il checkpoint del modello scaricato.
    sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)

    # Sostituisci "<your_image>" con l'immagine su cui vuoi fare la segmentazione.
    # L'immagine può essere un array NumPy oppure un percorso al file dell'immagine.
    predictor.set_image(np.array(Image.open(r"C:\Users\tosca\Desktop\Ballet-Ballerina-1843.jpg")))
    masks, _, _ = predictor.predict()
    masks = np.transpose(masks,  (2,1,0))
    print(masks.shape)
    plt.imshow(masks)
    plt.show()
