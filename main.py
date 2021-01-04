# preso da instagram, non funziona
import sys
import cv2
import os

def recognition():
    # importo i file
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    imagePath = ROOT_DIR + '/conte.jpg'
    cascPath = ROOT_DIR + '/haarcascade_frontalface_default.xml'

    # creo la cascata e la inizializzo con il file xml cascata (file xml per rilevare i volti)
    # carica in memoria la cascata del viso
    faceCascade = cv2.CascadeClassifier(cascPath)

    # leggo l'immagine
    image = cv2.imread(imagePath)
    #la converto in scale di grigi (la maggior parte delle operazioni in open cv vengono fatte cosi)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # rileva il volto
    # detectMultiScale è una funzione generale che rileva gli oggetti, dal momento che lo richiamo
    # a cascata facciale, è quella che rileva
    faces = faceCascade.detectMultiScale(
        gray,               # opzione1: immagine in scala di grigi
        scaleFactor=1.1,    # opzione2: fattore di compensazione per volti vicino alla fotocamera
        minNeighbors=5,     # opzione3: definisce quanti oggetti vengono rilevati vicino a quello corrente prima
                            #           che dichiari il volto trovato
        minSize=(30, 30)    # opzione4: fornisce la dimensione di ciascune finestra
    )
    # La funzione restituisce un elenco di rettangoli in cui ritiene di aver trovato una faccia.

    # Nota: ho preso i valori comunemente usati per questi campi. Nella vita reale,
    # sperimenteresti valori diversi per le dimensioni della finestra,
    # il fattore di scala e così via finché non ne trovi uno che funziona meglio per te.

    print("Found {0} faces!".format(len(faces)))

    # Successivamente, eseguiremo un loop su dove pensa di aver trovato qualcosa.
    # Questa funzione restituisce 4 valori:
    # la posizione xe y del rettangolo e la larghezza e l'altezza del rettangolo ( w, h).
    for (x, y, w, h) in faces:
        #Usiamo questi valori per disegnare un rettangolo utilizzando la rectangle() funzione incorporata.
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Alla fine, visualizziamo l'immagine e aspettiamo che l'utente prema un tasto.
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

# MAIN
recognition()
