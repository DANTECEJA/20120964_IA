# Tareas IA

## CEJA GARCIA DANTE DALI

### Tareas de la materia de Inteligencia Artificial

En esta colección de tareas, exploramos diversos aspectos de la inteligencia artificial, desde el reconocimiento de emociones hasta la implementación de redes neuronales convolucionales y el uso de tecnologías avanzadas como Phaser. Cada tarea está diseñada para profundizar en los conocimientos y habilidades necesarios para desarrollar soluciones innovadoras en el campo de la inteligencia artificial.

## Índice

1. [Tarea 1: Detector de emociones](#tarea-1-detector-de-emociones)
2. [Tarea 2: Wally](#tarea-2-wally)
3. [Tarea 3: CNN](#tarea-3-cnn)
4. [Tarea 4: Phaser 1](#tarea-4-phaser-1)
5. [Tarea 5: Phaser 2](#tarea-5-phaser-2)

## Tarea 1: Detector de emociones

Esta tarea consiste en desarrollar un sistema que pueda reconocer las emociones de una persona a partir de imágenes capturadas en tiempo real. El sistema utiliza un clasificador Haar para detectar rostros y un modelo FisherFaceRecognizer para identificar las emociones.

### Código

```python
import numpy as np 
import cv2 as cv

rostro = cv.CascadeClassifier('C:\\Users\\cejad\\haarcascade_frontalface_alt.xml')

cap = cv.VideoCapture(0)
i = 284
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in rostros:
        #frame = cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        frame2 = frame[y:y+h, x:x+w]
        #cv.imshow('rostros2', frame2)
        frame2 = cv.resize(frame2, (100,100), interpolation = cv.INTER_AREA)
        cv.imwrite(f'C:\\Users\\cejad\\datasetemociones\\sorpresa\\sorpresa{i}.png', frame2)

    cv.imshow('rostros', frame)
    i=i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
#aqui crea el xml pa que jale el programa
import cv2 as cv 
import numpy as np 
import os

dataSet = 'C:\\Users\\cejad\\datasetemociones'
faces  = os.listdir(dataSet)
print(faces)

labels = []
facesData = []
label = 0 
for face in faces:
    facePath = dataSet+'\\'+face
    for faceName in os.listdir(facePath):
        labels.append(label)
        facesData.append(cv.imread(facePath+'\\'+faceName,0))
    label = label + 1
print(np.count_nonzero(np.array(labels)==0)) 

faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.train(facesData, np.array(labels))
faceRecognizer.write('emociones2Eigenface.xml')
#progama jalando
import cv2 as cv
import os

faceRecognizer = cv.face.FisherFaceRecognizer_create()
faceRecognizer.read('emociones2Eigenface.xml')

cap = cv.VideoCapture(0)
rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cpGray = gray.copy()
    rostros = rostro.detectMultiScale(gray, 1.3, 3)
    for(x, y, w, h) in rostros:
        frame2 = cpGray[y:y+h, x:x+w]
        frame2 = cv.resize(frame2,  (100,100), interpolation=cv.INTER_CUBIC)
        result = faceRecognizer.predict(frame2)
        cv.putText(frame, '{}'.format(result), (x,y-20), 1,3.3, (0,0,0), 1, cv.LINE_AA)
        if result[1] < 400:
            cv.putText(frame,'{}'.format(faces[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv.LINE_AA)
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
```
## Tarea 2: Wally

Esta tarea consiste en desarrollar un sistema para encontrar a Wally en un escenario dado. Utilizamos técnicas de procesamiento de imágenes para detectar a Wally en una imagen cargada y resaltarlo con un rectángulo.

### Código

```python
#importa el cascade
import numpy as np
import cv2 as cv
import math

# Cargar el clasificador entrenado
wally = cv.CascadeClassifier('cascadeya.xml')
if wally.empty():
    print("Error: no se pudo cargar el clasificador.")
else:
    print("Clasificador cargado correctamente.")
#carga el escenario
frame = cv.imread("C:\\Users\\cejad\\datsetwally\\test.jpg")
if frame is None:
    print("Error: no se pudo cargar la imagen.")
else:
    print("Imagen cargada correctamente.")

gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
wallys = wally.detectMultiScale(gray,1.3,5)

#recorre la imagen buscando a wally
for (x, y, w, h) in wallys:
    frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Mostrar la imagen con los rectángulos
cv.imshow('rostros', frame)
cv.waitKey(0)
cv.destroyAllWindows()

```

## Tarea 3: CNN

En esta tarea, se desarrollará un sistema de reconocimiento capaz de identificar cinco situaciones diferentes a partir de una imagen proporcionada. El sistema utilizará técnicas avanzadas de procesamiento de imágenes y aprendizaje automático para clasificar la imagen en una de las cinco categorías predefinidas.

### Código

```python
#CODIGO PARA SACAR LAS IMAGENES DE LOS VIDEOS
import numpy as np 
import cv2 as cv

rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
#IGNORAR
#saca las imagenes frame por frame

cap = cv.VideoCapture('C:\\Users\\cejad\\deteccion de desastres\\Videos\\robocasa1.mp4')
i = 0
while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (50,50), interpolation = cv.INTER_AREA)
    cv.imwrite('C:\\Users\\cejad\\deteccion de desastres\\5 situaciones\\robocasa\\robocasaimg'+str(i)+'.jpg', frame)
        
    cv.imshow('situacion', frame)
    i=i+1
    k = cv.waitKey(1)
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
#IGNORAR
#escalar las imagenes

def escala(imx, escala):
    width = int(imx.shape[1] * escala / 100)
    height = int(imx.shape[0] * escala / 100)
    size = (width, height)
    im = cv.resize(imx, size, interpolation = cv.INTER_AREA)
    return im
    #importa librerias necesarias
import numpy as np
import os
import re
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#importa librerias necesarias2
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential,Model
from tensorflow.keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D
)
from keras.layers import LeakyReLU
#obtiene todas la img del dataset
dirname = os.path.join(os.getcwd(),'C:\\Users\\cejad\\deteccion de desastres\\5 situaciones')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            if(len(image.shape)==3):
                
                images.append(image)
            b = "Leyendo..." + str(cant)
            print (b, end="\r")
            if prevRoot !=root:
                print(root, cant)
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))
# etiqueta la cantidad total de imagenes
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))
#agarra las situacones en el directorio
situaciones=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    situaciones.append(name[len(name)-1])
    indice=indice+1
    #enumera las situaciones
y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy



# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X/255.
test_X = test_X/255.
plt.imshow(test_X[0,:,:])
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])
#Mezclar todo y crear los grupos de entrenamiento y testing
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
#declaramos variables con los parámetros de configuración de la red
INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
epochs = 20 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria
risk_situation_model = Sequential()
risk_situation_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(50,50,3)))
risk_situation_model.add(LeakyReLU(alpha=0.1))
risk_situation_model.add(MaxPooling2D((2, 2),padding='same'))
risk_situation_model.add(Dropout(0.5))


risk_situation_model.add(Flatten())
risk_situation_model.add(Dense(32, activation='linear'))
risk_situation_model.add(LeakyReLU(alpha=0.1))
risk_situation_model.add(Dropout(0.5))
risk_situation_model.add(Dense(nClasses, activation='softmax'))
risk_situation_model.summary()
risk_situation_model.compile(
    loss=keras.losses.categorical_crossentropy, 
    optimizer=tf.keras.optimizers.SGD(learning_rate=INIT_LR, decay=INIT_LR / 100),
    metrics=['accuracy']
)
# este paso puede tomar varios minutos, dependiendo de tu ordenador, cpu y memoria ram libre
# Entrenamiento del modelo
risk_situation_train = risk_situation_model.fit(
    train_X, 
    train_label, 
    batch_size=batch_size, 
    epochs=epochs, 
    verbose=1, 
    validation_data=(valid_X, valid_label)
)
# guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
risk_situation_model.save("C:\\Users\\cejad\\deteccion de desastres\\risk_situation.h5")
test_eval = risk_situation_model.evaluate(test_X, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
risk_situation_train.history
accuracy = risk_situation_train.history['accuracy']
val_accuracy = risk_situation_train.history['val_accuracy']
loss = risk_situation_train.history['loss']
val_loss = risk_situation_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#que tan apegado esta el entrenamiento a las imagenes
accuracy = risk_situation_train.history['accuracy']
val_accuracy = risk_situation_train.history['val_accuracy']
loss = risk_situation_train.history['loss']
val_loss = risk_situation_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#que tan apegado esta el entrenamiento a las imagenes
predicted_classes.shape, test_Y.shape
predicted_classes.shape, test_Y.shape
incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[0:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(50,50,3), cmap='gray', interpolation='none')
    plt.title("{}, {}".format(situaciones[predicted_classes[incorrect]],
                                                    situaciones[test_Y[incorrect]]))
    plt.tight_layout()
    #pone las incorrectas
    target_names = ["Class {}".format(i) for i in range(nClasses)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
#carga la imagen saca los resultados y abre la img con ellos
from skimage.transform import resize

images=[]
# AQUI ESPECIFICAMOS UNAS IMAGENES
filenames = ['C:\\Users\\cejad\\deteccion de desastres\\incendio3.jpg']

for filepath in filenames:
    image = plt.imread(filepath,0)
    image_resized = resize(image, (50, 50),anti_aliasing=True,clip=False,preserve_range=True)
    images.append(image_resized)

X = np.array(images, dtype=np.uint8) #convierto de lista a numpy
test_X = X.astype('float32')
test_X = test_X / 255.

predicted_classes = risk_situation_model.predict(test_X)

for i, img_tagged in enumerate(predicted_classes):
    print(filenames[i], situaciones[img_tagged.tolist().index(max(img_tagged))])

predicted_classes = np.argmax(predicted_classes, axis=1)

# Mostrar resultados y abrir la imagen con el texto de predicción
for i, img_tagged in enumerate(predicted_classes):
    # Obtener el nombre de la clase predicha
    pred_class = situaciones[img_tagged]

    # Cargar la imagen original en color
    img_color = cv.imread(filepath)
    if img_color is None:
        print(f"Error: no se pudo cargar la imagen {filepath}")
    else:
        # Redimensionar la imagen a su tamaño original
        img_color = cv.resize(img_color, (500, 500))

        # Dibujar el texto de predicción en la imagen
        cv.putText(img_color, f"Prediccion: {pred_class}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar la imagen con el texto de predicción
        cv.imshow(filenames[i], img_color)
        cv.waitKey(0)
        cv.destroyAllWindows()
```

## Tarea 4: Phaser 1

En esta tarea, se implementará un juego utilizando Phaser, un marco de trabajo para la creación de juegos en HTML5. El juego involucra un personaje controlado por el jugador que debe evitar ser golpeado por balas disparadas desde diferentes direcciones. El jugador puede moverse y saltar para esquivar las balas, mientras que las balas se mueven automáticamente hacia el jugador.

### Código

```js
var w = 800;
var h = 400;
var jugador;
var fondo;

var bala,
  balaD = false,
  nave;

var salto;
var menu;

var velocidadBala;
var despBala;

var estatusAire;
var estatuSuelo;

var nnNetwork,
  nnEntrenamiento,
  nnSalida,
  datosEntrenamiento = [];
var modoAuto = false,
  eCompleto = false;

//Nuevas variables
var nnNetwork2,
  nnEntrenamiento2,
  nnSalida2,
  datosEntrenamiento2 = [];

var nnNetwork3,
  nnEntrenamiento3,
  nnSalida3,
  datosEntrenamiento3 = [];

// status
let statusOriginal = 0,
  statusRight = 0;
let statusOriginal2 = 0,
  statusRight2 = 0;
let velocidadBala2, despBala2, bala2, nave2;
let bala2D = false;

let velocidadBala3x, velocidadBala3y, despBala3, bala3, nave3;
let bala3D = false;

let derecha, enter, izquierda;
let dir = false;
let distancia_euclidiana, distancia_euclidiana2;

var juego = new Phaser.Game(w, h, Phaser.CANVAS, "", {
  preload: preload,
  create: create,
  update: update,
  render: render,
});

function preload() {
  juego.load.image("fondo", "assets/game/fondo.jpg");
  juego.load.spritesheet("mono", "assets/sprites/altair.png", 32, 48);
  juego.load.image("nave", "assets/game/ufo.png");
  juego.load.image("bala", "assets/sprites/purple_ball.png");
  juego.load.image("menu", "assets/game/menu.png");

  //segunda bala
  juego.load.image("nave2", "assets/game/ufo.png");
  juego.load.image("bala2", "assets/sprites/purple_ball.png");

  //tercer bala
  juego.load.image("nave3", "assets/game/ufo.png");
  juego.load.image("bala3", "assets/sprites/purple_ball.png");
}

function create() {
  juego.physics.startSystem(Phaser.Physics.ARCADE);
  juego.physics.arcade.gravity.y = 800;
  juego.time.desiredFps = 30;

  fondo = juego.add.tileSprite(0, 0, w, h, "fondo");
  nave = juego.add.sprite(w - 100, h - 70, "nave");
  bala = juego.add.sprite(w - 100, h, "bala");
  jugador = juego.add.sprite(25, h, "mono");

  juego.physics.enable(jugador);
  jugador.body.collideWorldBounds = true;

  juego.physics.enable(bala);
  bala.body.collideWorldBounds = true;

  //bala y nave 2
  nave2 = juego.add.sprite(0, 0, "nave2");
  bala2 = juego.add.sprite(0, 0, "bala2");
  juego.physics.enable(bala2);

  //bala y nave 3
  nave3 = juego.add.sprite(w - 100, 0, "nave3");
  bala3 = juego.add.sprite(w - 75, 0, "bala3");
  juego.physics.enable(bala3);

  //
  pausaL = juego.add.text(w - 100, 20, "Pausa", {
    font: "20px Arial",
    fill: "#fff",
  });
  pausaL.inputEnabled = true;
  pausaL.events.onInputUp.add(pausa, self);
  juego.input.onDown.add(mPausa, self);

  salto = juego.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);

  derecha = juego.input.keyboard.addKey(Phaser.Keyboard.D);
  izquierda = juego.input.keyboard.addKey(Phaser.Keyboard.A);

  enter = juego.input.keyboard.addKey(Phaser.Keyboard.ENTER);

  nnNetwork = new synaptic.Architect.Perceptron(2, 6, 6, 2);
  nnEntrenamiento = new synaptic.Trainer(nnNetwork);

  nnNetwork2 = new synaptic.Architect.Perceptron(4, 6, 6, 6, 2);
  nnEntrenamiento2 = new synaptic.Trainer(nnNetwork2);
}

function enRedNeural() {
  nnEntrenamiento.train(datosEntrenamiento, {
    rate: 0.0003,
    iterations: 10000,
    shuffle: true,
  });
}

const enRedNeuronal2 = () => {
  nnEntrenamiento2.train(datosEntrenamiento2, {
    rate: 0.0003,
    iterations: 10000,
    shuffle: true,
  });
};

function datosDeEntrenamiento(param_entrada) {
  console.log("Entrada 1 ", param_entrada);

  nnSalida = nnNetwork.activate(param_entrada);

  var aire = Math.round(nnSalida[0] * 100);
  var piso = Math.round(nnSalida[1] * 100);

  console.log("Valor aire " + aire + " valor piso " + piso);
  return nnSalida[0] >= nnSalida[1];
}

function datosDeEntrenamiento2(param_entrada) {
  console.log("Entrada 2 ", param_entrada);

  nnSalida2 = nnNetwork2.activate(param_entrada);

  var aire = Math.round(nnSalida2[0] * 100);
  var piso = Math.round(nnSalida2[1] * 100);

  console.log("Valor derecha " + aire + " valor izq " + piso);
  return nnSalida2[0] > nnSalida2[1]
    ? true
    : nnSalida2[0] == nnSalida2[1]
    ? 0
    : false;
}

function pausa() {
  juego.paused = true;
  menu = juego.add.sprite(w / 2, h / 2, "menu");
  menu.anchor.setTo(0.5, 0.5);
}

function mPausa(event) {
  if (juego.paused) {
    var menu_x1 = w / 2 - 270 / 2,
      menu_x2 = w / 2 + 270 / 2,
      menu_y1 = h / 2 - 180 / 2,
      menu_y2 = h / 2 + 180 / 2;

    var mouse_x = event.x,
      mouse_y = event.y;
    //volver a iniciar
    if (
      mouse_x > menu_x1 &&
      mouse_x < menu_x2 &&
      mouse_y > menu_y1 &&
      mouse_y < menu_y2
    ) {
      if (
        mouse_x >= menu_x1 &&
        mouse_x <= menu_x2 &&
        mouse_y >= menu_y1 &&
        mouse_y <= menu_y1 + 90
      ) {
        eCompleto = false;
        datosEntrenamiento = [];
        datosEntrenamiento2 = [];
        datosEntrenamiento3 = [];
        modoAuto = false;
      } else if (
        mouse_x >= menu_x1 &&
        mouse_x <= menu_x2 &&
        mouse_y >= menu_y1 + 90 &&
        mouse_y <= menu_y2
      ) {
        if (!eCompleto) {
          console.log(
            "Entrenamiento " + datosEntrenamiento.length + " valores"
          );
          enRedNeural();
          enRedNeuronal2();
          eCompleto = true;
          console.log("entrenado");
        }
        jugador.position.x = 15;
        modoAuto = true;
      }

      menu.destroy();
      resetVariables();
      resetVariables2();
      juego.paused = false;
    }
  }
}

function resetVariables() {
  jugador.body.velocity.x = 0;
  jugador.body.velocity.y = 0;
  bala.body.velocity.x = 0;
  bala.position.x = w - 100;
  balaD = false;
}

const resetVariables2 = () => {
  //bala2
  bala2.body.velocity.x = 0;
  bala2.position.y = 30;
  bala2D = false;
};

const resetVariables3 = () => {
  //bala3
  bala3.body.velocity.x = 0;
  bala3.body.velocity.y = 0;
  bala3.position.y = 30;
  bala3.position.x = w - 75;
  bala3D = false;
};

function saltar() {
  jugador.body.velocity.y = -270;
}

const avanzar = () => {
  statusOriginal = 0;
  statusRight = 1;
  if (jugador.position.x >= w / 5) return;
  jugador.position.x += 3;
};

const regresar = () => {
  statusOriginal = 1;
  statusRight = 0;

  if (jugador.position.x < 15) return;
  jugador.position.x -= 3;
};

function update() {
  fondo.tilePosition.x -= 1;

  juego.physics.arcade.collide(bala, jugador, colisionH, null, this);
  //collision con bala2
  juego.physics.arcade.collide(bala2, jugador, colisionH, null, this);
  //collision con bala3
  juego.physics.arcade.collide(bala3, jugador, colisionH, null, this);

  estatuSuelo = 1;
  estatusAire = 0;

  if (!jugador.body.onFloor()) {
    estatuSuelo = 0;
    estatusAire = 1;
  }
  despBala = Math.floor(jugador.position.x - bala.position.x);
  //desplazamiento de bala 2
  despBala2 = Math.floor(jugador.position.y - bala2.position.y);
  // distancia eucli bala2
  distancia_euclidiana2 =
    Math.floor(
      Math.sqrt(
        Math.abs(Math.pow(bala2.position.x - jugador.position.x, 2)) -
          Math.abs(Math.pow(bala2.position.y - jugador.position.y, 2))
      )
    ) ||
    Math.floor(
      Math.sqrt(
        Math.abs(
          Math.abs(Math.pow(bala2.position.x - jugador.position.x, 2)) -
            Math.abs(Math.pow(bala2.position.y - jugador.position.y, 2))
        )
      )
    );
  // distancia eucli bala3
  distancia_euclidiana =
    Math.floor(
      Math.sqrt(
        Math.abs(Math.pow(bala3.position.x - jugador.position.x, 2)) -
          Math.abs(Math.pow(bala3.position.y - jugador.position.y, 2))
      )
    ) ||
    Math.floor(
      Math.sqrt(
        Math.abs(
          Math.abs(Math.pow(bala3.position.x - jugador.position.x, 2)) -
            Math.abs(Math.pow(bala3.position.y - jugador.position.y, 2))
        )
      )
    );
  //saltar
  if (modoAuto == false && salto.isDown && jugador.body.onFloor()) {
    saltar();
  }
  //desplzar a derecha
  if (modoAuto == false && derecha.isDown && jugador.body.onFloor()) {
    avanzar();
  }
  //desplzar a izquierda
  if (modoAuto == false && izquierda.isDown && jugador.body.onFloor()) {
    regresar();
  }
  // Modo auto, hay una bala y el jugador está en la superficie
  if (
    modoAuto == true &&
    (bala.position.x > 0 || bala2.position.y < h) &&
    jugador.body.onFloor()
  ) {
    // saltar si el modelo lo indica
    if (datosDeEntrenamiento([despBala, velocidadBala])) {
      saltar();
    }
    // desplazarse a la derecha si el modelo lo indica
    if (
      datosDeEntrenamiento2([
        distancia_euclidiana2,
        dir ? 1 : 0,
        distancia_euclidiana,
        velocidadBala3x,
      ]) != 0
    ) {
      if (
        datosDeEntrenamiento2([
          distancia_euclidiana2,
          dir ? 1 : 0,
          distancia_euclidiana,
          velocidadBala3x,
        ])
      )
        avanzar();
      else regresar();
    }
  }

  if (balaD == false) {
    disparo();
  }

  if (bala2D == false) {
    disparo2();
  }

  if (bala3D == false) {
    disparo3();
  }

  if (bala.position.x <= 0) {
    resetVariables();
  }
  if (bala2.position.y > h) {
    resetVariables2();
  }

  if (bala3.position.y > h || bala3.position.x <= 0) {
    resetVariables3();
  }

  if (
    modoAuto == false &&
    (bala.position.x > 0 ||
      bala2.position.y < h ||
      bala3.position.x > 0 ||
      bala3.position.y < h)
  ) {
    datosEntrenamiento.push({
      input: [despBala, velocidadBala],
      output: [estatusAire, estatuSuelo],
    });

    datosEntrenamiento2.push({
      input: [
        distancia_euclidiana2,
        dir ? 1 : 0,
        distancia_euclidiana,
        velocidadBala3x,
      ],
      output: [statusRight, statusOriginal],
    });
  }
}

function disparo() {
  velocidadBala = -1 * velocidadRandom(400, 600);
  bala.body.velocity.y = 0;
  bala.body.velocity.x = velocidadBala;
  balaD = true;
}

const disparo2 = () => {
  velocidadBala2 = velocidadRandom(5, 10);
  //bala2
  bala2.body.velocity.y = velocidadBala2;
  bala2.body.velocity.x = 0;
  bala2D = true;
};

const disparo3 = () => {
  velocidadBala3x = -1 * velocidadRandom(300, 800);
  velocidadBala3y = velocidadRandom(5, 10);
  //bala3
  bala3.body.velocity.y = velocidadBala3y;
  bala3.body.velocity.x = velocidadBala3x;
  bala3D = true;
};

function colisionH() {
  pausa();
}

function velocidadRandom(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render() {}
```

## Tarea 5: Phaser 2

El segundo juego también utiliza Phaser y Synaptic.js para implementar un juego con mecánicas de salto y esquivar balas, donde también se aplica el aprendizaje automático para controlar el movimiento del jugador, esta ves con un movimiento libre por el mapa tanto de jugador como de la bala.

### Código

```js
var w = 500;  
var h = 500;  
var jugador; 
var fondo;  

var bala,  
  balaD = false; 

var salto;  
var menu;  

var velocidadBala;  
var despBala;  

var estatusAire; 
var estatuSuelo;  
var cuadrante;  // Cuadrante donde se encuentra el jugador

// Variables para la red neuronal y su entrenamiento
var nnNetwork,  
  nnEntrenamiento,  // Entrenador de la red neuronal principal
  nnSalida,  // Salida de la red neuronal principal
  datosEntrenamiento = [];
  
var modoAuto = false,  
  eCompleto = false; 

// Variables para el estado del juego y balas adicionales
let statusOriginal = 0,  
  statusRight = 0,  
  nul = 1,  // Estado nulo
  nulx = 1,  // Estado nulo en x
  nuly = 1;  // Estado nulo en y


let derecha,  
  enter,  
  izquierda;
let distancia_euclidiana;  // Distancia euclidiana entre la bala y el jugador

var juego = new Phaser.Game(w, h, Phaser.CANVAS, "", {
  preload: preload,  
  create: create, 
  update: update,  
  render: render,  
});

function preload() {
  juego.load.image('fondo', 'assets/game/descarga.jpg');  
  juego.load.spritesheet("mono", "assets/sprites/altair.png", 32, 48);  
  juego.load.image("menu", "assets/game/menu.png"); 
  juego.load.image('bala', 'assets/sprites/bola.png');  
}

function create() {
  juego.physics.startSystem(Phaser.Physics.ARCADE); 

  juego.time.desiredFps = 30;  

  fondo = juego.add.tileSprite(0, 0, w, h, "fondo");  
  bala = juego.add.sprite(w - 100, h, "bala");  
  jugador = juego.add.sprite(w / 2, h / 2, "mono");  

  juego.physics.enable(jugador);  
  jugador.body.collideWorldBounds = true;  
  var corre = jugador.animations.add("corre", [8, 9, 10, 11]);  
  jugador.animations.play("corre", 10, true);  

  juego.physics.enable(bala); 
  bala.body.collideWorldBounds = true;  
  bala.body.bounce.set(1);

  pausaL = juego.add.text(w - 100, 20, "Pausa", {
    font: "20px Arial",
    fill: "#fff",
  });  

  pausaL.inputEnabled = true; 
  pausaL.events.onInputUp.add(pausa, self);  
  juego.input.onDown.add(mPausa, self);  

  salto = juego.input.keyboard.addKey(Phaser.Keyboard.UP);  

  derecha = juego.input.keyboard.addKey(Phaser.Keyboard.RIGHT);  
  izquierda = juego.input.keyboard.addKey(Phaser.Keyboard.LEFT);  
  abajo = juego.input.keyboard.addKey(Phaser.Keyboard.DOWN);  

  enter = juego.input.keyboard.addKey(Phaser.Keyboard.ENTER);  

  // Inicializa la red neuronal con 2 entradas, 6 neuronas en dos capas ocultas y 4 salidas
  nnNetwork = new synaptic.Architect.Perceptron(2, 6, 6, 4);
  nnEntrenamiento = new synaptic.Trainer(nnNetwork);  
}

function enRedNeural() {
  nnEntrenamiento.train(datosEntrenamiento, {
    rate: 0.0003,  // Tasa de aprendizaje
    iterations: 10000,  // Número de iteraciones
    shuffle: true,  // Barajar los datos en cada iteración
  });
}

// Función para obtener la salida de la red neuronal
function datosDeEntrenamiento(param_entrada) {
  console.log("Entrada 1 ", param_entrada);  

  nnSalida = nnNetwork.activate(param_entrada);  // Activa la red neuronal con los parámetros de entrada

  var aire = Math.round(nnSalida[0] * 100);  // Calcula el valor de aire a partir de la salida de la red neuronal
  var piso = Math.round(nnSalida[1] * 100);  // Calcula el valor de piso a partir de la salida de la red neuronal

  console.log("Valor distancia eu" + aire + " valor cuadrante " + piso);  // Imprime los valores de aire y piso
  return nnSalida;  // Devuelve la salida de la red neuronal
}

function pausa() {
  juego.paused = true;
  menu = juego.add.sprite(w / 2, h / 2, "menu");  
  menu.anchor.setTo(0.5, 0.5); 
}

function mPausa(event) {
  if (juego.paused) {  
    var menu_x1 = w / 2 - 270 / 2,
      menu_x2 = w / 2 + 270 / 2,
      menu_y1 = h / 2 - 180 / 2,
      menu_y2 = h / 2 + 180 / 2;

    var mouse_x = event.x,  
      mouse_y = event.y;  

    if (
      mouse_x > menu_x1 &&
      mouse_x < menu_x2 &&
      mouse_y > menu_y1 &&
      mouse_y < menu_y2
    ) {
      if (
        mouse_x >= menu_x1 &&
        mouse_x <= menu_x2 &&
        mouse_y >= menu_y1 &&
        mouse_y <= menu_y1 + 90
      ) {
        eCompleto = false;  // Reinicia el estado de entrenamiento
        datosEntrenamiento = [];  // Vacía los datos de entrenamiento
        modoAuto = false;  // Desactiva el modo automático
      } else if (
        mouse_x >= menu_x1 &&
        mouse_x <= menu_x2 &&
        mouse_y >= menu_y1 + 90 &&
        mouse_y <= menu_y2
      ) {
        if (!eCompleto) {  // Si el entrenamiento no está completo
          console.log(
            "Entrenamiento " + datosEntrenamiento.length + " valores"
          );
          enRedNeural();  // Entrena la red neuronal
          eCompleto = true;  // Indica que el entrenamiento está completo
          console.log("entrenado");
        }
        jugador.position.x = w / 2;  // Reposiciona el jugador en el centro
        jugador.position.y = h / 2;
        modoAuto = true;  // Activa el modo automático
      }

      menu.destroy();  
      resetVariables();  
      juego.paused = false;  
    }
  }
}


function resetVariables() {
  jugador.body.velocity.x = 0;  
  jugador.body.velocity.y = 0;  

  bala.body.velocity.x = 0;  
  bala.position.x = w - 100;  

  balaD = false;  
}


function saltar() {
  if (modoAuto) if (jugador.position.y <= h / 4) return;  // Si está en modo automático y el jugador está demasiado alto, no saltar
  jugador.position.y -= 15;  // Disminuye la posición y del jugador (simulando un salto)
}

// Función para hacer bajar al jugador
function bajar() {
  if (modoAuto) if (jugador.position.y >= h * (3 / 4)) return;  // Si está en modo automático y el jugador está demasiado bajo, no bajar
  jugador.position.y += 15;  // Aumenta la posición y del jugador (simulando una bajada)
}

// Función para mover al jugador hacia la derecha
const avanzar = () => {
  if (modoAuto) if (jugador.position.x >= w * (4 / 5)) return;  // Si está en modo automático y el jugador está demasiado a la derecha, no avanzar
  if (jugador.position.x >= w) return;  // Si el jugador está en el borde derecho, no avanzar
  jugador.position.x += 15;  // Aumenta la posición x del jugador (simulando un movimiento a la derecha)
};

// Función para mover al jugador hacia la izquierda
const regresar = () => {
  if (modoAuto) if (jugador.position.x <= w / 4) return;  // Si está en modo automático y el jugador está demasiado a la izquierda, no regresar
  if (jugador.position.x < 15) return;  // Si el jugador está en el borde izquierdo, no regresar
  jugador.position.x -= 15;  // Disminuye la posición x del jugador (simulando un movimiento a la izquierda)
};

function update() {
  juego.physics.arcade.collide(bala, jugador, colisionH, null, this);  

  // Cálculo de la distancia euclidiana entre la bala y el jugador
  distancia_euclidiana =
    Math.floor(
      Math.sqrt(
        Math.abs(Math.pow(bala.position.x - jugador.position.x, 2)) -
          Math.abs(Math.pow(bala.position.y - jugador.position.y, 2))
      )
    ) ||
    Math.floor(
      Math.sqrt(
        Math.abs(
          Math.abs(Math.pow(bala.position.x - jugador.position.x, 2)) -
            Math.abs(Math.pow(bala.position.y - jugador.position.y, 2))
        )
      )
    );

  let x = w / 2;  // Coordenada x del centro
  let y = h / 2;  // Coordenada y del centro

  // Determinar el cuadrante en el que se encuentra el jugador
  if (jugador.position.y < y && jugador.position.x > x) {
    cuadrante = 1;  // Cuadrante superior derecho
    estatuSuelo = 0;  // Jugador no está en el suelo
    estatusAire = 1;  // Jugador está en el aire
    statusRight = 1;  // Jugador está a la derecha del centro
    statusOriginal = 0;  // Jugador no está en la posición original
  }
  if (bala.position.y < y && bala.position.x < x) {
    cuadrante = 2;  // Cuadrante superior izquierdo
    estatuSuelo = 0;  // Jugador no está en el suelo
    estatusAire = 1;  // Jugador está en el aire
    statusRight = 0;  // Jugador está a la izquierda del centro
    statusOriginal = 1;  // Jugador está en la posición original
  }
  if (bala.position.y > y && bala.position.x > x) {
    cuadrante = 3;  // Cuadrante inferior derecho
    estatuSuelo = 1;  // Jugador está en el suelo
    estatusAire = 0;  // Jugador no está en el aire
    statusRight = 1;  // Jugador está a la derecha del centro
    statusOriginal = 0;  // Jugador no está en la posición original
  }
  if (bala.position.y > y && bala.position.x < x) {
    cuadrante = 4;  // Cuadrante inferior izquierdo
    estatuSuelo = 1;  // Jugador está en el suelo
    estatusAire = 0;  // Jugador no está en el aire
    statusRight = 0;  // Jugador está a la izquierda del centro
    statusOriginal = 1;  // Jugador está en la posición original
  }

  // Manejar los controles de salto, bajar, avanzar y regresar en modo manual
  if (modoAuto == false && salto.isDown) {
    saltar();  // Llama a la función saltar si la tecla de salto está presionada
    nul = 0;  // Resetear el estado nulo
    nuly = 0;  // Resetear el estado nulo en y
  }

  if (modoAuto == false && abajo.isDown) {
    bajar();  // Llama a la función bajar si la tecla de abajo está presionada
    nul = 0;  // Resetear el estado nulo
    nuly = 0;  // Resetear el estado nulo en y
  }

  if (modoAuto == false && derecha.isDown) {
    avanzar();  // Llama a la función avanzar si la tecla de derecha está presionada
    nul = 0;  // Resetear el estado nulo
    nulx = 0;  // Resetear el estado nulo en x
  }

  if (modoAuto == false && izquierda.isDown) {
    regresar();  // Llama a la función regresar si la tecla de izquierda está presionada
    nul = 0;  // Resetear el estado nulo
    nulx = 0;  // Resetear el estado nulo en x
  }

  // Modo automático para mover el jugador basado en la red neuronal
  if (modoAuto) {
    if (nul == 0) {  // Si el estado nulo es 0
      if (nuly != 1) {  // Si el estado nulo en y no es 1
        if (
          datosDeEntrenamiento([distancia_euclidiana, cuadrante])[0] >=
          datosDeEntrenamiento([distancia_euclidiana, cuadrante])[1]
        ) {
          saltar();  // Salta si el primer valor de la salida de la red neuronal es mayor o igual al segundo
        } else {
          bajar();  // Baja si no
        }
      }

      if (nulx == 0) {  // Si el estado nulo en x es 0
        if (
          datosDeEntrenamiento([distancia_euclidiana, cuadrante])[2] >=
          datosDeEntrenamiento([distancia_euclidiana, cuadrante])[3]
        ) {
          regresar();  // Regresa si el tercer valor de la salida de la red neuronal es mayor o igual al cuarto
        } else {
          avanzar();  // Avanza si no
        }
      }
    }
  }

  if (balaD == false) {
    disparo();
  }

  // Recolectar datos de entrenamiento mientras se juega en modo manual
  if (modoAuto == false) {
    datosEntrenamiento.push({
      input: [distancia_euclidiana, cuadrante],  
      output: [estatusAire, estatuSuelo, statusOriginal, statusRight],
    });
  }
}

function disparo() {
  velocidadBala = velocidadRandom(400, 600);  
  bala.body.velocity.y = velocidadRandom(400, 600);  
  bala.body.velocity.x = velocidadBala; 
  balaD = true;  
}


function colisionH() {
  pausa();
}


function velocidadRandom(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render() {}
```
[Volver al inicio](#Índice)
