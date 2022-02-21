from PIL import Image
import keras
import os, glob
import numpy as np
import random, math
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

#画像が保存されているルートディレクトリのパス
root_dir = "learning data"
# 商品名
categories = ["キャリーケースあり","キャリーケースなし"]


X = [] # 画像データ用配列
Y = [] # ラベルデータ用配列


#画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)
    
#渡された画像データを読み込んでXに格納し、また、
#画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)
    
#全データ格納用配列    
allfiles = []

#カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))
        

#シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test  = allfiles[th:]
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)

#データを保存する（データの名前を「carry_data.npy」としている）
np.save("carrycase_data", xy)


#モデルの構築

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.25))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(2,activation="sigmoid")) #分類先の種類分設定

#モデル構成の確認
model.summary()


#モデルのコンパイル


model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=["acc"])
             
              
#データの準備


categories = ["キャリーケースあり","キャリーケースなし"]
nb_classes = len(categories)

X_train, X_test, y_train, y_test = np.load("carrycase_data.npy",allow_pickle=True)

#データの正規化
X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float")  / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)


#モデルの学習

model = model.fit(X_train,
                  y_train,
                  epochs=75,
                  batch_size=32,
                  validation_data=(X_test,y_test))
                  
                  
#学習結果を表示


acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Training_and_validation_accuracy_data')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Training_and_validation_loss_data')


#モデルの保存

json_string = model.model.to_json()
open('carrycase_data_predict.json', 'w').write(json_string)

#重みの保存

model.model.save_weights("carryrcase_data_predict.hdf5")


# モデルの精度を測る

#評価用のデータの読み込み
test_X = np.load("carrycase_data_test_X_150.npy")
test_Y = np.load("carrycase_data_test_Y_150.npy")

#Yのデータをone-hotに変換

test_Y = np_utils.to_categorical(test_Y, 2)

score = model.model.evaluate(x=test_X,y=test_Y)

print('loss=', score[0])
print('accuracy=', score[1])

