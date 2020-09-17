def training_model():
  model = Sequential()
  model.add(Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=(50,50,1)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=2))

  model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same', kernel_regularizer=keras.regularizers.l2(l2=0.01)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=2))
  model.add(Dropout(0.5))

  model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',kernel_regularizer=keras.regularizers.l2(l2=0.01)))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2),padding='same',strides=2))
  model.add(Dropout(0.3))

  model.add(Flatten())
  model.add(Dense(units=256,activation='relu'))
  model.add(Dropout(0.5))

  model.add(Dense(units=37,activation='softmax'))

  model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
  lr_red=ReduceLROnPlateau(patience=3,verbose=1,min_lr=0.0001)

  history = model.fit(X_train, Y_train, batch_size=128, epochs=10, callbacks=[lr_red], validation_data=(X_val, Y_val),)
  return model

model=training_model()


def eval_model(model):
  print("Evaluate on test data")
  results = model.evaluate(X_test, Y_test, batch_size=128)
  print("test loss, test acc:", results)

eval_model(model)

model.save("SL_model.h5")
