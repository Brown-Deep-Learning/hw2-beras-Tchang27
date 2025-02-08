from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import RMSProp, Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
            Dense(784, 32, initializer="xavier"),
            ReLU(),
            Dense(32, 10, initializer="xavier"),
            Softmax(),
        ]
    )
    return model

def get_optimizer():
    # choose an optimizer, initialize it and return it!
    opt = Adam(0.001)
    return opt

def get_loss_fn():
    # choose a loss function, initialize it and return it!
    loss = CategoricalCrossEntropy()
    return loss

def get_acc_fn():
    # choose an accuracy metric, initialize it and return it!
    acc = CategoricalAccuracy()
    return acc

if __name__ == '__main__':

    ### Use this area to test your implementation!

    # 1. Create a SequentialModel using get_model
    print("creating model!")
    model = get_model()

    # 2. Compile the model with optimizer, loss function, and accuracy metric
    model.compile(
        optimizer=get_optimizer(),
        loss_fn=get_loss_fn(),
        acc_fn=get_acc_fn(),
    )
    epoch = 1
    batch_size = 256
    
    # 3. Load and preprocess the data
    print("loading data!")
    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()
    ohe = OneHotEncoder()
    concat_labels = np.concatenate([train_labels, test_labels], axis=-1)
    ohe.fit(concat_labels)
    
    # 4. Train the model
    print("training model!")
    train_agg_metrics = model.fit(
        train_inputs,
        ohe(train_labels),
        epochs=epoch,
        batch_size=batch_size
    )

    # 5. Evaluate the model
    test_agg_metrics = model.evaluate(test_inputs, ohe(test_labels), batch_size=256)
    print("Testing Performance:", test_agg_metrics)
    