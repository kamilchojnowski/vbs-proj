import numpy
import os

from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.layers import CuDNNLSTM, Input, Bidirectional, Dense, Layer
from keras.models import Model


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


def lstm_model(input_shape):
    input_instance = Input(shape=(input_shape[0], input_shape[1],))

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_instance)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(input_shape[0])(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(3, activation="sigmoid")(x)

    model = Model(inputs=input_instance, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])

    return model

def transform_to_ones(ts, min_data=-128, max_data=127, range_needed=(-1,1)):
    ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]

def transform_to_standard(ts, n_dim=160, min_max=(-1, 1)):
    sample_size = 800000
    ts_std = transform_to_ones(ts)
    bucket_size = int(sample_size / n_dim)
    new_ts = []
    for i in range(0, sample_size, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std

        percentil_calc = numpy.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]
        relative_percentile = percentil_calc - mean

        features = [
            numpy.asarray([mean, std, std_top, std_bot, max_range]),
            percentil_calc,
            relative_percentile
        ]
        new_ts.append(numpy.concatenate(features))

    return numpy.asarray(new_ts)

def get_data_to_predict(data):
    X = []
    for index in data.columns:
        X_signal = []
        for phase in range(3):
            X_signal.append(transform_to_standard(data[str(index)]))
        X_signal = numpy.concatenate(X_signal, axis=1)
        X.append(X_signal)
    return numpy.asarray(X)


def labelize(prediction, settings):
    replace = settings.REPLACE_MESSAGE
    no_replace = settings.NO_REPLACE_MESSAGE
    to_replace = prediction > settings.MODEL_THRESHOLD
    return {
        "state": replace if to_replace else no_replace,
        "confidence": prediction if to_replace else 1 - prediction
    }


def get_labels(predictions, data, settings):
    labels = list()
    pre_label = dict(zip(data.columns, predictions))
    for obj in pre_label.items():
        labels.append(
            {
                "id": obj[0],
                "phase 0": labelize(obj[1][0], settings),
                "phase 1": labelize(obj[1][1], settings),
                "phase 2": labelize(obj[1][2], settings)
            }
        )
    return labels
