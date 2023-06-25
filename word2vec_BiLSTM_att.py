# -*- coding:utf-8 -*-
import os
import warnings
import datetime
import tensorflow as tf
import traceback
import time
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from task_01.bert_bilstm_config import BertBilstmConfig
from evaluate import binary_recall, binary_f_beta, binary_precision, binary_accuracy
import gensim
# wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/smcontractt/word2Vec.bin", binary=True)

warnings.filterwarnings("ignore")

def load_sample(fn, max_seq_len, word_dict):
    raw_y = []
    raw_x = []
    with open(fn, "r") as file:
        text_dfs = json.load(file)

        for text_df in text_dfs:
            label = text_df["lable"]

            label = int(label)
            raw_y.append(int(label))
            text = text_df["contractCode"]
            text = str(text).strip().split()

            text_len = len(text)
            x = np.zeros(max_seq_len, dtype=np.int32)
            if text_len <= max_seq_len:
                for i in range(text_len):
                    x[i] = word_dict[text[i]]
            else:
                for i in range(max_seq_len):
                    x[i] = word_dict[text[i]]
            raw_x.append(x)

        data_len = len(raw_y)
        raw_x = np.array(raw_x)
        raw_y = np.array(raw_y)

        indices = np.random.permutation(np.arange(data_len))
        all_x = raw_x[indices]
        all_y = raw_y[indices]

        print(data_len)
        file.close()

        train_all_x = all_x[0: int(data_len*0.8)]
        train_all_y = all_y[0: int(data_len*0.8)]

        test_all_x = all_x[int(data_len*0.8): data_len]
        test_all_y = all_y[int(data_len*0.8): data_len]

        return train_all_x, train_all_y, test_all_x, test_all_y



def load_sample_ori(fn, max_seq_len, word_dict):
    raw_y = []
    raw_x = []
    with open(fn, "r") as file:
        text_dfs = json.load(file)

        for text_df in text_dfs:
            label = text_df["lable"]
            raw_y.append(int(label))
            text = text_df["contractCode"]
            text = str(text).strip().split()

            text_len = len(text)
            x = np.zeros(max_seq_len, dtype=np.int32)
            if text_len <= max_seq_len:
                for i in range(text_len):
                    x[i] = word_dict[text[i]]
            else:
                for i in range(max_seq_len):
                    x[i] = word_dict[text[i]]
            raw_x.append(x)

        all_x = np.array(raw_x)
        all_y = np.array(raw_y)

        print(len(all_y))

        file.close()
        return all_x, all_y


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = (data_len) // batch_size
    indices = np.random.permutation(np.arange(data_len))
    x_shuff = x[indices]
    y_shuff = y[indices]

    for i in range(num_batch):
        start_offset = i * batch_size
        end_offset = min(start_offset + batch_size, data_len)
        yield i, num_batch, x_shuff[start_offset:end_offset], y_shuff[start_offset:end_offset]


######################### model #####################
class RnnLayer(layers.Layer):
    def __init__(self, rnn_size, drop_rate):
        super().__init__()
        fwd_lstm = LSTM(rnn_size, return_sequences=True, go_backwards=False, dropout=drop_rate, name="fwd_lstm")
        bwd_lstm = LSTM(rnn_size, return_sequences=True, go_backwards=True, dropout=drop_rate, name="bwd_lstm")
        self.bilstm = Bidirectional(merge_mode="sum", layer=fwd_lstm, backward_layer=bwd_lstm, name="bilstm")

    def call(self, inputs, training):
        outputs = self.bilstm(inputs, training=training)
        outputs = tf.concat([outputs[:, 0, :], outputs[:, -1, :]], 1)
        return outputs


def create_embeddings_matrix(word2vec, vocab_size, vocab_list, word_dict):
    embeddings_matrix = np.zeros((vocab_size, 200))

    for i in range(len(vocab_list)):
        word = vocab_list[i] 
        word_index = word_dict[word]
        embeddings_matrix[word_index] = word2vec[word]  
        return embeddings_matrix


class Model(tf.keras.Model):
    def __init__(self, num_classes, drop_rate, vocab_size, embedding_size, rnn_size, attention_size, embeddings_matrix):
        super().__init__()

        self.embedding_layer = Embedding(
            input_dim=vocab_size,  
            output_dim=embedding_size,  
            weights=[embeddings_matrix],  
            input_length=max_seq_len,  
            trainable=True,  
            )

        self.rnn_layer = RnnLayer(rnn_size, drop_rate)

        self.attention_layer = RnnAttentionLayer(attention_size, drop_rate)

        self.dense_layer = Dense(num_classes, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.001),
                                 name="dense_1")

    def call(self, input_x, training):
        x = self.embedding_layer(input_x)
        x = self.rnn_layer(x, training=training)
        x = self.dense_layer(x)

        return x

# 日志记录
log_dir = os.path.join('Log_Dir', "word2vec", datetime.datetime.now().strftime("%Y-%m-%d"))
writer = tf.summary.create_file_writer(log_dir)


def train(xy_train, xy_val, num_classes, vocab_size, nbr_epoches, batch_size, embeddings_matrix):
    uniq_cfg_name = datetime.datetime.now().strftime("%Y")
    model_prefix = os.path.join(os.getcwd(), "word2vec-model", uniq_cfg_name)
    if not os.path.exists(model_prefix):
        print("create model dir: %s" % model_prefix)
        os.mkdir(model_prefix)

    model_path = model_prefix
    model = Model(num_classes, drop_rate=0.05, vocab_size=vocab_size, embedding_size=200, rnn_size=128, attention_size=128, embeddings_matrix=embeddings_matrix)

    optimizer = tf.keras.optimizers.Adam(5e-4)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    loss_metric = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(input_x, input_y, training=True):
        with tf.GradientTape() as tape:
            raw_prob = model(input_x, training)
            pred_loss = loss_fn(input_y, raw_prob)
        gradients = tape.gradient(pred_loss, model.trainable_variables)
        if training:
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_metric.update_state(pred_loss)
        return raw_prob

    for i in range(nbr_epoches):
        t0 = time.time()
        batch_train = batch_iter(xy_train[0], xy_train[1], batch_size=batch_size)
        loss_metric.reset_states()


        for batch_no, batch_tot, data_x, data_y in batch_train:
            predict_prob = train_step(data_x, data_y, True)
            one = tf.ones_like(batch_size)
            zero = tf.zeros_like(batch_size)
            y_pre = tf.where(predict_prob > 0.5, one, zero)

            proto_tensor = tf.make_tensor_proto(y_pre)
            y_pre = tf.make_ndarray(proto_tensor)
           
            if batch_no % 10 == 0:
                print("[%s]: %d [%s]: %0.4f  [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f" % (
                    "index",  batch_no,
                    "loss", loss_metric.result(),
                    "acc", binary_accuracy(y_pre, data_y),
                    "recall", binary_recall(y_pre, data_y),
                    "precision", binary_precision(y_pre, data_y),
                    "f1", binary_f_beta(y_pre, data_y)
                    ))

                with writer.as_default():
                    tf.summary.scalar('word2vec_loss', loss_metric.result(), step=batch_no + i*batch_tot)
                    tf.summary.scalar('word2vec_accuracy', binary_accuracy(y_pre, data_y), step=batch_no + i*batch_tot)


    model.save_weights(model_path, overwrite=True)

    # 所有 epoch 训练完后进行测试
    count = 0
    sumAcc = 0
    sumRecall = 0
    sumPrecision = 0
    sumF1 = 0

    loss_metric.reset_states()
    batch_test = batch_iter(xy_val[0], xy_val[1], batch_size=batch_size)
    for _, _, data_x, data_y in batch_test:
        raw_p = train_step(data_x, data_y, False)

        size = len(xy_val[1])
        one = tf.ones_like(size)
        zero = tf.zeros_like(size)
        y_pre = tf.where(raw_p > 0.5, one, zero)
        proto_tensor = tf.make_tensor_proto(y_pre)
        y_pre = tf.make_ndarray(proto_tensor)

        count += 1
        sumAcc += binary_accuracy(y_pre, data_y)
        sumPrecision += binary_precision(y_pre, data_y)
        sumRecall += binary_recall(y_pre, data_y)
        sumF1 += binary_f_beta(y_pre, data_y)

        print("[%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f" % (
            "acc", binary_accuracy(y_pre, data_y),
            "recall", binary_recall(y_pre, data_y),
            "precision", binary_precision(y_pre, data_y),
            "f1", binary_f_beta(y_pre, data_y)
            ))

    print("************************ test result *******************************")
    print("[%s]: %0.4f [%s]: %0.4f [%s]: %0.4f [%s]: %0.4f" % (
            "acc", sumAcc/count,
            "recall", sumRecall/count,
            "precision", sumPrecision/count,
            "f1", sumF1/count
    ))


if __name__ == "__main__":
    try:
        cur_dir = os.getcwd()
        max_seq_len = BertBilstmConfig["max_seq_len"]
        num_classes = BertBilstmConfig["num_classes"]
        word_dict = BertBilstmConfig["word_dict"]

        index_dict = BertBilstmConfig["index_dict"]
        train_sample_path01 = BertBilstmConfig["train_sample_path"]
        # train_sample_path = BertBilstmConfig["train_sample_path02"]
        # test_sample_path = BertBilstmConfig["test_sample_path02"]

        train_sample_path = BertBilstmConfig["Ree_train"]
        test_sample_path = BertBilstmConfig["Ree_test"]

        train_sample_pathMaD = BertBilstmConfig["train_sample_path_MaD"]
        vocab_list = BertBilstmConfig["vocab_list"]


        word2Vec = gensim.models.KeyedVectors.load_word2vec_format("word2vec/word2Vec.bin", binary=True)

        embeddings_matrix = create_embeddings_matrix(word2vec=word2Vec, vocab_size= len(vocab_list), vocab_list= vocab_list, word_dict=word_dict)
        train_x, train_y = load_sample_ori(train_sample_path, max_seq_len, word_dict)
        test_x, test_y = load_sample_ori(test_sample_path, max_seq_len, word_dict)
        train([train_x, train_y], [test_x, test_y], num_classes, len(vocab_list), nbr_epoches=3, batch_size=64, embeddings_matrix=embeddings_matrix)
    except:
        traceback.print_exc()
