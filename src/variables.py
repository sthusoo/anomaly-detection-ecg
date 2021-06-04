import tensorflow as tf
import numpy as np
import src.training as tr

model = tf.keras.models.load_model('autoencoder_model')
# Normal
encoder_out = model.encoder(tr.normal_test_data).numpy()
decoder_out = model.decoder(encoder_out).numpy()

# Anomaly
encoder_out_a = model.encoder(tr.anomaly_test_data).numpy()
decoder_out_a = model.decoder(encoder_out_a).numpy()

# Reconstruction - normal data
reconstructions = model.predict(tr.normal_test_data)
reconstruction_loss = tf.keras.losses.mae(reconstructions, tr.normal_test_data)

# Reconstruction - anomaly data
reconstructions_a = model.predict(tr.anomaly_test_data)
reconstruction_loss_a = tf.keras.losses.mae(reconstructions_a, tr.anomaly_test_data)

threshold = np.mean(reconstruction_loss) + 2*np.std(reconstruction_loss) 

# Normal
preds = tf.math.less(reconstruction_loss, threshold)
actual_normal_count = preds.shape
pred_normal_count = tf.math.count_nonzero(preds)

# Anomaly
preds_a = tf.math.greater(reconstruction_loss_a, threshold)
actual_anomaly_count = preds_a.shape
pred_anomaly_count = tf.math.count_nonzero(preds_a)

norm_accuracy_score = str(round((pred_normal_count.numpy() / actual_normal_count[0])*100, 2)) + '%'
anomaly_accuracy_score = str(round((pred_anomaly_count.numpy() / actual_anomaly_count[0])*100, 2)) + '%'
eval_data = [ [pred_normal_count.numpy(), pred_anomaly_count.numpy()], [actual_normal_count[0], actual_anomaly_count[0]], [norm_accuracy_score, anomaly_accuracy_score] ]

# Guessing
## 1
train_loss_1 = tf.keras.losses.mae(reconstructions[5], tr.normal_test_data[5])
is_normal_1 = tf.math.less(train_loss_1, threshold).numpy()
## 2
train_loss_2 = tf.keras.losses.mae(reconstructions[20], tr.anomaly_test_data[20])
is_normal_2 = tf.math.less(train_loss_2, threshold).numpy()
## 3
train_loss_3 = tf.keras.losses.mae(reconstructions[6], tr.anomaly_test_data[6])
is_normal_3 = tf.math.less(train_loss_3, threshold).numpy()