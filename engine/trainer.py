import datetime
import tensorflow as tf


def train(cfg, optimizer, dataset):

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = cfg.LOGGER.TB_OUTPUT_DIR + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for iteration, (x, y) in enumerate(dataset):
        loss = optimizer.step(x, y)
        train_loss(loss)

        if not (iteration % cfg.LOGGER.TB_PERIOD):
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=iteration)
