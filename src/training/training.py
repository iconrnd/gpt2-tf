import tensorflow as tf
import os
from config.config import global_config
from utils.utils import Context, MyLRSchedule
from model.gpt2 import GPT
from data.dataset import get_datasets


def build_model(model_config):
    return GPT(model_config)


def get_model_and_ctx(model_config, restore=False):
    ctx = Context()
    ctx.strategy = tf.distribute.MirroredStrategy()
    ctx.step = 0
    ctx.best_val_loss = global_config.best_val_loss

    with ctx.strategy.scope():
        ctx.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)
        ctx.val_loss = tf.keras.metrics.Mean(name='test_loss')
        ctx.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        ctx.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_accuracy')

        global_batch_size = global_config.batch_per_replica * ctx.strategy.num_replicas_in_sync

        train_set, valid_set = get_datasets(global_batch_size, model_config)
        ctx.train_set_dist = ctx.strategy.experimental_distribute_dataset(train_set)
        ctx.valid_set_dist = ctx.strategy.experimental_distribute_dataset(valid_set.take(global_config.eval_iters))

        model = build_model(model_config)

        # This is how callbacks can be incorporated with custom loop
        # We use custom summary writer instead
        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=global_config.log_dir)
        # tb_callback.set_model(model)
        # ctx.callbacks = tf.keras.callbacks.CallbackList([
        #     tb_callback
        # ])

        ctx.writer = tf.summary.create_file_writer(str(global_config.log_dir))

        ctx.optimizer = tf.keras.optimizers.AdamW(learning_rate=MyLRSchedule(global_config.learning_rate,
                                                                             global_config.warmup_iters,
                                                                             global_config.min_lr,
                                                                             global_config.lr_decay_iters))
        ctx.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(ctx.optimizer)

        ctx.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                       optimizer=ctx.optimizer,
                                       model=model,
                                       best_val_loss=tf.Variable(global_config.best_val_loss),
                                       train_accuracy=ctx.train_accuracy,
                                       val_accuracy=ctx.val_accuracy,
                                       val_loss=ctx.val_loss)

        ctx.manager = tf.train.CheckpointManager(ctx.ckpt,
                                                 global_config.checkpoint_directory,
                                                 max_to_keep=3)

        if restore:
            ctx.ckpt.restore(ctx.manager.latest_checkpoint)
            # ckpt.restore(manager.checkpoints[-2])
            ctx.step = int(ctx.ckpt.step.value().numpy())
            ctx.best_val_loss = float(ctx.ckpt.best_val_loss.value().numpy())
            model = ctx.ckpt.model
            ctx.optimizer = ctx.ckpt.optimizer
            ctx.train_accuracy = ctx.ckpt.train_accuracy
            ctx.val_accuracy = ctx.ckpt.val_accuracy
            ctx.val_loss = ctx.ckpt.val_loss

    return model, ctx


@tf.function
def compute_loss(Y, logits, model_losses, ctx):
    per_example_loss = ctx.loss_object(tf.reshape(Y, [-1]),
                                       tf.reshape(logits, [-1, logits.shape[-1]]))
    loss = tf.nn.compute_average_loss(per_example_loss)
    if model_losses:
        loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    return loss


@tf.function
def train_step(sample, model, ctx):
    X, Y = sample
    with tf.GradientTape() as tape:
        logits = model(X, training=True)
        loss = compute_loss(Y, logits, model.losses, ctx)
        scaled_loss = ctx.optimizer.get_scaled_loss(loss)

    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = ctx.optimizer.get_unscaled_gradients(scaled_gradients)
    gradients = [tf.clip_by_value(gradient,
                 clip_value_min=-global_config.grad_clip,
                 clip_value_max=global_config.grad_clip) for gradient in gradients]
    ctx.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ctx.train_accuracy.update_state(tf.reshape(Y, [-1]),
                                    tf.reshape(logits, [-1, logits.shape[-1]]))
    return loss, ctx.optimizer.learning_rate


@tf.function
def val_step(sample, model, ctx):
    X, Y = sample
    logits = model(X, training=False)
    v_loss = ctx.loss_object(tf.reshape(Y, [-1]),
                             tf.reshape(logits, [-1, logits.shape[-1]]))
    ctx.val_loss.update_state(v_loss)
    ctx.val_accuracy(tf.reshape(Y, [-1]),
                     tf.reshape(logits, [-1, logits.shape[-1]]))


def distributed_train_epoch(step, model, ctx):
    total_loss = 0.0
    num_batches = 0
    for sample in ctx.train_set_dist:
        # ctx.callbacks.on_train_batch_begin(step)
        per_replica_losses, per_replica_lr = ctx.strategy.run(train_step, args=(sample, model, ctx, ))
        # ctx.callbacks.on_train_batch_end(step)
        total_loss += ctx.strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None)
        num_batches += 1
        lr_mean = ctx.strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_lr,
            axis=None)
        with ctx.writer.as_default():
            tf.summary.scalar('Learning Rate', lr_mean, step=step)

    return total_loss / float(num_batches)


def distributed_val_epoch(step, model, ctx):
    for sample in ctx.valid_set_dist:
        # ctx.callbacks.on_test_batch_begin(step)
        ctx.strategy.run(val_step, args=(sample, model, ctx, ))
        # ctx.callbacks.on_test_batch_end(step)


def training(model, ctx):
    step = ctx.step
    # ctx.callbacks.on_train_begin()

    while True:
        # ctx.callbacks.on_epoch_begin(step)

        # Train Epoch
        train_loss = distributed_train_epoch(step, model, ctx)

        # Val Epoch
        distributed_val_epoch(step, model, ctx)
        # ctx.callbacks.on_epoch_end(step)
        out_format = ("Epoch {}\nLoss: {}, Accuracy: {}\nVal Loss: {}, Val Accuracy: {}")
        print(out_format.format(step, train_loss, ctx.train_accuracy.result() * 100,
              ctx.val_loss.result(), ctx.val_accuracy.result() * 100))

        with ctx.writer.as_default():
            tf.summary.scalar('Epoch', step, step=step)
            tf.summary.scalar('Loss', train_loss, step=step)
            tf.summary.scalar('Accuracy', ctx.train_accuracy.result() * 100, step=step)
            tf.summary.scalar('Val Loss', ctx.val_loss.result(), step=step)
            tf.summary.scalar('Val Accuracy', ctx.val_accuracy.result() * 100, step=step)

        # Checkpointing
        if step > 0:
            if ctx.val_loss.result() < ctx.best_val_loss or global_config.always_save_checkpoint:
                ctx.best_val_loss = ctx.val_loss.result()
                print(f'Saving checkpoint to {os.path.join(global_config.checkpoint_directory, "ckpt")}')
                ctx.ckpt.step.assign(step)
                ctx.ckpt.best_val_loss.assign(ctx.best_val_loss)
                ctx.manager.save()

        ctx.train_accuracy.reset_states()
        ctx.val_accuracy.reset_states()
        ctx.val_loss.reset_states()

        step += 1
        if step > global_config.max_iters:
            break

    # ctx.callbacks.on_train_end()
