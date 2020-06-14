import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants,signature_constants
# from tensorflow.python.client import graph_util
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def func_ck2pb():
    trained_checkpoint_prefix = 'vae_40/vae-0'
    export_dir = os.path.join('save_model', '1')

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
        loader.restore(sess, trained_checkpoint_prefix)

        # Export checkpoint to SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        x = tf.placeholder(tf.float32, [1, 120, 200, 1])
        y = tf.placeholder(tf.float32, [1, 120, 200, 1])
        inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
        outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs)
        signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING] )
        # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'test_signature': None})
        # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], strip_default_attrs=True)
        # builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map={'vae_image':signature})
        builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], signature_def_map=signature_map)
        builder.save()
        print("covnert from ck2 to saved pb model successfully!")


def func_pb2tflite():
    x = tf.placeholder(tf.float32, [1, 120, 200, 1])  # Placeholder:0
    y = tf.placeholder(tf.float32, [1, 120, 200, 1])  # Placeholder_1:0
    inputs = {'input': tf.saved_model.utils.build_tensor_info(x)}
    outputs = {'output': tf.saved_model.utils.build_tensor_info(y)}
    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs)
    signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

    converter = tf.contrib.lite.TFLiteConverter.from_saved_model("save_model/1",signature_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
    # converter.optimizations = [tf.contrib.lite.TFLiteConverter.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    open("vae.tflite", "wb").write(tflite_quant_model)

    converter.post_training_quantize = True
    tflite_quantized_model = converter.convert()
    open("quantized_vae.tflite", "wb").write(tflite_quantized_model)
    print("covnert from pb to tflite model successfully!")

if __name__ == "__main__":
    func_ck2pb()
    func_pb2tflite()

