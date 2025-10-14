import tensorflow as tf
p = r"D:/Intellimedicare/Brain_tumor/model_saved"
obj = tf.saved_model.load(p)
print("Available signatures:", list(obj.signatures.keys()))
sig = obj.signatures["serving_default"]
print("INPUT:", sig.structured_input_signature)
print("OUTPUTS:", sig.structured_outputs)
