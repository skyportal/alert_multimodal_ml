import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import register_keras_serializable

# Définir les composants du modèle sans TFMOT
class ConvEmbedding(layers.Layer):
    def __init__(self, num_filters, kernel_size=1, activation='relu', **kwargs):
        super(ConvEmbedding, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv1d = layers.Conv1D(
            filters=self.num_filters, kernel_size=self.kernel_size, activation=self.activation, padding='same'
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
        })
        return config

    def call(self, inputs):
        return self.conv1d(inputs)
    
class PositionalEncoding(layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super(PositionalEncoding, self).__init__(dtype=dtype, **kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims

        if max_dims % 2 == 1:
            max_dims += 1
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(np.float32))

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_steps": self.max_steps,
            "max_dims": self.max_dims,
        })
        return config

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[1], :shape[2]]
    
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
    
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output
    
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
        })
        return config

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class ImageProcessor(tf.keras.layers.Layer):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.conv1 = layers.Conv2D(32, (5, 5), activation='relu', padding="same")
        self.conv2 = layers.Conv2D(32, (5, 5), activation='relu', padding="same")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.drop1 = layers.Dropout(0.50)
        self.conv3 = layers.Conv2D(64, (5, 5), activation='relu', padding="same")
        self.conv4 = layers.Conv2D(64, (5, 5), activation='relu', padding="same")
        self.pool2 = layers.MaxPooling2D(pool_size=(4, 4))
        self.drop2 = layers.Dropout(0.55)
        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return self.flatten(x)

class MetadataProcessor(tf.keras.layers.Layer):
    def __init__(self):
        super(MetadataProcessor, self).__init__()
        self.batch_norm = layers.BatchNormalization()
        self.dense1 = layers.Dense(128, activation='relu')
        self.drop1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(128, activation='relu')

    def call(self, inputs):
        y = self.batch_norm(inputs)
        y = self.dense1(y)
        y = self.drop1(y)
        return self.dense2(y)

class MultimodalModel(tf.keras.Model):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.image_processor = ImageProcessor()
        self.metadata_processor = MetadataProcessor()
        self.concat = layers.Concatenate()
        self.final_dense1 = layers.Dense(8, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.final_dense2 = layers.Dense(1)

    def call(self, inputs):
        images, metadata = inputs
        processed_images = self.image_processor(images)
        processed_metadata = self.metadata_processor(metadata)
        combined = self.concat([processed_images, processed_metadata])
        x = self.final_dense1(combined)
        x = self.dropout(x)
        return self.final_dense2(x)
    
class CustomMaskingLayer(layers.Layer):
    def call(self, inputs):
        mask = tf.reduce_any(inputs != 0, axis=-1)
        column_mask = tf.reduce_any(mask, axis=0)
        masked_photometry = tf.boolean_mask(inputs, column_mask, axis=1)
        return masked_photometry

@register_keras_serializable()
class T2Model(tf.keras.Model):
    def __init__(self, num_filters, num_classes, num_layers, d_model, num_heads, dff, input_shapes, rate=0.1):
        super(T2Model, self).__init__()
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.masking_layer = CustomMaskingLayer()
        self.embedding = ConvEmbedding(num_filters=self.num_filters, kernel_size=3, activation='relu')
        self.pos_encoding = PositionalEncoding(max_steps=input_shapes[0][0], max_dims=d_model)
        self.encoder_layers = [TransformerBlock(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.image_processor = ImageProcessor()
        self.metadata_processor = MetadataProcessor()

        self.multimodal = MultimodalModel()
        
        self.dropout = layers.Dropout(rate)
        self.concat = layers.Concatenate(axis=-1)
        self.dense1 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.final_output = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        photometry, metadata, image = inputs

        photometry = self.masking_layer(photometry)

        x = self.embedding(photometry)

        x = self.pos_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, training=training)

        photometry_output = layers.GlobalAveragePooling1D()(x)

        image_output = self.image_processor(image)
        metadata_output = self.metadata_processor(metadata)
        multi = self.multimodal([image, metadata])

        x = self.concat([photometry_output, multi])
        # # x = self.dense1(x)
        # # x = self.dropout(x, training=training)
        # x = self.dense2(x)
        x = self.dropout(x, training=training)
        return self.final_output(x)
    
    def build(self, input_shapes):
        photometry = tf.keras.Input(shape=input_shapes[0], name="photometry_input")
        metadata = tf.keras.Input(shape=input_shapes[1], name="metadata_input")
        images = tf.keras.Input(shape=input_shapes[2], name="image_input")
        _ = self.call([photometry, metadata, images])

from tensorflow.keras import layers, models, optimizers, regularizers

def MiniVGG(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    model = models.Model(inputs, x)
    return model


class T2_Image(tf.keras.Model):
    def __init__(self, input_shape):
        super(T2_Image, self).__init__()
        self.vgg16 = MiniVGG(input_shape)
        
        self.final_dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        images = inputs
        x = self.vgg16(images)

        # # Process photometry and metadata
        # photometry_x = self.photometry_dense(photometry)
        # metadata_x = self.metadata_dense(metadata)

        # Concatenate all features
        # x = self.concat([x, photometry_x, metadata_x])
        return self.final_dense(x)

    def build(self, input_shapes):
        images = tf.keras.Input(shape=input_shapes, name="image_input")
        _ = self.call(images)
