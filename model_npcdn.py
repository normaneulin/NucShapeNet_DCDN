# model_npcdn.py
# Hybrid Dual-Stream CNN for Nucleosome Positioning
# 
# NO Co-DeepNet logic — just a single CNN with two parallel streams:
#   Stream 1: Dilated Conv2D (global sequence patterns)
#   Stream 2: DenseNet-121 Conv2D (local motifs)
#   
# Input: (145, 12) → reshaped to (145, 12, 1) for Conv2D
# Output: Binary classification (nucleosomal vs linker)

import math
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="npcdn")
class DualStreamCNN(tf.keras.Model):
    """
    Single CNN with two parallel streams (no cooperative training).
    
    Stream 1: Dilated Conv2D with (3,12) kernels
    Stream 2: DenseNet-121 style with (3,3) kernels
    Fused → FC → sigmoid
    """

    def __init__(
        self,
        # Dilated stream
        dilated_filters:    int   = 48,
        dilation_rates:     tuple = (1, 2, 4, 8, 16),
        
        # DenseNet stream
        ds_initial_filters: int   = 64,
        ds_growth_rate:     int   = 24,
        ds_num_blocks:      int   = 4,
        ds_layers_per_block: tuple = (3, 6, 12, 8),
        ds_compression:     float = 0.5,
        
        # Shared
        dropout_rate: float = 0.2,
        l2_reg:       float = 5e-5,
        
        # FC
        fc_units:     int   = 256,
        output_dim:   int   = 1,
        **kwargs,
    ):
        kwargs.pop("name",        None)
        kwargs.pop("trainable",   None)
        kwargs.pop("dtype",       None)
        kwargs.pop("input_shape", None)

        super(DualStreamCNN, self).__init__(name="DualStreamCNN", **kwargs)

        self._init_args = dict(
            dilated_filters      = dilated_filters,
            dilation_rates       = tuple(dilation_rates),
            ds_initial_filters   = ds_initial_filters,
            ds_growth_rate       = ds_growth_rate,
            ds_num_blocks        = ds_num_blocks,
            ds_layers_per_block  = tuple(ds_layers_per_block),
            ds_compression       = ds_compression,
            dropout_rate         = dropout_rate,
            l2_reg               = l2_reg,
            fc_units             = fc_units,
            output_dim           = output_dim,
        )

        reg = tf.keras.regularizers.l2(l2_reg)

        # Reshape (145,12) → (145,12,1)
        self.reshape_input = tf.keras.layers.Reshape((145, 12, 1))

        # ══════════════════════════════════════════════════════════════════
        # STREAM 1: Dilated Conv2D
        # ══════════════════════════════════════════════════════════════════
        drates = list(dilation_rates)
        n_dil  = len(drates)

        self.dil_bns   = [tf.keras.layers.BatchNormalization(name=f"dil_bn{i+1}")
                          for i in range(n_dil)]
        self.dil_acts  = [tf.keras.layers.Activation("relu", name=f"dil_relu{i+1}")
                          for i in range(n_dil)]
        self.dil_convs = [tf.keras.layers.Conv2D(
                              dilated_filters, kernel_size=(3, 12),
                              dilation_rate=(drates[i], 1), padding="same",
                              use_bias=False, kernel_regularizer=reg,
                              name=f"dil_conv{i+1}")
                          for i in range(n_dil)]
        self.dil_drops = [tf.keras.layers.Dropout(dropout_rate, name=f"dil_drop{i+1}")
                          for i in range(n_dil)]
        self.dil_gap   = tf.keras.layers.GlobalAveragePooling2D(name="dil_gap")

        # ══════════════════════════════════════════════════════════════════
        # STREAM 2: DenseNet-121 Conv2D
        # ══════════════════════════════════════════════════════════════════

        # Initial conv + pool
        self.ds_init_conv = tf.keras.layers.Conv2D(
            ds_initial_filters, kernel_size=(7, 7), strides=(2, 2),
            padding="same", use_bias=False, kernel_regularizer=reg,
            name="ds_init_conv")
        self.ds_init_bn  = tf.keras.layers.BatchNormalization(name="ds_init_bn")
        self.ds_init_act = tf.keras.layers.Activation("relu", name="ds_init_relu")
        self.ds_init_pool = tf.keras.layers.MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), padding="same", name="ds_init_pool")

        # Dense blocks + transitions
        self.db_blocks = []
        self.tr_layers = []

        for block_idx in range(ds_num_blocks):
            n_layers = ds_layers_per_block[block_idx]
            
            block = []
            for layer_idx in range(n_layers):
                bn   = tf.keras.layers.BatchNormalization(
                    name=f"db{block_idx+1}_l{layer_idx+1}_bn")
                act  = tf.keras.layers.Activation(
                    "relu", name=f"db{block_idx+1}_l{layer_idx+1}_relu")
                conv = tf.keras.layers.Conv2D(
                    ds_growth_rate, kernel_size=(3, 3), padding="same",
                    use_bias=False, kernel_regularizer=reg,
                    name=f"db{block_idx+1}_l{layer_idx+1}_conv")
                drop = tf.keras.layers.Dropout(
                    dropout_rate, name=f"db{block_idx+1}_l{layer_idx+1}_drop")
                cat  = tf.keras.layers.Concatenate(
                    axis=-1, name=f"db{block_idx+1}_l{layer_idx+1}_cat")
                block.append((bn, act, conv, drop, cat))
            self.db_blocks.append(block)

            # Transition (except last block)
            if block_idx < ds_num_blocks - 1:
                tr_bn   = tf.keras.layers.BatchNormalization(name=f"tr{block_idx+1}_bn")
                tr_act  = tf.keras.layers.Activation("relu", name=f"tr{block_idx+1}_relu")
                tr_drop = tf.keras.layers.Dropout(dropout_rate, name=f"tr{block_idx+1}_drop")
                tr_pool = tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 1), name=f"tr{block_idx+1}_pool")
                self.tr_layers.append((tr_bn, tr_act, tr_drop, tr_pool))

        self.ds_gap = tf.keras.layers.GlobalAveragePooling2D(name="ds_gap")

        self._ds_compression = ds_compression
        self._tr_convs = []  # Created dynamically in call()
        self._reg = reg

        # ══════════════════════════════════════════════════════════════════
        # FC CLASSIFIER (simple, no knowledge transfer)
        # ══════════════════════════════════════════════════════════════════
        self.fc1      = tf.keras.layers.Dense(
            fc_units, activation="relu", kernel_regularizer=reg, name="fc1")
        self.fc1_drop = tf.keras.layers.Dropout(dropout_rate, name="fc1_drop")
        
        self.fc2      = tf.keras.layers.Dense(
            int(fc_units * 0.7), activation="relu", kernel_regularizer=reg, name="fc2")
        self.fc2_drop = tf.keras.layers.Dropout(dropout_rate, name="fc2_drop")
        
        self.out_layer = tf.keras.layers.Dense(
            output_dim, activation="sigmoid", name="output")

    # ─────────────────────────────────────────────────────────────────────────
    def call(self, x, training=False):
        """
        Simple forward pass — no knowledge transfer, no feature map return.
        
        Args:
            x: (batch, 145, 12)
            training: bool
            
        Returns:
            (batch, 1) — sigmoid probabilities
        """
        # Reshape to 2D image
        x = self.reshape_input(x)
        # print("[DEBUG] After reshape_input:", x.shape)

        # ── Stream 1: Dilated Conv2D ─────────────────────────────────────
        z = x
        for i, (bn, act, conv, drop) in enumerate(zip(
                self.dil_bns, self.dil_acts, self.dil_convs, self.dil_drops)):
            z = bn(z,   training=training)
            z = act(z)
            z = conv(z)
            print(f"[DEBUG] After dil_conv{i+1}:", z.shape)
            z = drop(z, training=training)
        dil_out = self.dil_gap(z)  # (batch, 48)
        # print("[DEBUG] After dil_gap:", dil_out.shape)

        # ── Stream 2: DenseNet Conv2D ─────────────────────────────────────
        d = self.ds_init_conv(x)
        # print("[DEBUG] After ds_init_conv:", d.shape)
        d = self.ds_init_bn(d,  training=training)
        d = self.ds_init_act(d)
        d = self.ds_init_pool(d)
        # print("[DEBUG] After ds_init_pool:", d.shape)

        for block_idx, block in enumerate(self.db_blocks):
            # Dense block
            for layer_idx, (bn, act, conv, drop, cat) in enumerate(block):
                h = bn(d,   training=training)
                h = act(h)
                h = conv(h)
                # print(f"[DEBUG] After db{block_idx+1}_l{layer_idx+1}_conv:", h.shape)
                h = drop(h, training=training)
                d = cat([d, h])
                # print(f"[DEBUG] After db{block_idx+1}_l{layer_idx+1}_cat:", d.shape)

            # Transition
            if block_idx < len(self.db_blocks) - 1:
                tr_bn, tr_act, tr_drop, tr_pool = self.tr_layers[block_idx]
                # Create transition conv on first call
                if len(self._tr_convs) <= block_idx:
                    in_ch = d.shape[-1]
                    out_ch = max(1, math.floor(in_ch * self._ds_compression))
                    tr_conv = tf.keras.layers.Conv2D(
                        out_ch, kernel_size=(1, 1), padding="same",
                        use_bias=False, kernel_regularizer=self._reg,
                        name=f"tr{block_idx+1}_conv")
                    self._tr_convs.append(tr_conv)
                else:
                    tr_conv = self._tr_convs[block_idx]

                d = tr_bn(d,   training=training)
                d = tr_act(d)
                d = tr_conv(d)
                # print(f"[DEBUG] After tr{block_idx+1}_conv:", d.shape)
                d = tr_drop(d, training=training)
                d = tr_pool(d)
                # print(f"[DEBUG] After tr{block_idx+1}_pool:", d.shape)

        ds_out = self.ds_gap(d)  # (batch, 389)
        # print("[DEBUG] After ds_gap:", ds_out.shape)

        # ── Fusion ────────────────────────────────────────────────────────
        fused = tf.concat([dil_out, ds_out], axis=-1)
        # print("[DEBUG] After fusion:", fused.shape)

        # ── FC classifier ─────────────────────────────────────────────────
        z = self.fc1(fused, training=training)
        z = self.fc1_drop(z, training=training)
        z = self.fc2(z,      training=training)
        z = self.fc2_drop(z, training=training)
        # print("[DEBUG] After FC:", z.shape)
        
        out = self.out_layer(z, training=training)
        # print("[DEBUG] After out_layer:", out.shape)
        return out

    # ─────────────────────────────────────────────────────────────────────────
    def get_config(self):
        cfg = dict(self._init_args)
        cfg["dilation_rates"] = list(cfg["dilation_rates"])
        cfg["ds_layers_per_block"] = list(cfg["ds_layers_per_block"])
        return cfg

    @classmethod
    def from_config(cls, config):
        allowed = {
            "dilated_filters", "dilation_rates",
            "ds_initial_filters", "ds_growth_rate", "ds_num_blocks",
            "ds_layers_per_block", "ds_compression",
            "dropout_rate", "l2_reg", "fc_units", "output_dim",
        }
        clean = {k: v for k, v in config.items() if k in allowed}
        if "dilation_rates" in clean:
            clean["dilation_rates"] = tuple(clean["dilation_rates"])
        if "ds_layers_per_block" in clean:
            clean["ds_layers_per_block"] = tuple(clean["ds_layers_per_block"])
        return cls(**clean)


# ──────────────────────────────────────────────────────────────────────────────
def build_model(input_shape=(145, 12), **kwargs):
    """Build a single dual-stream model."""
    model = DualStreamCNN(**kwargs)
    
    # Warm-up call to build all weights
    dummy_x = tf.zeros((2,) + input_shape)
    _ = model(dummy_x, training=False)
    
    print(f"[INFO] Model built: {len(model.trainable_variables)} trainable tensors")
    return model


if __name__ == "__main__":
    print("Building Dual-Stream CNN...")
    model = build_model()
    model.summary()
    model.save("DualStreamCNN.keras")
    print("Saved DualStreamCNN.keras")