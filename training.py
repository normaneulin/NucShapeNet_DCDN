# training.py
# Standard training for Dual-Stream CNN (no Co-DeepNet logic)

import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
import model_npcdn
import evaluator

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", dest="path", type=str,
                    default="data/DatasetNup_1/encoded_melanogaster")
parser.add_argument("-o", "--out",  dest="outPath", type=str, default="result")
parser.add_argument("-e", "--experiments", dest="exp",
                    default="Experiment_melanogaster")
parser.add_argument("-f", "--foldName", dest="foldName", default="folds.pickle")
parser.add_argument("-tmohn", "--tmohnuc",  dest="nuc_pickle",
                    type=str, default="three_mer_one_hot_nuc.pickle")
parser.add_argument("-tmohl", "--tmohlin",  dest="link_pickle",
                    type=str, default="three_mer_one_hot_link.pickle")
args = parser.parse_args()

inPath   = args.path
outPath  = args.outPath
expName  = args.exp
foldName = args.foldName

# ══════════════════════════════════════════════════════════════════════════════
# Hyper-parameters
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS       = 200
PATIENCE     = 20
BATCH_SIZE   = 64
LEARNING_RATE = 1e-3
K_FOLDS      = 10

model_name = "npcdn"
modelPath  = os.path.join(outPath, expName, "models", model_name)
foldPath   = os.path.join(outPath, expName, foldName)

os.makedirs(modelPath,                 exist_ok=True)
os.makedirs(os.path.dirname(foldPath), exist_ok=True)

print("[INFO] Loading encoded data...")

for p in (os.path.join(inPath, args.nuc_pickle),
          os.path.join(inPath, args.link_pickle)):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Pickle not found: {p}")

with open(os.path.join(inPath, args.nuc_pickle),  "rb") as fp:
    nuc_tmoh  = pickle.load(fp)
with open(os.path.join(inPath, args.link_pickle), "rb") as fp:
    link_tmoh = pickle.load(fp)

labels = np.concatenate([
    np.ones( (len(nuc_tmoh),  1), dtype=np.float32),
    np.zeros((len(link_tmoh), 1), dtype=np.float32),
], axis=0)

data = np.concatenate(
    [np.array(nuc_tmoh), np.array(link_tmoh)], axis=0
).astype(np.float32)

print(f"[INFO] data.shape: {data.shape}  labels.shape: {labels.shape}")

folds = evaluator.build_kfold(data, labels, k=K_FOLDS, shuffle=True, seed=42)
with open(foldPath, "wb") as fp:
    pickle.dump(folds, fp)
print(f"[INFO] {K_FOLDS}-fold splits saved")


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(fold_idx, fold):
    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx}  |  Dual-Stream CNN")
    print(f"  max_epochs={EPOCHS}  patience={PATIENCE}  batch_size={BATCH_SIZE}")
    print(f"{'='*70}")

    tf.keras.backend.clear_session()

    # Build model
    model = model_npcdn.build_model(input_shape=(145, 12))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            evaluator.acc,
            evaluator.precision,
            evaluator.recall,
            evaluator.f1score,
            evaluator.aucScore,
        ]
    )

    X_train = fold["X1_train"].astype(np.float32)
    y_train = fold["y_train"].astype(np.float32)
    X_val   = fold["X1_test"].astype(np.float32)
    y_val   = fold["y_test"].astype(np.float32)

    # Callbacks
    checkpoint_path = os.path.join(modelPath, f"npcdn_best-fold{fold_idx}.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n[FOLD {fold_idx} DONE]  Best val_loss={min(history.history['val_loss']):.4f}")
    print(f"  -> {checkpoint_path}")
    
    return history


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

for fold_idx, fold in enumerate(folds, start=1):
    train_fold(fold_idx, fold)

print("\n[INFO] All folds complete.")
print("Models saved to:", modelPath)